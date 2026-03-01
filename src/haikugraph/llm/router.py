"""LLM router with latency-aware routing, caching, and telemetry.

This module routes LLM calls across ollama/openai/anthropic providers and
captures per-call metrics used by runtime diagnostics and benchmarks.
"""

from __future__ import annotations

import contextvars
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from haikugraph.llm.ollama_client import ollama_chat


# Default models per provider
DEFAULT_MODELS = {
    "ollama": {
        "planner": "qwen2.5:14b-instruct",
        "narrator": "llama3.1:8b",
        "intent": "qwen2.5:14b-instruct",
    },
    "anthropic": {
        "planner": "claude-sonnet-4-6",
        "narrator": "claude-haiku-4-5-20251001",
        "intent": "claude-haiku-4-5-20251001",
    },
    "openai": {
        "planner": "gpt-4o",
        "narrator": "gpt-4o-mini",
        "intent": "gpt-4o-mini",
    },
}

# Legacy defaults for backward compatibility
DEFAULT_PLANNER_MODEL = "qwen2.5:14b-instruct"
DEFAULT_NARRATOR_MODEL = "llama3.1:8b"

# Model fallback chains — try in order if primary model fails
MODEL_FALLBACKS: dict[str, dict[str, list[str]]] = {
    "ollama": {
        "planner": [
            "qwen2.5:7b-instruct",
            "llama3.2:latest",
            "llama3.1:8b",
            "mistral:7b",
        ],
        "narrator": [
            "llama3.1:8b",
            "qwen2.5:7b-instruct",
            "llama3.2:latest",
        ],
        "intent": [
            "qwen2.5:7b-instruct",
            "llama3.2:latest",
            "llama3.1:8b",
        ],
    },
    "anthropic": {
        "planner": ["claude-sonnet-4-6", "claude-haiku-4-5-20251001"],
        "narrator": ["claude-haiku-4-5-20251001", "claude-sonnet-4-6"],
        "intent": ["claude-haiku-4-5-20251001", "claude-sonnet-4-6"],
    },
    "openai": {
        "planner": ["gpt-4o", "gpt-4o-mini"],
        "narrator": ["gpt-4o-mini", "gpt-4o"],
        "intent": ["gpt-4o-mini", "gpt-4o"],
    },
}

# Per-provider timeout defaults
PROVIDER_TIMEOUTS: dict[str, int] = {
    "anthropic": 30,
    "openai": 45,
    "ollama": 30,
}

_FAST_MODEL_DEFAULTS: dict[str, str] = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-haiku-4-5-20251001",
    "ollama": "qwen2.5:7b-instruct",
}

_ROLE_TIMEOUT_CAP_SECONDS: dict[str, int] = {
    "intent": 20,
    "planner": 25,
    "narrator": 40,
}


@dataclass
class _CacheEntry:
    value: str
    created_at: float
    hits: int = 0


_LLM_CACHE: OrderedDict[str, _CacheEntry] = OrderedDict()
_CACHE_LOCK = threading.RLock()
_METRICS_CTX: contextvars.ContextVar[list[dict[str, Any]] | None] = contextvars.ContextVar(
    "datada_llm_metrics",
    default=None,
)
_OVERLAY_LOCK = threading.RLock()
_OVERLAY_CACHE: dict[str, dict[str, Any]] = {}
_OVERLAY_CACHE_KEY: tuple[str, tuple[tuple[str, int], ...]] | None = None


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return int(default)
    try:
        return int(raw)
    except Exception:
        return int(default)


def _overlay_dir() -> Path:
    raw = os.environ.get("HG_PROVIDER_OVERLAY_DIR", "skills/providers")
    return Path(raw).expanduser()


def _load_provider_overlays() -> dict[str, dict[str, Any]]:
    root = _overlay_dir()
    if not root.exists() or not root.is_dir():
        return {}
    files = sorted(path for path in root.glob("*.json") if path.is_file())
    state_key = (
        str(root.resolve()),
        tuple((path.name, int(path.stat().st_mtime_ns)) for path in files),
    )
    with _OVERLAY_LOCK:
        global _OVERLAY_CACHE_KEY
        if _OVERLAY_CACHE_KEY == state_key:
            return dict(_OVERLAY_CACHE)

    loaded: dict[str, dict[str, Any]] = {}
    for path in files:
        provider = path.stem.strip().lower()
        if not provider:
            continue
        try:
            parsed = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(parsed, dict):
            loaded[provider] = parsed

    with _OVERLAY_LOCK:
        _OVERLAY_CACHE.clear()
        _OVERLAY_CACHE.update(loaded)
        _OVERLAY_CACHE_KEY = state_key
        return dict(_OVERLAY_CACHE)


def _provider_overlay(provider: str) -> dict[str, Any]:
    return dict(_load_provider_overlays().get(provider, {}))


def _overlay_role_model(provider: str, role: str) -> str:
    overlay = _provider_overlay(provider)
    models = overlay.get("models")
    if isinstance(models, dict):
        model = str(models.get(role) or "").strip()
        if model:
            return model
    return ""


def _overlay_timeout(provider: str, role: str) -> int | None:
    overlay = _provider_overlay(provider)
    timeouts = overlay.get("timeouts")
    if not isinstance(timeouts, dict):
        return None
    role_timeout = timeouts.get(role)
    if role_timeout is not None:
        try:
            return max(5, int(role_timeout))
        except Exception:
            return None
    default_timeout = timeouts.get("default")
    if default_timeout is not None:
        try:
            return max(5, int(default_timeout))
        except Exception:
            return None
    return None


def _overlay_fast_model(provider: str) -> str:
    overlay = _provider_overlay(provider)
    policy = overlay.get("routing")
    if isinstance(policy, dict):
        fast = str(policy.get("fast_model") or "").strip()
        if fast:
            return fast
    return ""


def _provider_timeout(provider: str, default_timeout: int) -> int:
    overlay_timeout = _overlay_timeout(provider, "default")
    if overlay_timeout is not None:
        default_timeout = int(overlay_timeout)
    env_name = f"HG_{provider.upper()}_TIMEOUT_SECONDS"
    return max(5, _env_int(env_name, default_timeout))


def reset_llm_metrics() -> None:
    """Initialize request-scoped LLM telemetry context."""
    _METRICS_CTX.set([])


def get_llm_metrics(*, reset: bool = False) -> list[dict[str, Any]]:
    """Return request-scoped LLM telemetry rows."""
    metrics = _METRICS_CTX.get()
    out = list(metrics or [])
    if reset:
        _METRICS_CTX.set([])
    return out


def summarize_llm_metrics(metrics: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Produce compact summary for runtime payloads."""
    rows = list(metrics if metrics is not None else get_llm_metrics())
    if not rows:
        return {
            "calls": 0,
            "cache_hits": 0,
            "cache_hit_ratio": 0.0,
            "total_latency_ms": 0.0,
            "avg_latency_ms": 0.0,
            "providers": {},
        }
    total_latency = sum(float(r.get("latency_ms") or 0.0) for r in rows)
    cache_hits = sum(1 for r in rows if bool(r.get("cache_hit")))
    providers: dict[str, dict[str, Any]] = {}
    for row in rows:
        prov = str(row.get("provider") or "unknown")
        curr = providers.setdefault(
            prov,
            {
                "calls": 0,
                "total_latency_ms": 0.0,
                "cache_hits": 0,
                "models": {},
            },
        )
        curr["calls"] += 1
        curr["total_latency_ms"] += float(row.get("latency_ms") or 0.0)
        curr["cache_hits"] += 1 if bool(row.get("cache_hit")) else 0
        model_name = str(row.get("model_actual") or row.get("model_requested") or "")
        if model_name:
            curr["models"][model_name] = int(curr["models"].get(model_name, 0)) + 1
    for curr in providers.values():
        calls = max(1, int(curr["calls"]))
        curr["avg_latency_ms"] = round(float(curr["total_latency_ms"]) / calls, 2)
        curr["cache_hit_ratio"] = round(float(curr["cache_hits"]) / calls, 4)
        curr["total_latency_ms"] = round(float(curr["total_latency_ms"]), 2)
    calls = len(rows)
    return {
        "calls": calls,
        "cache_hits": cache_hits,
        "cache_hit_ratio": round(cache_hits / max(1, calls), 4),
        "total_latency_ms": round(total_latency, 2),
        "avg_latency_ms": round(total_latency / max(1, calls), 2),
        "providers": providers,
    }


def _append_metric(metric: dict[str, Any]) -> None:
    rows = _METRICS_CTX.get()
    if rows is None:
        rows = []
        _METRICS_CTX.set(rows)
    rows.append(metric)


def _cache_enabled() -> bool:
    return _env_bool("HG_LLM_CACHE_ENABLED", True)


def _cache_ttl_seconds() -> int:
    return max(1, _env_int("HG_LLM_CACHE_TTL_SECONDS", 300))


def _cache_max_entries() -> int:
    return max(32, _env_int("HG_LLM_CACHE_MAX_ENTRIES", 512))


def _stable_messages_repr(messages: list[dict[str, str]]) -> str:
    clean = [
        {
            "role": str(m.get("role") or "").strip(),
            "content": str(m.get("content") or ""),
        }
        for m in messages
    ]
    return json.dumps(clean, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def _cache_key(
    *,
    provider: str,
    role: str,
    model: str,
    temperature: float,
    max_tokens: int | None,
    messages: list[dict[str, str]],
) -> str:
    payload = (
        f"{provider}|{role}|{model}|{temperature:.4f}|{max_tokens or 0}|"
        + _stable_messages_repr(messages)
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _cache_get(key: str) -> tuple[str, int] | None:
    ttl = _cache_ttl_seconds()
    now = time.time()
    with _CACHE_LOCK:
        # Prune stale entries opportunistically.
        stale_keys = [k for k, v in _LLM_CACHE.items() if now - v.created_at > ttl]
        for stale in stale_keys:
            _LLM_CACHE.pop(stale, None)

        entry = _LLM_CACHE.get(key)
        if entry is None:
            return None
        if now - entry.created_at > ttl:
            _LLM_CACHE.pop(key, None)
            return None
        entry.hits += 1
        _LLM_CACHE.move_to_end(key)
        return entry.value, entry.hits


def _cache_set(key: str, value: str) -> None:
    with _CACHE_LOCK:
        _LLM_CACHE[key] = _CacheEntry(value=value, created_at=time.time(), hits=0)
        _LLM_CACHE.move_to_end(key)
        max_entries = _cache_max_entries()
        while len(_LLM_CACHE) > max_entries:
            _LLM_CACHE.popitem(last=False)


def _should_cache(role: str, temperature: float) -> bool:
    if not _cache_enabled():
        return False
    if temperature > 0.25:
        return False
    if role == "narrator" and not _env_bool("HG_LLM_CACHE_NARRATOR", False):
        return False
    return True


def _infer_complexity(messages: list[dict[str, str]]) -> str:
    user_text = ""
    for msg in reversed(messages):
        if str(msg.get("role") or "").lower() == "user":
            user_text = str(msg.get("content") or "")
            break
    if not user_text:
        return "moderate"
    lower = user_text.lower()
    tokens = [t for t in lower.replace("\n", " ").split(" ") if t]
    token_count = len(tokens)
    hard_signals = [
        "join",
        "compare",
        "correlat",
        "hypothesis",
        "why",
        "root cause",
        "what if",
        "multi",
        "across",
        "decompose",
        "regression",
    ]
    if any(sig in lower for sig in hard_signals):
        return "complex"
    if token_count <= 22:
        return "simple"
    if token_count >= 70:
        return "complex"
    return "moderate"


def _maybe_route_fast_model(provider: str, role: str, model: str, complexity: str) -> tuple[str, str]:
    if not _env_bool("HG_ADAPTIVE_MODEL_ROUTING", True):
        return model, "adaptive_disabled"
    if role not in {"intent", "planner"}:
        return model, "role_excluded"
    if complexity != "simple":
        return model, "complexity_not_simple"

    env_name = f"HG_FAST_{provider.upper()}_MODEL"
    fast_model = os.environ.get(env_name) or _overlay_fast_model(provider) or _FAST_MODEL_DEFAULTS.get(provider)
    fast_model = str(fast_model or "").strip()
    if not fast_model or fast_model == model:
        return model, "fast_model_unset_or_same"
    return fast_model, "downshift_for_simple_query"


def _candidate_models(provider: str, role: str, primary_model: str) -> list[str]:
    models = [str(primary_model or "").strip()]
    overlay = _provider_overlay(provider)
    overlay_fallbacks = None
    policy = overlay.get("routing")
    if isinstance(policy, dict):
        fallback_map = policy.get("fallback_models")
        if isinstance(fallback_map, dict):
            overlay_fallbacks = fallback_map.get(role)
    if isinstance(overlay_fallbacks, list):
        for candidate in overlay_fallbacks:
            c = str(candidate or "").strip()
            if c and c not in models:
                models.append(c)
    if not _env_bool("HG_ENABLE_MODEL_FALLBACK", True):
        return [m for m in models if m]
    for candidate in MODEL_FALLBACKS.get(provider, {}).get(role, []):
        c = str(candidate or "").strip()
        if c and c not in models:
            models.append(c)
    return [m for m in models if m]


def _bounded_timeout(provider: str, role: str, timeout_seconds: int) -> int:
    if not _env_bool("HG_BOUNDED_LATENCY_ENABLED", True):
        return int(timeout_seconds)
    default_cap = _ROLE_TIMEOUT_CAP_SECONDS.get(role, timeout_seconds)
    cap = _env_int(
        f"HG_{provider.upper()}_{role.upper()}_MAX_SECONDS",
        _env_int(
            f"HG_{provider.upper()}_MAX_SECONDS",
            _env_int("HG_LLM_MAX_SECONDS", default_cap),
        ),
    )
    return max(5, min(int(timeout_seconds), int(cap)))


def _call_anthropic(
    messages: list[dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int | None,
    timeout: int,
) -> tuple[str, dict[str, Any]]:
    """Call Anthropic API (Claude models)."""
    try:
        import anthropic
    except ImportError:
        raise ImportError(
            "anthropic package not installed. Install with: pip install anthropic"
        )

    api_key = os.environ.get("HG_ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "Anthropic API key not found. Set HG_ANTHROPIC_API_KEY or ANTHROPIC_API_KEY."
        )

    client = anthropic.Anthropic(api_key=api_key, timeout=timeout)
    system_content = None
    api_messages: list[dict[str, str]] = []
    for msg in messages:
        if str(msg.get("role") or "") == "system":
            system_content = str(msg.get("content") or "")
        else:
            api_messages.append(
                {
                    "role": str(msg.get("role") or "user"),
                    "content": str(msg.get("content") or ""),
                }
            )

    response = client.messages.create(
        model=model,
        max_tokens=max_tokens or 4096,
        temperature=temperature,
        system=system_content or "You are a helpful data assistant.",
        messages=api_messages,
    )

    chunks: list[str] = []
    for item in getattr(response, "content", []) or []:
        txt = getattr(item, "text", None)
        if txt:
            chunks.append(str(txt))
    text = "".join(chunks).strip()
    if not text and chunks:
        text = str(chunks[0])
    usage = getattr(response, "usage", None)
    meta = {
        "model": str(getattr(response, "model", model) or model),
        "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
        "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
    }
    meta["total_tokens"] = int(meta["input_tokens"] + meta["output_tokens"])
    return text, meta


def _call_openai(
    messages: list[dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int | None,
    timeout: int,
) -> tuple[str, dict[str, Any]]:
    """Call OpenAI API (GPT models)."""
    try:
        import importlib

        openai_module = importlib.import_module("openai")
    except ImportError:
        raise ImportError(
            "openai package not installed. Install with: pip install openai"
        )

    api_key = os.environ.get("HG_OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OpenAI API key not found. Set HG_OPENAI_API_KEY or OPENAI_API_KEY."
        )

    client = openai_module.OpenAI(api_key=api_key, timeout=timeout)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens or 4096,
    )

    message = response.choices[0].message
    text = str(getattr(message, "content", "") or "")
    usage = getattr(response, "usage", None)
    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    total_tokens = int(getattr(usage, "total_tokens", 0) or (prompt_tokens + completion_tokens))
    meta = {
        "model": str(getattr(response, "model", model) or model),
        "input_tokens": prompt_tokens,
        "output_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }
    return text, meta


def call_llm(
    messages: list[dict[str, str]],
    *,
    role: str = "planner",
    max_tokens: int | None = None,
    timeout: int = 60,
    provider: str | None = None,
    model: str | None = None,
    temperature_override: float | None = None,
) -> str:
    """Route LLM call to appropriate model based on role and provider."""
    resolved_provider = (provider or os.environ.get("HG_LLM_PROVIDER", "ollama")).lower()
    if role not in {"planner", "intent", "narrator"}:
        raise ValueError(f"Invalid role: {role}. Must be planner/intent/narrator.")

    overlay_default_model = _overlay_role_model(resolved_provider, role)
    default_model = overlay_default_model or DEFAULT_MODELS.get(resolved_provider, {}).get(role, DEFAULT_PLANNER_MODEL)
    if role == "planner":
        role_model = os.environ.get("HG_PLANNER_MODEL", default_model)
        temperature = float(os.environ.get("HG_PLANNER_TEMPERATURE", "0"))
    elif role == "intent":
        role_model = os.environ.get("HG_INTENT_MODEL", default_model)
        temperature = float(os.environ.get("HG_INTENT_TEMPERATURE", "0"))
    else:
        role_model = os.environ.get("HG_NARRATOR_MODEL", default_model)
        temperature = float(os.environ.get("HG_NARRATOR_TEMPERATURE", "0.3"))
        timeout = max(timeout, 90)

    requested_model = str(model or role_model)
    if temperature_override is not None:
        temperature = float(temperature_override)

    effective_timeout = timeout
    if timeout == 60:
        effective_timeout = _provider_timeout(
            resolved_provider,
            PROVIDER_TIMEOUTS.get(resolved_provider, timeout),
        )
    overlay_role_timeout = _overlay_timeout(resolved_provider, role)
    if overlay_role_timeout is not None:
        effective_timeout = int(overlay_role_timeout)
    if role == "narrator":
        effective_timeout = max(effective_timeout, 45)
    effective_timeout = _bounded_timeout(resolved_provider, role, effective_timeout)

    complexity = _infer_complexity(messages)
    model_primary, routing_reason = _maybe_route_fast_model(
        resolved_provider,
        role,
        requested_model,
        complexity,
    )
    model_candidates = _candidate_models(resolved_provider, role, model_primary)
    if not model_candidates:
        raise ValueError(
            f"No model candidates for provider={resolved_provider}, role={role}"
        )

    cache_hit = False
    cache_key = ""
    cache_hits = 0
    if _should_cache(role, temperature):
        cache_key = _cache_key(
            provider=resolved_provider,
            role=role,
            model=model_primary,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=messages,
        )
        cached = _cache_get(cache_key)
        if cached is not None:
            cached_text, cache_hits = cached
            cache_hit = True
            _append_metric(
                {
                    "provider": resolved_provider,
                    "role": role,
                    "model_requested": requested_model,
                    "model_actual": model_primary,
                    "complexity": complexity,
                    "latency_ms": 0.0,
                    "cache_hit": True,
                    "cache_hits": cache_hits,
                    "routing_reason": routing_reason,
                    "temperature": round(float(temperature), 4),
                    "timeout_s": int(effective_timeout),
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "load_duration_ms": 0.0,
                    "prompt_eval_duration_ms": 0.0,
                    "eval_duration_ms": 0.0,
                }
            )
            return cached_text

    provider_meta: dict[str, Any] = {}
    text = ""
    model_selected = model_primary
    call_started = time.perf_counter()
    last_error: Exception | None = None
    for idx, candidate_model in enumerate(model_candidates):
        attempt_started = time.perf_counter()
        try:
            if resolved_provider == "anthropic":
                text, provider_meta = _call_anthropic(
                    messages,
                    candidate_model,
                    temperature,
                    max_tokens,
                    effective_timeout,
                )
            elif resolved_provider == "openai":
                text, provider_meta = _call_openai(
                    messages,
                    candidate_model,
                    temperature,
                    max_tokens,
                    effective_timeout,
                )
            elif resolved_provider == "ollama":
                text, provider_meta = ollama_chat(
                    messages,
                    model=candidate_model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=effective_timeout,
                    return_metadata=True,
                )
            else:
                raise ValueError(
                    f"Unsupported LLM provider: {resolved_provider}. Supported: ollama, anthropic, openai"
                )
            model_selected = candidate_model
            if idx > 0:
                routing_reason = f"{routing_reason}+fallback_model[{idx}]"
            break
        except Exception as exc:  # pragma: no cover - error path covered in integration tests
            last_error = exc
            _append_metric(
                {
                    "provider": resolved_provider,
                    "role": role,
                    "model_requested": requested_model,
                    "model_actual": candidate_model,
                    "complexity": complexity,
                    "latency_ms": round((time.perf_counter() - attempt_started) * 1000, 2),
                    "cache_hit": False,
                    "cache_hits": cache_hits,
                    "routing_reason": f"{routing_reason}+attempt_failed",
                    "temperature": round(float(temperature), 4),
                    "timeout_s": int(effective_timeout),
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "load_duration_ms": 0.0,
                    "prompt_eval_duration_ms": 0.0,
                    "eval_duration_ms": 0.0,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            if idx >= len(model_candidates) - 1:
                raise
            continue

    if last_error is not None and not text:
        raise last_error

    latency_ms = round((time.perf_counter() - call_started) * 1000, 2)
    load_duration_ms = float(provider_meta.get("load_duration_ns") or 0) / 1_000_000
    prompt_eval_duration_ms = float(provider_meta.get("prompt_eval_duration_ns") or 0) / 1_000_000
    eval_duration_ms = float(provider_meta.get("eval_duration_ns") or 0) / 1_000_000

    _append_metric(
        {
            "provider": resolved_provider,
            "role": role,
            "model_requested": requested_model,
            "model_actual": str(provider_meta.get("model") or model_selected),
            "complexity": complexity,
            "latency_ms": latency_ms,
            "cache_hit": cache_hit,
            "cache_hits": cache_hits,
            "routing_reason": routing_reason,
            "temperature": round(float(temperature), 4),
            "timeout_s": int(effective_timeout),
            "input_tokens": int(provider_meta.get("input_tokens") or 0),
            "output_tokens": int(provider_meta.get("output_tokens") or 0),
            "total_tokens": int(provider_meta.get("total_tokens") or 0),
            "load_duration_ms": round(load_duration_ms, 2),
            "prompt_eval_duration_ms": round(prompt_eval_duration_ms, 2),
            "eval_duration_ms": round(eval_duration_ms, 2),
        }
    )

    if cache_key and _should_cache(role, temperature) and model_selected == model_primary:
        _cache_set(cache_key, text)
    return text


def _has_module(module_name: str) -> bool:
    """Return True when a module is installed in the current environment."""
    return importlib.util.find_spec(module_name) is not None


def check_model_health(provider: str, role: str) -> dict[str, Any]:
    """Verify that a model responds to a minimal request."""
    model = DEFAULT_MODELS.get(provider, {}).get(role)
    if not model:
        return {"healthy": False, "model": None, "error": f"no default model for {provider}/{role}"}
    try:
        call_llm(
            [{"role": "user", "content": "Reply with the single word OK."}],
            role=role,
            provider=provider,
            model=model,
            max_tokens=5,
            timeout=10,
        )
        return {"healthy": True, "model": model, "error": None}
    except Exception as exc:
        return {"healthy": False, "model": model, "error": f"{type(exc).__name__}: {exc}"}


def get_available_providers() -> list[str]:
    """Get list of available LLM providers based on installed packages and API keys."""
    available = ["ollama"]  # Always available for local mode.

    if _has_module("anthropic") and (
        os.environ.get("HG_ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    ):
        available.append("anthropic")

    if _has_module("openai") and (
        os.environ.get("HG_OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    ):
        available.append("openai")

    return available


def get_current_config() -> dict[str, Any]:
    """Get current LLM configuration."""
    provider = os.environ.get("HG_LLM_PROVIDER", "ollama").lower()
    return {
        "provider": provider,
        "planner_model": os.environ.get(
            "HG_PLANNER_MODEL",
            DEFAULT_MODELS.get(provider, {}).get("planner", DEFAULT_PLANNER_MODEL),
        ),
        "narrator_model": os.environ.get(
            "HG_NARRATOR_MODEL",
            DEFAULT_MODELS.get(provider, {}).get("narrator", DEFAULT_NARRATOR_MODEL),
        ),
        "available_providers": get_available_providers(),
    }
