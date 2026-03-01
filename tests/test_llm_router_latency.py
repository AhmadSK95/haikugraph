from __future__ import annotations

import json
from pathlib import Path

import haikugraph.llm.router as router


def test_call_llm_uses_cache_for_identical_prompt(monkeypatch):
    calls = {"n": 0}

    def _fake_openai(messages, model, temperature, max_tokens, timeout):
        del messages, temperature, max_tokens, timeout
        calls["n"] += 1
        return f"ok:{model}", {"model": model, "input_tokens": 12, "output_tokens": 4, "total_tokens": 16}

    monkeypatch.setenv("HG_LLM_CACHE_ENABLED", "1")
    monkeypatch.setenv("HG_LLM_CACHE_NARRATOR", "1")
    monkeypatch.setenv("HG_LLM_CACHE_TTL_SECONDS", "120")
    monkeypatch.setattr(router, "_call_openai", _fake_openai)

    router.reset_llm_metrics()
    messages = [{"role": "user", "content": "How many transactions?"}]
    first = router.call_llm(
        messages,
        role="intent",
        provider="openai",
        model="gpt-4o-mini",
        timeout=5,
    )
    second = router.call_llm(
        messages,
        role="intent",
        provider="openai",
        model="gpt-4o-mini",
        timeout=5,
    )

    assert first == second
    assert calls["n"] == 1
    metrics = router.get_llm_metrics()
    assert len(metrics) == 2
    assert metrics[0]["cache_hit"] is False
    assert metrics[1]["cache_hit"] is True
    summary = router.summarize_llm_metrics(metrics)
    assert summary["calls"] == 2
    assert summary["cache_hits"] == 1
    assert summary["cache_hit_ratio"] == 0.5


def test_call_llm_adaptive_routing_downshifts_for_simple_queries(monkeypatch):
    seen_models: list[str] = []

    def _fake_openai(messages, model, temperature, max_tokens, timeout):
        del messages, temperature, max_tokens, timeout
        seen_models.append(model)
        return "ok", {"model": model, "input_tokens": 4, "output_tokens": 2, "total_tokens": 6}

    monkeypatch.setenv("HG_LLM_CACHE_ENABLED", "0")
    monkeypatch.setenv("HG_ADAPTIVE_MODEL_ROUTING", "1")
    monkeypatch.setenv("HG_FAST_OPENAI_MODEL", "gpt-4o-mini")
    monkeypatch.setattr(router, "_call_openai", _fake_openai)

    router.reset_llm_metrics()
    router.call_llm(
        [{"role": "user", "content": "Total transactions?"}],
        role="planner",
        provider="openai",
        model="gpt-4o",
        timeout=5,
    )

    assert seen_models == ["gpt-4o-mini"]
    metrics = router.get_llm_metrics()
    assert metrics
    assert metrics[0]["routing_reason"] == "downshift_for_simple_query"
    assert metrics[0]["model_requested"] == "gpt-4o"
    assert metrics[0]["model_actual"] == "gpt-4o-mini"


def test_call_llm_uses_provider_overlay_fast_model(monkeypatch, tmp_path: Path):
    overlay_dir = tmp_path / "providers"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    (overlay_dir / "openai.json").write_text(
        json.dumps(
            {
                "routing": {"fast_model": "overlay-fast-openai"},
                "timeouts": {"planner": 12},
            }
        ),
        encoding="utf-8",
    )

    seen: dict[str, object] = {}

    def _fake_openai(messages, model, temperature, max_tokens, timeout):
        del messages, temperature, max_tokens
        seen["model"] = model
        seen["timeout"] = timeout
        return "ok", {"model": model, "input_tokens": 4, "output_tokens": 2, "total_tokens": 6}

    monkeypatch.setenv("HG_PROVIDER_OVERLAY_DIR", str(overlay_dir))
    monkeypatch.setenv("HG_LLM_CACHE_ENABLED", "0")
    monkeypatch.setenv("HG_ADAPTIVE_MODEL_ROUTING", "1")
    monkeypatch.delenv("HG_FAST_OPENAI_MODEL", raising=False)
    monkeypatch.setattr(router, "_call_openai", _fake_openai)

    router.call_llm(
        [{"role": "user", "content": "Total transactions?"}],
        role="planner",
        provider="openai",
        model="gpt-4o",
        timeout=60,
    )
    assert seen["model"] == "overlay-fast-openai"
    assert int(seen["timeout"]) == 12


def test_call_llm_uses_provider_overlay_fallback_models(monkeypatch, tmp_path: Path):
    overlay_dir = tmp_path / "providers"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    (overlay_dir / "openai.json").write_text(
        json.dumps(
            {
                "routing": {
                    "fallback_models": {"planner": ["overlay-fallback-model"]},
                }
            }
        ),
        encoding="utf-8",
    )

    attempts: list[str] = []

    def _fake_openai(messages, model, temperature, max_tokens, timeout):
        del messages, temperature, max_tokens, timeout
        attempts.append(model)
        if model == "gpt-4o":
            raise ValueError("primary unavailable")
        return "ok", {"model": model, "input_tokens": 4, "output_tokens": 2, "total_tokens": 6}

    monkeypatch.setenv("HG_PROVIDER_OVERLAY_DIR", str(overlay_dir))
    monkeypatch.setenv("HG_LLM_CACHE_ENABLED", "0")
    monkeypatch.setenv("HG_ENABLE_MODEL_FALLBACK", "1")
    monkeypatch.setattr(router, "_call_openai", _fake_openai)

    out = router.call_llm(
        [{"role": "user", "content": "Compare transactions and quotes"}],
        role="planner",
        provider="openai",
        model="gpt-4o",
        timeout=30,
    )
    assert out == "ok"
    assert attempts[:2] == ["gpt-4o", "overlay-fallback-model"]
