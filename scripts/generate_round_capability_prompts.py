#!/usr/bin/env python3
"""Generate a fresh 45-capability prompt set from the product tracker."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CapabilityPrompt:
    capability_id: str
    capability_name: str
    requirement_refs: str
    prompt: str


_VERB_PAIRS = [
    ("show", "demonstrate"),
    ("explain", "clarify"),
    ("compare", "contrast"),
    ("summarize", "brief"),
]


def _freshen(text: str, *, round_id: str, capability_id: str) -> str:
    seed = hashlib.sha1(f"{round_id}:{capability_id}:{text}".encode("utf-8")).hexdigest()
    pick = int(seed[:2], 16) % len(_VERB_PAIRS)
    src, dst = _VERB_PAIRS[pick]
    low = text.lower()
    if src in low:
        idx = low.find(src)
        return text[:idx] + dst + text[idx + len(src):]
    return f"{text} (round {round_id[-6:]})"


def _extract_capabilities(product_tracker: Path) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    pattern = re.compile(r"^\|\s*([AT]\d{2})\s*\|\s*([^|]+)\|\s*[^|]+\|\s*[^|]+\|\s*[^|]+\|\s*[^|]+\|\s*([^|]+)\|")
    for line in product_tracker.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line.strip())
        if not match:
            continue
        cap_id = match.group(1).strip()
        name = match.group(2).strip()
        reqs = match.group(3).strip()
        rows.append((cap_id, name, reqs))
    rows = sorted(rows, key=lambda x: x[0])
    return rows


def generate_prompts(*, product_tracker: Path, round_id: str, quick: bool = False) -> list[CapabilityPrompt]:
    caps = _extract_capabilities(product_tracker)
    prompts: list[CapabilityPrompt] = []
    for cap_id, name, reqs in caps:
        if cap_id.startswith("A"):
            base = (
                f"Show {name.lower()} using this dataset, include exact SQL-backed evidence, "
                "and state any assumptions in business language."
            )
        else:
            base = (
                f"Show team capability {name.lower()} end-to-end, including decision trace, "
                "validation checks, and one follow-up question."
            )
        prompt = _freshen(base, round_id=round_id, capability_id=cap_id)
        prompts.append(
            CapabilityPrompt(
                capability_id=cap_id,
                capability_name=name,
                requirement_refs=reqs,
                prompt=prompt,
            )
        )
    if quick:
        prompts = prompts[:15]
    return prompts


def _render_markdown(round_id: str, prompts: list[CapabilityPrompt]) -> str:
    lines = [
        f"# Capability Prompt Set ({round_id})",
        "",
        f"- Prompt count: `{len(prompts)}`",
        "",
        "| Capability | Requirement refs | Prompt |",
        "| --- | --- | --- |",
    ]
    for row in prompts:
        lines.append(
            f"| {row.capability_id} {row.capability_name} | {row.requirement_refs} | {row.prompt.replace('|', '/')} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate fresh capability prompts from PRODUCT_GAP_TRACKER.md")
    parser.add_argument("--tracker", default="PRODUCT_GAP_TRACKER.md")
    parser.add_argument("--out-dir", default="reports")
    parser.add_argument("--round-id", default="")
    parser.add_argument("--quick", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    tracker = Path(args.tracker).expanduser()
    if not tracker.exists():
        raise FileNotFoundError(f"Tracker not found: {tracker}")
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    round_id = str(args.round_id).strip() or f"round-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
    prompts = generate_prompts(
        product_tracker=tracker,
        round_id=round_id,
        quick=bool(args.quick),
    )
    payload: dict[str, Any] = {
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "round_id": round_id,
        "quick": bool(args.quick),
        "count": len(prompts),
        "prompts": [asdict(item) for item in prompts],
    }
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"round_capability_eval_{stamp}.json"
    md_path = out_dir / f"round_capability_eval_{stamp}.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text(_render_markdown(round_id, prompts), encoding="utf-8")
    print(f"saved_json={json_path}")
    print(f"saved_md={md_path}")
    print(f"count={len(prompts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
