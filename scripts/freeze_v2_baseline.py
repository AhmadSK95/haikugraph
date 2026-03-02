#!/usr/bin/env python3
"""Freeze a signed baseline evidence set for v2 governance gates."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _latest(reports_dir: Path, pattern: str) -> Path | None:
    matches = list(reports_dir.glob(pattern))
    if not matches:
        return None
    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0]


def _artifact_row(path: Path, kind: str) -> dict[str, Any]:
    stat = path.stat()
    return {
        "kind": kind,
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_epoch_ms": int(stat.st_mtime * 1000),
        "sha256": _sha256(path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze v2 baseline evidence set")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--out-dir", default="reports")
    parser.add_argument("--label", default="v2_release_baseline")
    args = parser.parse_args()

    reports_dir = Path(args.reports_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    required = {
        "truth_report": _latest(reports_dir, "v2_qa_truth_report_*.json"),
        "blackbox_report": _latest(reports_dir, "qa_round11_blackbox_fresh_*.json"),
    }
    optional = {
        "semantic_probe": _latest(reports_dir, "blackbox_semantic_probe_*.json"),
        "ui_regression": _latest(reports_dir, "ui_regression_qa_*.json"),
        "latency_check": _latest(reports_dir, "latency_optimization_check_*.json"),
    }

    missing = [name for name, path in required.items() if path is None]
    if missing:
        raise SystemExit(f"Missing required artifacts: {', '.join(missing)}")

    artifacts: list[dict[str, Any]] = []
    for kind, path in required.items():
        artifacts.append(_artifact_row(path, kind))
    for kind, path in optional.items():
        if path is not None:
            artifacts.append(_artifact_row(path, kind))

    artifacts = sorted(artifacts, key=lambda row: row["kind"])
    signature_payload = json.dumps(artifacts, sort_keys=True, separators=(",", ":")).encode("utf-8")
    manifest_signature = hashlib.sha256(signature_payload).hexdigest()

    truth_payload = json.loads(Path(required["truth_report"]).read_text(encoding="utf-8"))
    summary = dict(truth_payload.get("summary") or {})

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "label": str(args.label),
        "release_gate_passed": bool(summary.get("release_gate_passed")),
        "composite_truth_score": summary.get("composite_truth_score"),
        "floor_violations": list(summary.get("floor_violations") or []),
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "manifest_signature_sha256": manifest_signature,
    }

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"v2_baseline_lock_{ts}.json"
    out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"baseline_manifest": str(out_path), "summary": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
