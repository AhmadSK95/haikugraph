from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def _load_round11_module():
    repo_root = Path(__file__).resolve().parents[1]
    scripts_dir = repo_root / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    module_path = repo_root / "scripts" / "qa_round11_blackbox_fresh.py"
    spec = importlib.util.spec_from_file_location("qa_round11_blackbox_fresh", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def test_round11_freshener_mutates_prompt_surface() -> None:
    module = _load_round11_module()
    baseline = module.build_atomic_cases()
    fresh = module._freshen_atomic_cases(baseline, round_id="RTEST-20260301")
    assert len(fresh) == len(baseline)
    changed = sum(1 for left, right in zip(baseline, fresh) if left.question != right.question)
    assert changed >= 1


def test_round11_quick_subset_reduces_case_count() -> None:
    module = _load_round11_module()
    baseline = module.build_atomic_cases()
    quick = module._quick_subset_atomic(baseline)
    assert len(quick) < len(baseline)
    categories = {}
    for case in quick:
        categories[case.category] = categories.get(case.category, 0) + 1
    assert all(count <= 2 for count in categories.values())
