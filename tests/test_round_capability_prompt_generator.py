from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def _load_prompt_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "generate_round_capability_prompts.py"
    spec = importlib.util.spec_from_file_location("generate_round_capability_prompts", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def test_round_capability_prompt_generator_outputs_45_unique_prompts() -> None:
    module = _load_prompt_module()
    tracker = Path(__file__).resolve().parents[1] / "PRODUCT_GAP_TRACKER.md"
    prompts = module.generate_prompts(
        product_tracker=tracker,
        round_id="RTEST-45",
        quick=False,
    )
    assert len(prompts) == 45
    capability_ids = [row.capability_id for row in prompts]
    assert len(set(capability_ids)) == 45


def test_round_capability_prompt_generator_quick_mode_is_smaller() -> None:
    module = _load_prompt_module()
    tracker = Path(__file__).resolve().parents[1] / "PRODUCT_GAP_TRACKER.md"
    prompts = module.generate_prompts(
        product_tracker=tracker,
        round_id="RTEST-QUICK",
        quick=True,
    )
    assert len(prompts) == 15
