from pathlib import Path

from haikugraph.poc.skill_runtime import SkillRuntime


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_skill_runtime_merges_agent_and_layer_skills(tmp_path: Path) -> None:
    skills_dir = tmp_path / "skills"
    _write(
        skills_dir / "agent-chiefAnalyst.md",
        "# x\n\nUse skills:\n- query-expert\n- verification-before-completion\n\nMust produce:\n- out\n",
    )
    _write(
        skills_dir / "layer4-planning.md",
        "# x\n\nUse skills:\n- dispatching-parallel-agents\n- query-expert\n",
    )

    runtime = SkillRuntime(skills_dir)
    selection = runtime.resolve("ChiefAnalystAgent")

    assert selection.skill_contract_file == "agent-chiefAnalyst.md"
    assert selection.skill_layer_file == "layer4-planning.md"
    assert selection.selected_skills == [
        "query-expert",
        "verification-before-completion",
        "dispatching-parallel-agents",
    ]


def test_skill_runtime_handles_missing_skills_directory(tmp_path: Path) -> None:
    runtime = SkillRuntime(tmp_path / "missing")
    selection = runtime.resolve("UnknownAgent")

    assert selection.selected_skills == []
    assert "skills directory not found" in selection.skill_policy_reason


def test_skill_runtime_contract_evaluation_passes_on_required_outputs(tmp_path: Path) -> None:
    skills_dir = tmp_path / "skills"
    _write(
        skills_dir / "agent-queryEngineer.md",
        (
            "# x\n\nUse skills:\n- query-expert\n\nExecution contract:\n- safe SQL\n\n"
            "Must produce:\n- final SQL\n- grouping correctness\n"
        ),
    )
    runtime = SkillRuntime(skills_dir)
    evaluation = runtime.evaluate_contract(
        "QueryEngineerAgent",
        {"sql": "SELECT platform_name, COUNT(*) FROM t GROUP BY 1", "dimensions": ["platform_name"]},
    )
    assert evaluation.enabled is True
    assert evaluation.passed is True
    assert all(item["passed"] for item in evaluation.checks)


def test_skill_runtime_contract_evaluation_fails_when_missing_outputs(tmp_path: Path) -> None:
    skills_dir = tmp_path / "skills"
    _write(
        skills_dir / "agent-narrative.md",
        (
            "# x\n\nUse skills:\n- verification-before-completion\n\nExecution contract:\n- grounded\n\n"
            "Must produce:\n- concise answer\n- caveats\n"
        ),
    )
    runtime = SkillRuntime(skills_dir)
    evaluation = runtime.evaluate_contract("NarrativeAgent", {"answer": "ok"})
    assert evaluation.enabled is True
    assert evaluation.passed is False
    assert any(not item["passed"] for item in evaluation.checks)
