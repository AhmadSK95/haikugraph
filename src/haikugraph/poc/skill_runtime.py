"""Skill contract loader for dataDa layered agent runtime.

Loads project-local skill contracts from `skills/` and provides
agent/layer skill selections for traceability and governance.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any


_AGENT_TO_FILE = {
    "ChiefAnalystAgent": "agent-chiefAnalyst.md",
    "DataEngineeringTeam": "agent-dataEngineering.md",
    "ConnectionRouter": "agent-connectionRouter.md",
    "MemoryAgent": "agent-memory.md",
    "OrganizationalKnowledgeAgent": "agent-organizationalKnowledge.md",
    "BlackboardAgent": "agent-blackboard.md",
    "IntakeAgent": "agent-intake.md",
    "DiscoveryPlannerAgent": "agent-discoveryPlanner.md",
    "CatalogProfilerAgent": "agent-catalogProfiler.md",
    "DocumentRetrievalAgent": "agent-documentRetrieval.md",
    "SemanticRetrievalAgent": "agent-semanticRetrieval.md",
    "PlanningAgent": "agent-planning.md",
    "QueryEngineerAgent": "agent-queryEngineer.md",
    "ExecutionAgent": "agent-execution.md",
    "AuditAgent": "agent-audit.md",
    "AutonomyAgent": "agent-autonomy.md",
    "ToolsmithAgent": "agent-toolsmith.md",
    "TrustAgent": "agent-trust.md",
    "SLOIncidentAgent": "agent-sloIncident.md",
    "NarrativeAgent": "agent-narrative.md",
    "VisualizationAgent": "agent-visualization.md",
    "ContextAgent": "agent-intake.md",
    "DomainKnowledgeAgent": "agent-organizationalKnowledge.md",
    "ClarificationAgent": "agent-intake.md",
}

_AGENT_TO_LAYER = {
    "ChiefAnalystAgent": "layer4-planning.md",
    "DataEngineeringTeam": "layer2-dataEngineer.md",
    "ConnectionRouter": "layer3-dispatcher.md",
    "MemoryAgent": "layer4-context.md",
    "OrganizationalKnowledgeAgent": "layer4-semantic.md",
    "BlackboardAgent": "layer4-planning.md",
    "IntakeAgent": "layer4-context.md",
    "DiscoveryPlannerAgent": "layer4-planning.md",
    "CatalogProfilerAgent": "layer4-semantic.md",
    "DocumentRetrievalAgent": "layer1-ingest.md",
    "SemanticRetrievalAgent": "layer4-semantic.md",
    "PlanningAgent": "layer4-planning.md",
    "QueryEngineerAgent": "layer4-queryEngineer.md",
    "ExecutionAgent": "layer4-execution.md",
    "AuditAgent": "layer5-trustValidation.md",
    "AutonomyAgent": "layer5-trustValidation.md",
    "ToolsmithAgent": "layer5-trustValidation.md",
    "TrustAgent": "layer5-trustValidation.md",
    "SLOIncidentAgent": "layer5-trustValidation.md",
    "NarrativeAgent": "layer6-explainability.md",
    "VisualizationAgent": "layer6-explainability.md",
    "ContextAgent": "layer4-context.md",
    "DomainKnowledgeAgent": "layer4-semantic.md",
    "ClarificationAgent": "layer4-context.md",
}


@dataclass
class SkillSelection:
    selected_skills: list[str]
    skill_policy_reason: str
    skill_contract_file: str
    skill_layer_file: str


@dataclass
class SkillContractEvaluation:
    enabled: bool
    passed: bool
    checks: list[dict[str, Any]]


class SkillRuntime:
    """Resolve project-local skills for each agent execution."""

    def __init__(self, skills_dir: Path):
        self.skills_dir = skills_dir
        self.available = skills_dir.exists()
        self._skills_cache: dict[Path, list[str]] = {}
        self._section_cache: dict[tuple[Path, str], list[str]] = {}

    @classmethod
    def discover(cls, *, db_path: Path | None = None) -> "SkillRuntime":
        candidates: list[Path] = []

        env_raw = str(__import__("os").environ.get("HG_SKILLS_DIR", "")).strip()
        if env_raw:
            candidates.append(Path(env_raw).expanduser())

        file_root = Path(__file__).resolve().parents[3]  # repo root
        candidates.append(file_root / "skills")
        candidates.append(Path.cwd() / "skills")

        if db_path is not None:
            candidates.append(db_path.parent / "skills")

        seen: set[str] = set()
        for path in candidates:
            k = str(path)
            if k in seen:
                continue
            seen.add(k)
            if path.exists() and path.is_dir():
                return cls(path)

        return cls(file_root / "skills")

    def _extract_skills(self, contract_file: Path) -> list[str]:
        if contract_file in self._skills_cache:
            return self._skills_cache[contract_file]
        if not contract_file.exists():
            self._skills_cache[contract_file] = []
            return []

        skills: list[str] = []
        in_use_section = False
        for raw_line in contract_file.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = raw_line.strip()
            lower = line.lower()
            if lower.startswith("use skills:"):
                in_use_section = True
                continue
            if in_use_section and line.startswith("#"):
                break
            if in_use_section and line.startswith("must produce:"):
                break
            if in_use_section and line.startswith("execution contract:"):
                break
            if in_use_section and line.startswith("-"):
                name = line.lstrip("-").strip()
                if name and name not in skills:
                    skills.append(name)
            elif in_use_section and line:
                # Non-bullet line marks the end of section for this format.
                if not line.startswith("-"):
                    break

        self._skills_cache[contract_file] = skills
        return skills

    def _extract_bullets_for_heading(self, contract_file: Path, heading: str) -> list[str]:
        key = (contract_file, heading.lower())
        if key in self._section_cache:
            return self._section_cache[key]
        if not contract_file.exists():
            self._section_cache[key] = []
            return []

        in_section = False
        out: list[str] = []
        prefix = heading.strip().lower() + ":"
        for raw_line in contract_file.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = raw_line.strip()
            lower = line.lower()
            if lower.startswith(prefix):
                in_section = True
                continue
            if in_section and line.startswith("#"):
                break
            if in_section and (lower.endswith(":") and not line.startswith("-")):
                break
            if in_section and line.startswith("-"):
                val = line.lstrip("-").strip()
                if val:
                    out.append(val)
                continue
            if in_section and line:
                break
        self._section_cache[key] = out
        return out

    def _contract_file_for_agent(self, agent_name: str) -> Path | None:
        if not self.available:
            return None
        file_name = _AGENT_TO_FILE.get(agent_name, "")
        if not file_name:
            return None
        p = self.skills_dir / file_name
        return p if p.exists() else None

    def _layer_file_for_agent(self, agent_name: str) -> Path | None:
        if not self.available:
            return None
        file_name = _AGENT_TO_LAYER.get(agent_name, "")
        if not file_name:
            return None
        p = self.skills_dir / file_name
        return p if p.exists() else None

    @staticmethod
    def _normalize_requirement(text: str) -> str:
        cleaned = re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()
        return re.sub(r"\s+", " ", cleaned)

    def _extract_contract_requirements(self, agent_name: str) -> list[str]:
        reqs: list[str] = []
        for p in (self._contract_file_for_agent(agent_name), self._layer_file_for_agent(agent_name)):
            if p is None:
                continue
            reqs.extend(self._extract_bullets_for_heading(p, "Must produce"))
            reqs.extend(self._extract_bullets_for_heading(p, "Required outputs"))
        deduped: list[str] = []
        seen: set[str] = set()
        for item in reqs:
            key = self._normalize_requirement(item)
            if key and key not in seen:
                seen.add(key)
                deduped.append(item)
        return deduped

    def _requirement_tokens(self, requirement: str) -> list[str]:
        key = self._normalize_requirement(requirement)
        synonyms = {
            "final sql": ["sql", "sql_executed", "sql_text"],
            "safe predicates": ["where", "has_mt103", "filter", "predicate"],
            "grouping correctness": ["group by", "dimension", "dimensions", "__month__", "currency_pair"],
            "concise answer": ["answer_markdown", "answer", "summary"],
            "analyst insights": ["insight", "driver", "trend", "takeaway", "concentration"],
            "caveats": ["warning", "warnings", "caveat", "risk", "limitation"],
            "check matrix": ["check", "checks", "validation", "audit"],
            "confidence decomposition": ["confidence", "calibration", "decomposition"],
            "warnings and failure reasons": ["warning", "failure", "reason"],
        }
        for canonical, mapped in synonyms.items():
            if canonical in key:
                return mapped
        return [tok for tok in key.split() if len(tok) >= 3][:4]

    def evaluate_contract(self, agent_name: str, output: Any) -> SkillContractEvaluation:
        if not self.available:
            return SkillContractEvaluation(enabled=False, passed=True, checks=[])
        requirements = self._extract_contract_requirements(agent_name)
        if not requirements:
            return SkillContractEvaluation(enabled=False, passed=True, checks=[])

        flat_text = str(output).lower()
        flat_keys = set()
        if isinstance(output, dict):
            for k in output.keys():
                flat_keys.add(str(k).lower())
        checks: list[dict[str, Any]] = []
        all_passed = True
        for req in requirements:
            tokens = self._requirement_tokens(req)
            matched_tokens: list[str] = []
            for token in tokens:
                t = token.lower()
                if t in flat_keys or t in flat_text:
                    matched_tokens.append(token)
            passed = len(matched_tokens) > 0
            all_passed = all_passed and passed
            checks.append(
                {
                    "requirement": req,
                    "passed": passed,
                    "matched_tokens": matched_tokens,
                }
            )
        return SkillContractEvaluation(enabled=True, passed=all_passed, checks=checks)

    def resolve(self, agent_name: str) -> SkillSelection:
        if not self.available:
            return SkillSelection(
                selected_skills=[],
                skill_policy_reason="skills directory not found; runtime running without skill contracts",
                skill_contract_file="",
                skill_layer_file="",
            )

        agent_file_name = _AGENT_TO_FILE.get(agent_name, "")
        layer_file_name = _AGENT_TO_LAYER.get(agent_name, "")
        agent_file = self.skills_dir / agent_file_name if agent_file_name else Path("")
        layer_file = self.skills_dir / layer_file_name if layer_file_name else Path("")

        agent_skills = self._extract_skills(agent_file) if agent_file_name else []
        layer_skills = self._extract_skills(layer_file) if layer_file_name else []

        merged: list[str] = []
        for skill in [*agent_skills, *layer_skills]:
            if skill not in merged:
                merged.append(skill)

        if merged:
            reason = "contract-driven selection from agent+layer skills"
        elif agent_file_name or layer_file_name:
            reason = "contract files found but no explicit skills listed"
        else:
            reason = "no matching agent skill contract"

        return SkillSelection(
            selected_skills=merged,
            skill_policy_reason=reason,
            skill_contract_file=str(agent_file.relative_to(self.skills_dir)) if agent_file_name else "",
            skill_layer_file=str(layer_file.relative_to(self.skills_dir)) if layer_file_name else "",
        )

    def summary(self) -> dict[str, Any]:
        enforceable_agents = 0
        if self.available:
            for agent_name in _AGENT_TO_FILE:
                if self._extract_contract_requirements(agent_name):
                    enforceable_agents += 1
        return {
            "available": self.available,
            "skills_dir": str(self.skills_dir),
            "enforceable_agents": enforceable_agents,
        }

    def has_skill(self, agent_name: str, skill_name: str) -> bool:
        selected = self.resolve(agent_name).selected_skills
        needle = str(skill_name or "").strip().lower()
        return any(str(s).strip().lower() == needle for s in selected)
