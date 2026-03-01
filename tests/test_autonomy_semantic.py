from __future__ import annotations

from pathlib import Path

from haikugraph.poc.autonomy import AgentMemoryStore


def test_semantic_glossary_resolves_paraphrase(tmp_path: Path) -> None:
    db = tmp_path / "memory.duckdb"
    store = AgentMemoryStore(db)
    store.upsert_glossary_term(
        tenant_id="public",
        term="forex markup",
        definition="Fee applied on exchange quotes.",
        target_table="datada_mart_quotes",
    )

    matches = store.resolve_glossary("show fx charges by month", tenant_id="public")
    assert matches, "Expected semantic glossary match for paraphrase"
    assert matches[0]["term"] == "forex markup"
    assert float(matches[0].get("match_score", 0.0)) >= 0.3


def test_business_rule_matching_uses_semantic_similarity(tmp_path: Path) -> None:
    db = tmp_path / "memory.duckdb"
    store = AgentMemoryStore(db)
    rule_id = store.create_business_rule(
        tenant_id="public",
        domain="transactions",
        name="customer spend rule",
        rule_type="plan_override",
        triggers=["customer spend"],
        action_payload={"target_metric": "customer_spend"},
        notes="semantic test",
        status="active",
        created_by="test",
    )
    assert rule_id

    matches = store.get_matching_business_rules(
        "show client spending trend",
        tenant_id="public",
    )
    assert matches, "Expected semantic business-rule match for paraphrase"
    assert matches[0]["rule_id"] == rule_id
    assert float(matches[0].get("semantic_quality", 0.0)) > 0.0


def test_semantic_glossary_long_tail_phrase_matches_metric(tmp_path: Path) -> None:
    db = tmp_path / "memory.duckdb"
    store = AgentMemoryStore(db)
    store.upsert_glossary_term(
        tenant_id="public",
        term="forex markup revenue",
        definition="Revenue from foreign exchange surcharge on quotes.",
        target_table="datada_mart_quotes",
    )

    matches = store.resolve_glossary(
        "show foreign exchange surcharge trend by month",
        tenant_id="public",
    )
    assert matches
    assert matches[0]["term"] == "forex markup revenue"


def test_semantic_glossary_blocks_unsafe_cross_table_near_tie(tmp_path: Path) -> None:
    db = tmp_path / "memory.duckdb"
    store = AgentMemoryStore(db)
    store.upsert_glossary_term(
        tenant_id="public",
        term="quote charges",
        definition="charges in quote flow",
        target_table="datada_mart_quotes",
    )
    store.upsert_glossary_term(
        tenant_id="public",
        term="transaction charges",
        definition="charges in transaction flow",
        target_table="datada_mart_transactions",
    )

    matches = store.resolve_glossary("charges by flow", tenant_id="public")
    assert matches == []
