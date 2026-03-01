#!/usr/bin/env python3
"""Generate an onboarding schema profile for a DuckDB dataset."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from haikugraph.io.onboarding_profile import build_onboarding_profile


def _render_markdown(profile: dict[str, Any]) -> str:
    lines = [
        "# Industry Onboarding Profile",
        "",
        f"- Generated at: `{profile.get('generated_at')}`",
        f"- Database: `{profile.get('db_path')}`",
        f"- Source table count: `{profile.get('table_count')}`",
        "",
        "## Tables",
    ]
    for table in profile.get("tables", []):
        lines.extend(
            [
                "",
                f"### `{table.get('table')}`",
                f"- Rows: `{table.get('row_count')}`",
                f"- Columns: `{table.get('column_count')}`",
                f"- ID columns: `{', '.join(table.get('id_columns') or []) or 'none'}`",
                f"- Time columns: `{', '.join(table.get('time_columns') or []) or 'none'}`",
                f"- Metric columns: `{', '.join(table.get('metric_columns') or []) or 'none'}`",
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate schema onboarding profile for a DuckDB database")
    parser.add_argument("--db-path", required=True, help="DuckDB database path")
    parser.add_argument("--out-json", default="", help="Output JSON path")
    parser.add_argument("--out-md", default="", help="Output markdown path")
    parser.add_argument(
        "--include-datada-views",
        action="store_true",
        help="Include datada_* semantic views/tables in the profile",
    )
    args = parser.parse_args()

    db_path = Path(args.db_path).expanduser()
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    profile = build_onboarding_profile(
        db_path,
        include_datada_views=bool(args.include_datada_views),
    )
    out_json = Path(args.out_json).expanduser() if args.out_json else Path("reports") / f"onboarding_profile_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    out_md = Path(args.out_md).expanduser() if args.out_md else out_json.with_suffix(".md")

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    out_md.write_text(_render_markdown(profile), encoding="utf-8")

    print(f"saved_json={out_json}")
    print(f"saved_md={out_md}")
    print(f"tables={profile.get('table_count')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
