"""
Full refresh — drops all tables and schemas, then reloads every file in inbox/.

Usage:
    python scripts/refresh.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from loader import load_file, connect, INBOX, SUPPORTED

_SYSTEM_SCHEMAS = {"information_schema", "pg_catalog"}


def refresh():
    print("Dropping all existing tables and schemas...")
    con = connect()

    schemas = [
        row[0] for row in con.execute(
            "SELECT schema_name FROM information_schema.schemata "
            "WHERE schema_name NOT IN ('information_schema', 'pg_catalog')"
        ).fetchall()
    ]

    total_dropped = 0
    for schema in schemas:
        tables = [
            row[0] for row in con.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = ?",
                [schema]
            ).fetchall()
        ]
        for tbl in tables:
            con.execute(f'DROP TABLE IF EXISTS "{schema}"."{tbl}"')
            total_dropped += 1
        if schema != "main":
            con.execute(f'DROP SCHEMA IF EXISTS "{schema}"')

    con.close()
    print(f"  Dropped {total_dropped} table(s).\n")

    files = sorted(
        f for f in INBOX.rglob("*")
        if f.is_file() and f.suffix.lower() in SUPPORTED
    )

    if not files:
        print(f"No supported files found in {INBOX}")
        return

    print(f"Loading {len(files)} file(s) from inbox...\n")
    for path in files:
        rel = path.relative_to(INBOX)
        print(f"  {rel}")
        tables = load_file(path)
        for tbl in tables:
            print(f"    -> {tbl}")

    print("\nDone. Current tables:")
    con = connect()
    rows = con.execute(
        "SELECT table_schema, table_name FROM information_schema.tables "
        "WHERE table_schema NOT IN ('information_schema', 'pg_catalog') "
        "ORDER BY table_schema, table_name"
    ).fetchall()
    current_schema = None
    for schema, tbl in rows:
        if schema != current_schema:
            print(f"\n  [{schema}]")
            current_schema = schema
        count = con.execute(f'SELECT COUNT(*) FROM "{schema}"."{tbl}"').fetchone()[0]
        print(f"    {tbl:40s} {count:>10,} rows")
    con.close()


if __name__ == "__main__":
    refresh()
