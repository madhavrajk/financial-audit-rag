"""Read-only SQLite adapter for the FinancialStore port.

The model writes SQL, so the database must be safe against any SQL it writes. Three independent
layers enforce read-only access:
  1. the file is opened with a `mode=ro` URI (the OS-level handle cannot write),
  2. `PRAGMA query_only = ON`,
  3. an authorizer callback that whitelists only read actions (SELECT/READ/FUNCTION/RECURSIVE),
     which also blocks ATTACH, PRAGMA, and temp-table creation.
A progress handler aborts runaway queries, and results are capped so a careless
`SELECT *` cannot flood the model's context window.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

from app.core.errors import ToolError
from app.core.ports import QueryResult

_ALLOWED_ACTIONS = {
    sqlite3.SQLITE_SELECT,
    sqlite3.SQLITE_READ,
    sqlite3.SQLITE_FUNCTION,
    getattr(sqlite3, "SQLITE_RECURSIVE", 33),  # WITH RECURSIVE
}


def _authorizer(
    action: int, _arg1: str | None, _arg2: str | None, _db: str | None, _trigger: str | None
) -> int:
    return sqlite3.SQLITE_OK if action in _ALLOWED_ACTIONS else sqlite3.SQLITE_DENY


class SQLiteFinancialStore:
    def __init__(self, db_path: Path, *, max_rows: int = 200, timeout_s: float = 5.0) -> None:
        if not db_path.exists():
            raise FileNotFoundError(f"Database not found: {db_path}")
        self._uri = f"file:{db_path.resolve()}?mode=ro"
        self._max_rows = max_rows
        self._timeout_s = timeout_s
        self._schema = self._build_schema_description()

    def _connect(self) -> sqlite3.Connection:
        # A connection per call: SQLite connections are cheap, and this keeps concurrent
        # requests (run in worker threads) fully isolated from each other.
        conn = sqlite3.connect(self._uri, uri=True, check_same_thread=False)
        conn.execute("PRAGMA query_only = ON")
        conn.set_authorizer(_authorizer)
        return conn

    def query(self, sql: str) -> QueryResult:
        sql = sql.strip().rstrip(";").strip()
        if not sql:
            raise ToolError("Empty SQL query.")
        conn = self._connect()
        deadline = time.monotonic() + self._timeout_s
        # Returning non-zero from the progress handler interrupts the query.
        conn.set_progress_handler(lambda: int(time.monotonic() > deadline), 10_000)
        try:
            cur = conn.execute(sql)
            columns = [d[0] for d in cur.description or []]
            rows = cur.fetchmany(self._max_rows + 1)
        except sqlite3.DatabaseError as exc:  # incl. not-authorized, syntax, multi-statement
            msg = str(exc)
            if "not authorized" in msg:
                msg = "not authorized: the database is read-only; only SELECT queries are allowed"
            elif "interrupted" in msg:
                msg = f"query exceeded {self._timeout_s:.0f}s and was aborted"
            raise ToolError(f"SQL error: {msg}") from exc
        finally:
            conn.close()
        truncated = len(rows) > self._max_rows
        return QueryResult(
            columns=columns, rows=[list(r) for r in rows[: self._max_rows]], truncated=truncated
        )

    def schema_description(self) -> str:
        return self._schema

    def companies(self) -> dict[str, tuple[str, str]]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT ticker, name, coalesce(fiscal_year_end, '') FROM companies "
                "WHERE ticker IN (SELECT company_ticker FROM income_statements) ORDER BY ticker"
            ).fetchall()
        finally:
            conn.close()
        return {t: (name, fye) for t, name, fye in rows}

    def _build_schema_description(self) -> str:
        """Compact schema plus the enumerated values the model would otherwise have to guess."""
        conn = self._connect()
        try:
            parts: list[str] = []
            tables = conn.execute(
                "SELECT name, sql FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%' ORDER BY name"
            ).fetchall()
            for name, _ in tables:
                cols = conn.execute(f"SELECT * FROM {name} LIMIT 0").description or []
                parts.append(f"{name}({', '.join(c[0] for c in cols)})")
            seg = conn.execute(
                "SELECT company_ticker, group_concat(DISTINCT segment_name) FROM segment_revenue "
                "GROUP BY company_ticker ORDER BY company_ticker"
            ).fetchall()
            geo = conn.execute(
                "SELECT company_ticker, group_concat(DISTINCT region) FROM geographic_revenue "
                "GROUP BY company_ticker ORDER BY company_ticker"
            ).fetchall()
            spans: dict[str, dict[str, str]] = {}  # ticker -> table -> "FY2023-FY2025"
            for table in (
                "income_statements",
                "balance_sheets",
                "segment_revenue",
                "geographic_revenue",
            ):
                for t, lo, hi in conn.execute(
                    f"SELECT company_ticker, min(fiscal_year), max(fiscal_year) FROM {table} "
                    "GROUP BY company_ticker"
                ):
                    spans.setdefault(t, {})[table] = f"FY{lo}-FY{hi}"
        finally:
            conn.close()
        lines = ["Tables:"] + [f"  {p}" for p in parts]
        lines.append("segment_revenue.segment_name values:")
        lines += [f"  {t}: {v}" for t, v in seg]
        lines.append("geographic_revenue.region values:")
        lines += [f"  {t}: {v}" for t, v in geo]
        lines.append("Fiscal years available (period_type='FY' only):")
        for t, by_table in sorted(spans.items()):
            main = by_table.get("income_statements", "")
            other = [f"{table} {span}" for table, span in by_table.items() if span != main]
            lines.append(f"  {t}: {main}" + (f" (except {'; '.join(other)})" if other else ""))
        return "\n".join(lines)
