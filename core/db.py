"""
db.py — PostgreSQL interface for complaint storage.

Single responsibility: save a completed complaint and return its ID.
All other modules call save_complaint() — none write SQL directly.

Connection credentials are read from .env via python-dotenv,
matching the pattern used throughout this codebase.
"""

from __future__ import annotations

import os

import psycopg2
from dotenv import load_dotenv

load_dotenv()

# ------------------------------------------------------------------ #
# Connection config — read from .env
# ------------------------------------------------------------------ #

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_NAME = os.getenv("DB_NAME", "moc_agent")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")


def _connect() -> psycopg2.extensions.connection:
    """Open and return a new database connection."""
    return psycopg2.connect(
        host=DB_HOST,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
    )


# ------------------------------------------------------------------ #
# Public interface
# ------------------------------------------------------------------ #


def save_complaint(fields: dict) -> int:
    """
    Insert a complaint row and return the new complaint ID.

    Expected keys in fields:
        store_name  (str) — required
        cr_number   (str) — required, 10-digit Saudi commercial registration number
        order_id    (str) — required
        order_date  (str) — required, ISO format preferred (YYYY-MM-DD)
        description (str) — required

    Raises:
        psycopg2.Error if the insert fails (e.g. missing required field).
    """
    conn = _connect()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO complaints
                        (store_name, cr_number, order_id, order_date, description)
                    VALUES
                        (%(store_name)s, %(cr_number)s, %(order_id)s,
                         %(order_date)s, %(description)s)
                    RETURNING id;
                    """,
                    fields,
                )
                complaint_id: int = cur.fetchone()[0]
        return complaint_id
    finally:
        conn.close()
