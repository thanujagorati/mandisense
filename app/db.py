"""Database access: one short-lived, read-only connection per request.

Why not a connection pool? Neon (our hosted Postgres) closes idle connections and can pause when
unused, so long-lived pooled connections go stale. Opening a connection per request is slower
(roughly 100-300 ms) but cannot go stale, which matters more for a small demo service.
"""
import os
from contextlib import contextmanager

import psycopg2
from dotenv import load_dotenv
from psycopg2.extras import RealDictCursor

load_dotenv()


def database_url() -> str:
    url = os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL is not set (put it in .env locally, or in the host's settings).")
    return url


@contextmanager
def cursor():
    conn = psycopg2.connect(database_url(), connect_timeout=15)
    try:
        conn.set_session(readonly=True, autocommit=True)   # the API never writes
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            yield cur
    finally:
        conn.close()
