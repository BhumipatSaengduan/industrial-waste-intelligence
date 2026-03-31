import psycopg2
from contextlib import contextmanager

DB_CONFIG = {
    'host'    : '34.104.146.13',
    'database': 'waste_intelligence',
    'user'    : 'postgres',
    'password': '12345678',
    'port'    : 5432
}

@contextmanager
def get_connection():
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        yield conn
    finally:
        conn.close()

def query_db(sql, params=None):
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(sql, params)
        cols = [desc[0] for desc in cur.description]
        rows = cur.fetchall()
        return [dict(zip(cols, row)) for row in rows]
