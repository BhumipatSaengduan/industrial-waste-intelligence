import psycopg2
import os
from contextlib import contextmanager
from dotenv import load_dotenv

load_dotenv()

DB_CONFIG = {
    'host'    : os.getenv('DB_HOST', '34.104.146.13'),
    'database': os.getenv('DB_NAME', 'waste_intelligence'),
    'user'    : os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', ''),
    'port'    : int(os.getenv('DB_PORT', '5432'))
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
