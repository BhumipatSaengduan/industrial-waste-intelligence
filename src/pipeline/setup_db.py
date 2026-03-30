import psycopg2
import os

# Connection info จาก Cloud SQL instance
DB_CONFIG = {
    'host'    : '34.104.146.13',  
    'database': 'waste_intelligence',
    'user'    : 'postgres',
    'password': '12345678',   
    'port'    : 5432
}

def setup_database():
    """Create all tables and indexes."""
    conn = psycopg2.connect(**DB_CONFIG)
    cur  = conn.cursor()

    # Read schema file
    with open('src/pipeline/schema.sql') as f:
        schema = f.read()

    cur.execute(schema)
    conn.commit()

    # Verify tables created
    cur.execute("""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
        ORDER BY table_name
    """)
    tables = [row[0] for row in cur.fetchall()]
    print('Tables created:')
    for t in tables:
        print(f'  {t}')

    cur.close()
    conn.close()
    print('\nDatabase setup complete')

if __name__ == '__main__':
    setup_database()
