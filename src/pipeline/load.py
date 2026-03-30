import pandas as pd
import psycopg2
from pathlib import Path
from google.cloud import storage

def save_local(df_spark, name, output_dir='data/processed'):
    """Save Spark DataFrame as CSV locally."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    path = f'{output_dir}/{name}.csv'
    df_spark.toPandas().to_csv(path, index=False)
    print(f'Saved locally: {path}')
    return path

def upload_to_gcs(local_path, bucket_name, gcs_path, project='waste-intelligence'):
    """Upload local file to Google Cloud Storage."""
    client = storage.Client(project=project)
    bucket = client.bucket(bucket_name)
    blob   = bucket.blob(gcs_path)
    blob.upload_from_filename(local_path)
    print(f'Uploaded to GCS: gs://{bucket_name}/{gcs_path}')


def load_to_cloudsql(df_spark, table_name, db_config):
    """Load Spark DataFrame into Cloud SQL PostgreSQL table."""
    df_pd   = df_spark.toPandas()
    conn    = psycopg2.connect(**db_config)
    cur     = conn.cursor()
    cols    = ', '.join(df_pd.columns)
    vals    = ', '.join(['%s'] * len(df_pd.columns))
    query   = f'INSERT INTO {table_name} ({cols}) VALUES ({vals}) ON CONFLICT DO NOTHING'
    records = [tuple(row) for row in df_pd.itertuples(index=False)]
    cur.executemany(query, records)
    conn.commit()
    print(f'Loaded {len(records)} rows into {table_name}')
    cur.close()
    conn.close()
