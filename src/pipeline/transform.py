from pyspark.sql import SparkSession
from pyspark.sql import functions as F

def build_spark_session():
    """Initialize Spark session in local mode."""
    spark = SparkSession.builder \
        .appName('WasteIntelligencePipeline') \
        .master('local[*]') \
        .config('spark.sql.shuffle.partitions', '4') \
        .config('spark.driver.memory', '2g') \
        .getOrCreate()
    spark.sparkContext.setLogLevel('WARN')
    return spark

def transform(df_raw):
    """
    Apply transformations to raw annotation DataFrame.
    Adds derived columns for time-based and area-based analysis.
    """
    df_clean = df_raw \
        .filter(F.col('bbox_area') >= 1.0) \
        .filter(F.col('class_name') != 'wastes') \
        .filter(F.col('date').isNotNull()) \
        .withColumn('bbox_area_log',  F.log1p(F.col('bbox_area'))) \
        .withColumn('bbox_area_pct',  F.col('bbox_area') / (F.col('image_width') * F.col('image_height'))) \
        .withColumn('time_of_day',
            F.when(F.col('hour').between(6, 11),  'morning')
             .when(F.col('hour').between(12, 17), 'afternoon')
             .when(F.col('hour').between(18, 23), 'evening')
             .otherwise('night')) \
        .withColumn('month',        F.month(F.col('date'))) \
        .withColumn('week_of_year', F.weekofyear(F.col('date'))) \
        .withColumn('day_of_week',  F.dayofweek(F.col('date')))
    return df_clean

def aggregate_daily(spark, df_clean):
    """Aggregate annotation count and stats per class per day."""
    df_clean.createOrReplaceTempView('clean_annotations')
    return spark.sql("""
        SELECT
            date,
            class_name,
            COUNT(*)                    AS annotation_count,
            ROUND(SUM(bbox_area), 2)    AS total_bbox_area,
            ROUND(AVG(bbox_area), 2)    AS mean_bbox_area,
            ROUND(STDDEV(bbox_area), 2) AS std_bbox_area,
            ROUND(MIN(bbox_area), 2)    AS min_bbox_area,
            ROUND(MAX(bbox_area), 2)    AS max_bbox_area,
            COUNT(DISTINCT image_id)    AS image_count
        FROM clean_annotations
        GROUP BY date, class_name
        ORDER BY date, class_name
    """)

def aggregate_weekly(spark):
    """Aggregate annotation count and stats per class per week."""
    return spark.sql("""
        SELECT
            DATE_TRUNC('week', date)    AS week_start,
            class_name,
            COUNT(*)                    AS annotation_count,
            ROUND(SUM(bbox_area), 2)    AS total_bbox_area,
            ROUND(AVG(bbox_area), 2)    AS mean_bbox_area,
            COUNT(DISTINCT image_id)    AS image_count
        FROM clean_annotations
        GROUP BY DATE_TRUNC('week', date), class_name
        ORDER BY week_start, class_name
    """)
