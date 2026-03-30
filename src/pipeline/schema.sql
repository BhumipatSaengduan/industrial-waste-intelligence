-- Daily summary statistics per waste class
CREATE TABLE IF NOT EXISTS daily_waste_summary (
    id              SERIAL PRIMARY KEY,
    date            DATE NOT NULL,
    class_name      VARCHAR(50) NOT NULL,
    annotation_count INTEGER NOT NULL DEFAULT 0,
    total_bbox_area  FLOAT NOT NULL DEFAULT 0,
    mean_bbox_area   FLOAT,
    std_bbox_area    FLOAT,
    min_bbox_area    FLOAT,
    max_bbox_area    FLOAT,
    image_count      INTEGER NOT NULL DEFAULT 0,
    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(date, class_name)
);

-- Weekly summary statistics per waste class
CREATE TABLE IF NOT EXISTS weekly_waste_summary (
    id               SERIAL PRIMARY KEY,
    week_start       DATE NOT NULL,
    class_name       VARCHAR(50) NOT NULL,
    annotation_count INTEGER NOT NULL DEFAULT 0,
    total_bbox_area  FLOAT NOT NULL DEFAULT 0,
    mean_bbox_area   FLOAT,
    image_count      INTEGER NOT NULL DEFAULT 0,
    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(week_start, class_name)
);

-- Anomaly detection results
CREATE TABLE IF NOT EXISTS anomaly_results (
    id              SERIAL PRIMARY KEY,
    date            DATE NOT NULL,
    class_name      VARCHAR(50),
    anomaly_score   FLOAT NOT NULL,
    is_anomaly      BOOLEAN NOT NULL DEFAULT FALSE,
    annotation_count INTEGER,
    explanation     TEXT,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Forecast results
CREATE TABLE IF NOT EXISTS forecast_results (
    id              SERIAL PRIMARY KEY,
    forecast_date   DATE NOT NULL,
    class_name      VARCHAR(50) NOT NULL,
    predicted_count FLOAT NOT NULL,
    lower_bound     FLOAT,
    upper_bound     FLOAT,
    model_name      VARCHAR(50),
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(forecast_date, class_name, model_name)
);

-- LLM generated insights
CREATE TABLE IF NOT EXISTS llm_insights (
    id              SERIAL PRIMARY KEY,
    insight_date    DATE NOT NULL,
    insight_type    VARCHAR(50) NOT NULL,
    content         TEXT NOT NULL,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_daily_date ON daily_waste_summary(date);
CREATE INDEX IF NOT EXISTS idx_daily_class ON daily_waste_summary(class_name);
CREATE INDEX IF NOT EXISTS idx_anomaly_date ON anomaly_results(date);
CREATE INDEX IF NOT EXISTS idx_forecast_date ON forecast_results(forecast_date);
