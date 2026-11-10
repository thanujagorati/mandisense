-- MandiSense schema (PostgreSQL). Safe to run more than once.

-- Which crop. One row for now (Onion), but the design supports more crops later.
CREATE TABLE IF NOT EXISTS commodities (
    id   SERIAL PRIMARY KEY,
    name TEXT UNIQUE NOT NULL
);
INSERT INTO commodities (name) VALUES ('Onion') ON CONFLICT (name) DO NOTHING;

-- One row per mandi (market). lat/lon are used later to estimate distance and transport cost.
CREATE TABLE IF NOT EXISTS mandis (
    id       SERIAL PRIMARY KEY,
    state    TEXT NOT NULL,
    district TEXT NOT NULL,
    name     TEXT NOT NULL,
    lat      DOUBLE PRECISION,
    lon      DOUBLE PRECISION,
    UNIQUE (state, district, name)
);

-- One row per mandi, per crop, per day. The primary key is what makes re-running
-- the loader safe: the same day can never be stored twice.
CREATE TABLE IF NOT EXISTS price_observations (
    mandi_id           INT  NOT NULL REFERENCES mandis (id),
    commodity_id       INT  NOT NULL REFERENCES commodities (id),
    obs_date           DATE NOT NULL,
    modal_price_rs_qtl NUMERIC(10, 2) NOT NULL CHECK (modal_price_rs_qtl > 0),
    min_price_rs_qtl   NUMERIC(10, 2),
    max_price_rs_qtl   NUMERIC(10, 2),
    arrivals_tonnes    NUMERIC(12, 2) CHECK (arrivals_tonnes >= 0),
    PRIMARY KEY (mandi_id, commodity_id, obs_date)
);

-- Speeds up "all prices in a date range" queries, which the forecaster and the API will use a lot.
CREATE INDEX IF NOT EXISTS idx_obs_date ON price_observations (obs_date);
