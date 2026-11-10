-- Out-of-sample forecasts produced by the walk-forward backtest.
-- Each row: "as of forecast_date, what did the model expect for the price horizon_days later?"
-- Because every row comes from a model that had NOT seen that period, the replay demo is honest.
CREATE TABLE IF NOT EXISTS forecasts (
    mandi_id        INT  NOT NULL REFERENCES mandis (id),
    commodity_id    INT  NOT NULL REFERENCES commodities (id),
    forecast_date   DATE NOT NULL,
    horizon_days    INT  NOT NULL,
    p_down          NUMERIC(5, 4) NOT NULL,
    p_flat          NUMERIC(5, 4) NOT NULL,
    p_up            NUMERIC(5, 4) NOT NULL,
    predicted_class TEXT NOT NULL CHECK (predicted_class IN ('down', 'flat', 'up')),
    model_version   TEXT NOT NULL,
    PRIMARY KEY (mandi_id, commodity_id, forecast_date, horizon_days, model_version)
);

CREATE INDEX IF NOT EXISTS idx_forecasts_date ON forecasts (forecast_date);
