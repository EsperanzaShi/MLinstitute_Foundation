-- db-init/01-create-predictions.sql
CREATE TABLE IF NOT EXISTS predictions (
  id         SERIAL PRIMARY KEY,
  ts         TIMESTAMPTZ NOT NULL DEFAULT now(),
  predicted  INT NOT NULL,
  true_label INT NOT NULL
);