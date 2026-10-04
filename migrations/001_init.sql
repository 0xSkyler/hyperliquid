-- Every record the engine emits (decisions, fills, raw market data) lands here as JSON.
CREATE TABLE IF NOT EXISTS events (
    id      BIGSERIAL PRIMARY KEY,
    ts      TIMESTAMPTZ NOT NULL,
    stream  TEXT        NOT NULL,
    payload JSONB       NOT NULL
);
CREATE INDEX IF NOT EXISTS events_stream_ts ON events (stream, ts);

CREATE OR REPLACE VIEW decisions AS
SELECT ts, payload->>'action' AS action, payload->>'state' AS state,
       (payload->>'f_current')::float AS f_current, (payload->>'f_target')::float AS f_target,
       (payload->>'expected_edge_bps')::float AS expected_edge_bps,
       payload->>'exec_style' AS exec_style, payload->>'reason' AS reason, payload
FROM events WHERE stream = 'decisions';

CREATE OR REPLACE VIEW fills AS
SELECT ts, (payload->>'is_buy')::bool AS is_buy, (payload->>'px')::float AS px,
       (payload->>'sz')::float AS sz, (payload->>'fee')::float AS fee,
       (payload->>'maker')::bool AS maker
FROM events WHERE stream = 'fills';
