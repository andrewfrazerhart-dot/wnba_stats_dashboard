-- WNBA Player-Stats Analytics Tool
-- Schema locked in: normalized (dim_player + fact_player_game), no preseason,
-- explicit DNP rows with NULL stats. Rolling/derived features live in a
-- separate player_game_features table, rebuilt by compute_features.py.

PRAGMA foreign_keys = ON;

-- ============================================================
-- dim_player: one row per player (static / slowly-changing bio)
-- ============================================================
CREATE TABLE dim_player (
    player_id           INTEGER PRIMARY KEY,
    player_name          TEXT NOT NULL,
    birthdate            TEXT,            -- ISO date 'YYYY-MM-DD'; age derived per-game
    draft_year           INTEGER,
    draft_position       INTEGER,
    years_experience     INTEGER,          -- can also be derived from draft_year + season
    rookie_flag          INTEGER,          -- 0/1, convenience flag
    height_in            INTEGER,
    main_position        TEXT,             -- Guard / Forward / Center
    position_detail       TEXT              -- PG / SG / SF / PF / C
);

-- ============================================================
-- fact_player_game: one row per player-season-game
-- Grain: (player_id, season, game_id)
-- ============================================================
CREATE TABLE fact_player_game (
    -- ============================================================
    -- ML SAFETY NOTE: this table mixes two kinds of column. For any row,
    -- only the PRE-GAME CONTEXT block below is knowable before tip-off and
    -- safe to use as a predictor of THAT SAME ROW's outcome. The OUTCOME
    -- block is only known after the game ends -- use it as a label/target,
    -- or as history (*_prior features in player_game_features) for FUTURE
    -- rows, never as a same-row predictor.
    -- ============================================================
    game_id              TEXT NOT NULL,     -- source game identifier
    player_id            INTEGER NOT NULL,
    season                INTEGER NOT NULL,
    game_date            TEXT NOT NULL,     -- ISO date
    stage                TEXT NOT NULL CHECK (stage IN ('Regular', 'Playoffs')),

    -- ---- PRE-GAME CONTEXT (known before tip-off; safe as same-row features) ----
    team                  TEXT NOT NULL,     -- per-game, NOT on dim_player (handles trades)
    opponent              TEXT NOT NULL,
    home_away             TEXT NOT NULL CHECK (home_away IN ('H', 'A')),
    days_rest            INTEGER,           -- NULL for player's first game of season
    b2b                   INTEGER,           -- 0/1 back-to-back flag

    -- availability (started is announced pre-game in practice, but treat
    -- cautiously -- lineups can be scratched late; dnp itself is only
    -- fully known post-game)
    dnp                   INTEGER NOT NULL DEFAULT 0,  -- 1 = did not play; stats below NULL
    started               INTEGER,           -- 0/1, NULL if dnp

    -- ---- OUTCOME (only known after the game; label/target, not a same-row feature) ----
    team_score            INTEGER,
    opp_score             INTEGER,
    result                TEXT CHECK (result IN ('W', 'L') OR result IS NULL),

    -- box score -- OUTCOME, same as above (NULL when dnp = 1)
    minutes               REAL,
    pts                   INTEGER,
    reb_off               INTEGER,
    reb_def               INTEGER,
    reb_tot               INTEGER,
    ast                    INTEGER,
    stl                    INTEGER,
    blk                    INTEGER,
    tov                    INTEGER,
    pf                     INTEGER,
    fgm                    INTEGER,
    fga                    INTEGER,
    fg3m                   INTEGER,
    fg3a                   INTEGER,
    ftm                    INTEGER,
    fta                    INTEGER,
    game_score            REAL,             -- can be computed at ingest or in features step

    PRIMARY KEY (player_id, season, game_id),
    FOREIGN KEY (player_id) REFERENCES dim_player(player_id)
);

CREATE INDEX idx_fact_player_season ON fact_player_game(player_id, season);
CREATE INDEX idx_fact_game_date ON fact_player_game(game_date);
CREATE INDEX idx_fact_team ON fact_player_game(team);

-- ============================================================
-- player_game_features: derived / rolling stats, rebuilt by
-- compute_features.py whenever new games are ingested.
-- One row per (player_id, season, game_id), aligned to fact_player_game.
-- ============================================================
CREATE TABLE player_game_features (
    game_id                     TEXT NOT NULL,
    player_id                   INTEGER NOT NULL,
    season                       INTEGER NOT NULL,

    -- ============================================================
    -- LEAKAGE-SAFETY CONVENTION (read before adding new columns):
    --   *_prior   -> computed over games strictly BEFORE this row's
    --               game_date. Current row's own box score is EXCLUDED.
    --               Safe as an ML feature (X) for predicting this row.
    --   *_final   -> the completed, full-season average, same value
    --               repeated on every row. Contains future information
    --               relative to most rows. NEVER use as an ML feature.
    --               Retrospective/display use only.
    --   No column here should include the current row's own game in its
    --   average unless explicitly suffixed _final.
    -- ============================================================

    -- season-to-date, prior games only, vs. final (completed season)
    season_avg_pts_prior        REAL,
    season_avg_pts_final        REAL,
    season_avg_reb_prior        REAL,
    season_avg_reb_final        REAL,
    season_avg_ast_prior        REAL,
    season_avg_ast_final        REAL,
    -- (same _prior / _final pattern repeats for stl, blk, tov, minutes, etc.)

    -- last-5-games rolling (the 5 games immediately preceding this one)
    avg_pts_l5_prior             REAL,
    avg_reb_l5_prior             REAL,
    avg_ast_l5_prior             REAL,

    -- last-14-calendar-days rolling (window ends the day before game_date)
    avg_pts_l14d_prior           REAL,
    avg_reb_l14d_prior           REAL,
    avg_ast_l14d_prior           REAL,

    -- how many games actually back each *_prior average (so a 1-game
    -- "average" isn't treated the same as a 14-game one downstream)
    games_played_season_prior    INTEGER,
    games_played_l14d_prior      INTEGER,

    -- threshold hit-rate flags describe THIS row's own outcome (dashboard
    -- display, or an ML label). Never use as a predictor of itself.
    pts_10plus                   INTEGER,   -- 0/1
    pts_15plus                   INTEGER,
    pts_20plus                   INTEGER,

    PRIMARY KEY (player_id, season, game_id),
    FOREIGN KEY (player_id, season, game_id)
        REFERENCES fact_player_game(player_id, season, game_id)
);

CREATE INDEX idx_features_player_season ON player_game_features(player_id, season);

-- ============================================================
-- CONSUMPTION LAYER: one source of truth above, two views below.
-- Do NOT duplicate the underlying data for dashboard vs. ML use --
-- that creates two copies that can silently drift apart. Instead each
-- consumer reads only the columns appropriate to it, via these views.
-- ============================================================

-- v_dashboard: everything. Safe for retrospective display/reporting
-- (Streamlit). Includes raw outcomes, _final season stats, and _prior
-- rolling stats. Never feed this view's outcome columns into a model
-- as same-row features -- it's for humans looking backward, not for
-- predicting forward.
CREATE VIEW v_dashboard AS
SELECT
    p.player_id, p.player_name, p.main_position, p.position_detail,
    f.*,
    ft.season_avg_pts_prior, ft.season_avg_pts_final,
    ft.season_avg_reb_prior, ft.season_avg_reb_final,
    ft.season_avg_ast_prior, ft.season_avg_ast_final,
    ft.avg_pts_l5_prior, ft.avg_reb_l5_prior, ft.avg_ast_l5_prior,
    ft.avg_pts_l14d_prior, ft.avg_reb_l14d_prior, ft.avg_ast_l14d_prior,
    ft.games_played_season_prior, ft.games_played_l14d_prior,
    ft.pts_10plus, ft.pts_15plus, ft.pts_20plus
FROM fact_player_game f
JOIN dim_player p ON p.player_id = f.player_id
JOIN player_game_features ft
    ON ft.player_id = f.player_id AND ft.season = f.season AND ft.game_id = f.game_id;

-- v_ml_features: ML-safe subset only. Pre-game context + *_prior
-- columns as X, threshold flags as y/label. No raw box-score outcome
-- column from this same row is exposed here -- structurally impossible
-- to leak by accident when reading from this view.
-- Remember: split train/test by season/date, never by random row sample.
CREATE VIEW v_ml_features AS
SELECT
    f.player_id, f.season, f.game_id, f.game_date, f.stage,
    -- pre-game context (known before tip-off)
    f.team, f.opponent, f.home_away, f.days_rest, f.b2b, f.started,
    p.main_position, p.position_detail, p.height_in, p.birthdate, p.draft_year,
    -- prior-only rolling / season-to-date features
    ft.season_avg_pts_prior, ft.season_avg_reb_prior, ft.season_avg_ast_prior,
    ft.avg_pts_l5_prior, ft.avg_reb_l5_prior, ft.avg_ast_l5_prior,
    ft.avg_pts_l14d_prior, ft.avg_reb_l14d_prior, ft.avg_ast_l14d_prior,
    ft.games_played_season_prior, ft.games_played_l14d_prior,
    -- labels (this row's own outcome -- target y, never feed back in as X)
    ft.pts_10plus, ft.pts_15plus, ft.pts_20plus
FROM fact_player_game f
JOIN dim_player p ON p.player_id = f.player_id
JOIN player_game_features ft
    ON ft.player_id = f.player_id AND ft.season = f.season AND ft.game_id = f.game_id
WHERE f.dnp = 0;
