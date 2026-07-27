# WNBA Player-Season-Game Analytics — Ingestion Pipeline

This is the data layer for the dashboard: a SQLite database of WNBA
player game logs (2 prior seasons + current), at the player-season-game
grain, with leakage-safe rolling/season-average features precomputed.

## ⚠️ Run this locally, not in a sandboxed environment

`wnba_client.py` calls `stats.wnba.com`, which requires normal outbound
internet access. It will not work in network-restricted sandboxes. Run
it via Claude Code (or any machine with internet access) instead.

## Setup

```bash
pip install -r requirements.txt
```

## 1. Find player IDs and team IDs

You'll need WNBA Stats API `PERSON_ID`s for the players you want to
track, and `TEAM_ID`s for DNP backfill. Options:
- Call `fetch_league_player_index("2026")` in `wnba_client.py` to pull
  the full active-player index (includes IDs, names, teams).
- Or look up a player's ID from their stats.wnba.com URL, e.g.
  `stats.wnba.com/player/<PERSON_ID>/`.

## 2. Run the ingestion pipeline

```bash
cd src
python ingest.py --db ../data/wnba.db --season 2026 \
    --players 1628932,1629483,203399 \
    --team-ids 1611661320,1611661321,1611661313
```

Repeat with `--season 2025` and `--season 2024` to backfill the two
prior seasons (schema is additive — re-running is safe, existing rows
are skipped via `ON CONFLICT ... DO NOTHING`).

## 3. Compute rolling / hit-rate features

```bash
python compute_features.py --db ../data/wnba.db
```

Re-run this any time after loading new games — it fully recomputes
(idempotent), so it's always safe to re-run.

## 4. Query the flat view

Everything (bio + box score + rolling stats) is joined in one place:

```sql
SELECT * FROM player_game_flat WHERE player_name = 'A''ja Wilson' ORDER BY game_date;
```

This is what the Streamlit dashboard should query against.

## Run the test suite first

Before pointing this at the live API, run the mocked-data tests to
confirm the parsing/derived-stat/DNP logic behaves as expected:

```bash
cd src
python test_ingest_logic.py
python test_compute_features.py
```

## Known limitations / things to revisit

1. **`started` (starter flag) is currently always NULL.** The
   `playergamelog` endpoint doesn't reliably expose a starter flag.
   If you need this, pull `boxscoretraditionalv2` per game instead
   (slower — one call per game rather than one call per player-season)
   and merge in `START_POSITION`.

2. **`dnp_reason` defaults to `'Other/Unknown'`.** The WNBA Stats API
   has no structured injury-reason field. Getting real injury
   attribution would require a separate source (e.g. cross-referencing
   injury-report news scraping, or a paid feed like Sportradar's daily
   injuries endpoint) and matching by player + date. Left as a
   follow-up rather than blocking the pipeline.

3. **Season format for the API** (`"2026"` vs `"2025-26"`) may need
   adjusting once you make a real call — different WNBA Stats API
   endpoints are inconsistent about this. Check the first real
   response and adjust `season` formatting in `ingest.py` if needed.

4. **`_result_set_to_dicts` assumes a single relevant result set.**
   A few endpoints (e.g. `leaguedashplayerbiostats`) return exactly
   one; if you add endpoints that return multiple named result sets,
   pass `result_set_name` explicitly.

5. **Trades mid-season**: `team` is stored per-game (not in
   `dim_player`), so a trade just shows up as a change in the `team`
   column across consecutive `game_id`s for the same player — no
   special handling needed, but worth spot-checking once real data
   is loaded.

## Dashboard (Streamlit)

`dashboard/app.py` reads from `player_game_flat` and is agnostic to
whether the underlying data is real or mock -- same schema either way.

**Try it now with mock data** (no live API needed):

```bash
cd src
python seed_mock_data.py --db ../data/wnba_demo.db
cd ../dashboard
streamlit run app.py -- --db ../data/wnba_demo.db
```

(Note the `--` before `--db` -- that's how Streamlit passes args
through to the script itself rather than consuming them.)

Once the real pipeline (`ingest.py` + `compute_features.py`) has been
run against the live API, point the dashboard at that database instead:

```bash
streamlit run app.py -- --db ../data/wnba.db
```

The dashboard currently shows, per selected player + season:
- Header with bio, team, position, season averages, and an L5-vs-season
  delta indicator
- Scoring threshold hit-rates (10+/15+/20+/25+ pts), split home vs. away
- A hot/cold trend chart: points per game (colored by home/away) with
  the leakage-safe rolling 5-game and trailing-14-day averages overlaid
- Rebounds/assists and shooting-efficiency (TS%/eFG%) trend charts
- Full game log table, including DNP rows with reason

## Hot / Cold Markers (SD-based)

In addition to the rolling averages and threshold hit-rates, the schema
tracks whether each game was 1 or 2 standard deviations above/below a
player's own to-date mean, for six stats: points, rebounds, assists,
steals, blocks, and Game Score.

- **Storage**: `fact_player_game_zscore` is a long-format table (one
  row per game × player × stat), not one wide column per stat — this
  keeps the schema extensible (add a 7th tracked stat with zero schema
  changes). `player_game_zscore_wide` pivots it into the one-row-per-game
  shape for actual querying, and is already joined into `player_game_flat`.
- **Leakage-safe**: mean/SD are computed entering each game (only prior
  played games this season), same convention as everything else.
- **Minimum sample size**: requires 5+ prior played games before a
  z-score is computed at all (`ZSCORE_MIN_GAMES` in `compute_features.py`)
  — earlier games get `NULL` rather than a statistically meaningless flag.
- **DNP handling**: a DNP game gets `NULL` z-score/flags and does not
  break or extend a streak — the streak simply carries forward unchanged
  to the next played game.
- **Nesting**: `above_2sd` implies `above_1sd` (not mutually exclusive),
  same convention as the `hit_pts_10/15/20/25` threshold flags.
- **Streaks**: `streak_above_1sd` (etc.) counts consecutive played games,
  ending at that row, satisfying the flag — resets to 0 the moment the
  flag is false. The "current streak" is just the value on the most
  recent row; no separate calculation needed.
- **Hit-rate fractions**: not stored — computed on demand in the
  dashboard (`valid[col].mean()`), same pattern as the point-threshold
  hit-rates.

## Running Standard Deviation

`fact_player_game_rolling` also tracks a running **standard deviation**
for every stat that already gets a rolling average — `season_sd_*_td`
(entering each game, leakage-safe) and `season_sd_*_final` (completed
season, review-only), mirroring the `season_avg_*_td` / `_final` pattern.

Unlike the z-score flags (which require 5+ prior games before turning
on, since a 1/2-SD *flag* needs to be reliable), the raw SD number
itself is shown as soon as it's mathematically possible — starting at
2 prior played games. Expect it to be noisy early in a season (based
on very few games) and to settle down as more games accumulate; that's
expected, not a bug. DNP games are skipped when accumulating it, same
as everywhere else.

## Design decisions already locked in (see project discussion)

- Normalized storage (`dim_player` + `fact_player_game` +
  `fact_player_game_rolling`), flattened via the `player_game_flat`
  view for querying/ML export.
- DNP rows are kept, not excluded, with all box-score stats set to
  `NULL` — averages skip them rather than treating them as zero.
- `season_avg_*_td` and all rolling windows (`_l5`, `_l14d`) are
  computed **entering** each game (they exclude that game's own
  stats) — this is what makes them safe to use as predictive features
  for that game's outcome, not just descriptive review numbers.
- `season_avg_*_final` is the completed-season average, repeated on
  every row, for human review only — never use as an ML feature since
  it includes future information relative to earlier rows.
- Game Score, TS%, eFG% are computed directly from box score fields
  (no additional data source needed). Usage Rate, PER, and Win Shares
  were deliberately deferred — additive later, not built in now.
