# WNBA Player-Game Analytics — Ingestion Pipeline + Dashboard

A SQLite database of WNBA player game logs (current season + 2 prior),
at the player-season-game grain, with leakage-safe rolling/season-average
features precomputed, plus a Streamlit dashboard on top.

## Project structure

- `sql/schema.sql` — the schema: `dim_player`, `fact_player_game`,
  `player_game_features`, and two consumption views (`v_dashboard`,
  `v_ml_features`).
- `src/wnba_client.py` — browser-driven wrapper around the (undocumented)
  stats.wnba.com JSON API.
- `src/ingest.py` — pulls real data via `wnba_client.py` and loads it
  into the schema.
- `src/compute_features.py` — rolling/season-average feature computation,
  shared by `ingest.py` and `seed_mock_data.py` so both paths compute
  identically.
- `src/seed_mock_data.py` — generates realistic-but-fake data for testing
  the dashboard without hitting the real API.
- `dashboard/app.py` — the Streamlit dashboard.

## ⚠️ Run this locally, not in a sandboxed environment

`ingest.py` needs real outbound internet access, **and** a real browser
(see below) — it will not work in network-restricted sandboxes.

### Why a browser, not just HTTP requests

stats.wnba.com sits behind Akamai bot-protection. Plain HTTP clients
(curl, Python `requests`) get **silently stalled** — the connection
succeeds but the server never responds. Two things were needed to get
through, found by bisecting failures:

1. **A real Chrome browser**, not Playwright's bundled Chromium — its
   default headless binary ("headless shell") gets stalled the same way
   plain `requests` does.
2. **Hiding `navigator.webdriver`** plus a realistic user agent — without
   this, real Chrome driven via automation still gets an actual
   "Access Denied" page back.

`wnba_client.py` handles both; see the comments there for specifics.

## Setup

```bash
pip install -r requirements.txt
playwright install chromium
```

(Chromium itself isn't used for the actual API calls — see above — but
Playwright needs it installed regardless.)

## 1. Run the real ingestion pipeline

```bash
cd src
python ingest.py --seasons 2026,2025,2024
```

This:
- Fetches the full active roster for the **first** season listed (treated
  as "current") — no need to look up player IDs yourself.
- Backfills each of those players' game logs for every season listed.
- Derives DNP rows by diffing each player's own game log against their
  team's full schedule for that season.
- Rebuilds `data/wnba.db` from scratch each run (so there's no stale mix
  of old and new data) — takes on the order of 15–25 minutes for the
  full active roster (~850 API calls, deliberately rate-limited).

Progress prints per player as it goes; Python buffers stdout when
redirected to a file, so if you've piped output somewhere, don't worry if
it looks quiet for a bit.

## 2. Try it with mock data instead (no live API needed)

```bash
cd src
python seed_mock_data.py --db ../data/wnba_demo.db
cd ../dashboard
streamlit run app.py -- --db ../data/wnba_demo.db
```

## 3. Run the dashboard against real data

```bash
cd dashboard
streamlit run app.py -- --db ../data/wnba.db
```

(Note the `--` before `--db` — that's how Streamlit passes args through
to the script itself rather than consuming them.)

The dashboard shows, per selected player + season:
- Header with bio, season averages, and an L5-vs-season delta indicator
- Scoring threshold hit-rates (10+/15+/20+ pts), split home vs. away
- A rolling hot/cold trend: points per game (colored by home/away) with
  leakage-safe rolling 5-game and trailing-14-day averages overlaid
- Hot/cold markers: a z-score vs. each game's entering (prior-games)
  mean/SD, computed client-side in the dashboard, gated to 5+ prior
  played games
- Rebounds/assists trend, and the full game log

## 4. Query the flat view directly

```sql
SELECT * FROM v_dashboard WHERE player_name = 'A''ja Wilson' ORDER BY game_date;
```

## Known limitations / things to revisit

1. **`started` (starter flag) is always NULL.** `playergamelog` doesn't
   expose it; would need `boxscoretraditionalv2` per game instead (one
   call per game rather than one call per player-season — much slower).

2. **`position_detail` duplicates `main_position`.** The WNBA stats API
   only exposes a general position ("Guard"/"Forward"/"Center"), not
   PG/SG/SF/PF/C granularity, so both columns currently hold the same
   value.

3. **Regular season only** — `ingest.py` doesn't currently pull playoff
   games (`SEASON_TYPE` is hardcoded to `"Regular Season"`).

4. **DNP team assignment uses a per-season roster snapshot.** A player's
   team-of-record for DNP-derived rows comes from a single roster fetch
   for that season, so a mid-season trade could misattribute DNP rows
   around the trade date. Games the player actually played always show
   the correct team (parsed straight from their own game log), so this
   only affects the DNP-derived rows, and only near a trade.

5. **`opp_score` is occasionally NULL** if the opponent's abbreviation
   doesn't resolve to a tracked team for that season (e.g. an unusual
   exhibition game).

6. **Trades mid-season**: for played games, `team` is stored per-game
   (parsed from that game's own matchup), so a trade just shows up as a
   change in the `team` column across consecutive `game_id`s — no special
   handling needed there.

## Design decisions already locked in

- Normalized storage (`dim_player` + `fact_player_game` +
  `player_game_features`), flattened via the `v_dashboard` view for
  querying, and `v_ml_features` for ML export (pre-game context + prior
  rolling features as X, threshold flags as y — no same-row outcome
  columns exposed, so it's structurally impossible to leak by accident).
- DNP rows are kept, not excluded, with all box-score stats set to
  `NULL` — averages skip them rather than treating them as zero.
- `season_avg_*_prior` and all rolling windows (`_l5`, `_l14d`) are
  computed **entering** each game (they exclude that game's own stats) —
  this is what makes them safe to use as predictive features for that
  game's outcome, not just descriptive review numbers.
- `season_avg_*_final` is the completed-season average, repeated on
  every row, for human review only — never use as an ML feature since it
  includes future information relative to earlier rows.
- Game Score is computed directly from box score fields (standard
  Hollinger formula). Usage Rate, PER, and Win Shares were deliberately
  deferred — additive later, not built in now.
