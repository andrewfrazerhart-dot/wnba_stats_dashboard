# WNBA Player Stats Dashboard — Project Status

_Last updated: 2026-07-28_

## What this is

A real-data WNBA player analytics dashboard: a SQLite database built by
scraping stats.wnba.com / wnba.com (no manual data entry), covering the
full active roster's game logs for the current season plus 2 prior
seasons, with a Streamlit dashboard on top for browsing per-player
stats, trends, hit-rates, and a league-wide consistency leaderboard.

Started as a mock-data demo, then moved to real data, then went through
many rounds of dashboard design iteration. Everything below is currently
live and working, committed to git, and backed up to GitHub at
**github.com/andrewfrazerhart-dot/wnba_stats_dashboard**.

## How to run it

```powershell
cd "C:\Users\andre\Desktop\wnba_files\src"
python ingest.py --seasons 2026,2025,2024      # pulls new games since last run (a few minutes)

cd "C:\Users\andre\Desktop\wnba_files\dashboard"
streamlit run app.py -- --db ../data/wnba.db   # opens the dashboard in your browser
```

Full setup/technical details are in `README.md` in this same folder.

## The data pipeline

- **Real data, not mock**: `src/ingest.py` + `src/wnba_client.py` pull
  from the actual WNBA stats API. 205 active players, 3 seasons of
  history, ~15,600+ player-game rows.
- **Getting past bot protection was the hard part**: stats.wnba.com
  silently stalls plain HTTP requests (curl, Python `requests`) — the
  connection succeeds but nothing ever comes back. Getting through
  required driving a real (non-bundled) Chrome browser via Playwright,
  with `navigator.webdriver` hidden and a realistic user agent. Fully
  documented in `README.md` and in `wnba_client.py`'s comments.
- **Incremental by default**: re-running `ingest.py` only re-checks the
  current season and only backfills history for brand-new players —
  ~200 API calls / a few minutes for a routine update, vs. ~1,300+
  calls / 15–25 minutes for a full rebuild (`--full-refresh` flag still
  available if ever needed).
- **Schema**: `dim_player`, `fact_player_game` (played + DNP rows),
  `player_game_features` (precomputed leakage-safe rolling averages),
  `team_schedule` (full-season schedule incl. upcoming games, from a
  separate wnba.com API), `dim_team` (team logo ID lookup).
- **Design principle carried through everywhere**: all rolling/season
  averages are computed *entering* each game (excluding that game's own
  stats), so they're safe to use as predictive features, not just
  retrospective display numbers.

## The dashboard — what it shows

**Player selection (sidebar)**: filter by team, pick a player and
season. A "Compare with another player" checkbox splits the page into
two side-by-side panels.

**Header**: player name + team logo, age, height, position. Season
averages clustered by stat (Points cluster: season/L5/L14D avg, then
Rebounds/Assists), Games Played/Missed/Current Streak grouped together,
and a **"Consistent in: ..."** tag listing which of the 6 major stats
this player currently ranks in the top-30 league-wide for consistency,
with each stat's CV% shown.

**Next Game**: upcoming opponent (with home/away, team logo), their
current record, season PPG for/against, this player's team's PPG, and
the opponent's top 3 players by average Game Score (with position and
PPG).

**Statistical Threshold Hit Rates**: pick any major stat, see hit-rates
against several thresholds, split Home/Away/Overall/Last-5/Last-14-Days.

**Game Log**: full game-by-game table including DNP games, with
per-game season-to-date average, hot/cold z-score, and streak flags.

**Statistical Trends**: per-game value (bar, colored by home/away) with
leakage-safe rolling 5-game and 14-day averages, DNP markers, and a
season-average reference line. (This used to be two separate,
largely-redundant charts — merged into one after review.)

**Hot / Cold Markers**: z-score vs. the player's own to-date mean/SD,
gated to 5+ prior games, with current hot/cold streak counts.

**Consistency Leaderboard for Safer Bets**: league-wide top 30 by
lowest Coefficient of Variation (CV = sample SD ÷ mean) for a
selectable stat — CV rather than raw SD, since raw SD would just
surface low-minute players who score near zero every night as "most
consistent." Includes avg minutes, current played-streak (no DNP),
1-SD range, streak above the range's lower bound, SD, and CV%.
**Clicking a row jumps the whole page to that player's full stats.**

**Every stat-picker dropdown** is available both inline (right above
its chart) and in the sidebar, kept in sync — and most section titles
in the sidebar are clickable links that jump to that chart.

## Predictive points model

Built on top of the real data pipeline: a leakage-safe model that
predicts a player's points distribution for her next game, and powers
a **"Model Fair Probability"** column in the dashboard's Odds
Calculator alongside the market's own devigged probability.

- **Per-minute rate, not raw points**: separates "how well she scores
  when on the floor" from "how much she played," then recombines with
  a simple expected-minutes estimate and a position-relative opponent
  defense index (points allowed to a position, normalized to league
  average, capped at ±20 so one thin-sample team -- like a new
  expansion franchise -- can't swing a prediction too far).
- **Empirical-Bayes shrinkage** of a player's per-minute rate toward a
  cross-season prior -- her own prior season if it clears a
  games-played sanity check (skips injury-shortened seasons
  automatically), else a position-average fallback for rookies.
  Shrinkage strength is *estimated from the data itself* (between- vs.
  within-player variance), not guessed, for both the player-level
  rate/minutes and the opponent defense index.
- **A small, hand-maintained injury watchlist**
  (`MANUAL_OVERRIDES` in `src/points_model.py`) for the handful of
  players where real-world injury news says the automatic rule needs a
  nudge -- e.g. trusting a thin-but-healthy return (Napheesa Collier's
  2026) instead of falling back further, or widening the predicted
  spread for a player playing through nagging injuries (Caitlin
  Clark's 2026) instead of just lowering her mean.
- **Two competing distribution shapes** compared against a naive
  flat-average baseline via a proper walk-forward backtest
  (`src/backtest.py`, chronological train/test split, never random):
  Negative Binomial won on both log loss (0.688 vs. naive's 0.707) and
  Brier score, and calibrates far more consistently than the naive
  model, which is badly overconfident at the extremes.
- Evaluated at a realistic line per player-game (her own predicted
  mean, rounded to the nearest half point), not one fixed threshold
  for every player regardless of role.

Deliberately deferred for now: a full second distribution for minutes
(currently a simple point estimate), pace adjustment on top of the
opponent index, and per-position/per-usage-tier dispersion instead of
one global value.

## Notable technical problems solved along the way

- **Bot-protection bypass** for the ingestion pipeline (see above) —
  the single biggest technical hurdle.
- **A real, silent data bug**: an API parameter (`IsOnlyCurrentSeason`)
  that looked correct was silently dropping most players from
  historical-season queries. Caught by noticing a well-known veteran
  had zero games in a season she definitely played.
- **A real sqlite bug**: `numpy.int64` (from pandas) silently fails to
  match against a plain SQLite `INTEGER` column — no error, just zero
  rows returned. Hit this twice (Next Game lookups, then the
  consistency-tag matching) and now explicitly cast to `int` wherever a
  pandas-derived value is bound as a query parameter.
- **Streamlit widget state ordering**: you can't modify a widget's
  `session_state` after it's already been instantiated in the same run.
  This bit both the "linked dropdown" sync mechanism and the
  leaderboard's click-to-navigate feature — both work around it by
  stashing the pending change and applying it at the very top of the
  next rerun.
- **A pandas/Styler crash**: formatting `None` (vs. `NaN`) with a
  numeric format string throws — found via a real player with zero
  away games this season.

## Known limitations (see README.md for the full list)

- No starter/bench flag (API doesn't expose it cheaply).
- Regular season only, no playoffs yet.
- Position is general (Guard/Forward/Center), not detailed (PG/SG/etc.)
  — the API doesn't expose finer granularity.
- DNP-derived rows use a per-season roster snapshot for team
  assignment, so a rare mid-season trade could misattribute a DNP row
  right around the trade date.

## Where things stand

Everything above is built, tested (dashboard via Streamlit's headless
`AppTest` harness plus spot-checks against known real player stats;
the predictive model via `backtest.py`'s held-out walk-forward
evaluation plus live browser verification), committed, and pushed. No
open/in-progress work as of this writing — the project is in a stable,
working state ready for normal use (run `ingest.py` periodically, then
browse the dashboard).
