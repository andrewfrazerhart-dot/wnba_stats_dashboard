"""
WNBA Player Stats Dashboard

Reads from v_dashboard (defined in sql/schema.sql) -- works unchanged
whether pointed at the mock demo database (src/seed_mock_data.py) or a
real database with the same schema.

Run:
    cd dashboard
    streamlit run app.py -- --db ../data/wnba.db
(Streamlit needs the extra `--` before script args.)
"""

import os
import sqlite3
import sys
from datetime import date, datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DEFAULT_DB = "../data/wnba.db"
HOT_COLD_MIN_GAMES = 5  # min prior played games before a hot/cold read is shown
TEAM_LOGO_URL = "https://cdn.wnba.com/logos/wnba/{team_id}/primary/L/logo.svg"
CONSISTENCY_MIN_GAMES = 10  # min played games before a player qualifies for the leaderboard


def get_db_path() -> str:
    if "--db" in sys.argv:
        return sys.argv[sys.argv.index("--db") + 1]
    return DEFAULT_DB


def load_last_updated(db_path: str):
    """Uses the database file's own last-modified time as a proxy for
    "when was ingest.py last run" -- ingest.py is the only thing that
    writes to it."""
    try:
        return datetime.fromtimestamp(os.path.getmtime(db_path))
    except OSError:
        return None


@st.cache_data(ttl=300)
def load_teams(db_path: str) -> list:
    conn = sqlite3.connect(db_path)
    teams = pd.read_sql(
        "SELECT DISTINCT team FROM fact_player_game ORDER BY team", conn
    )["team"].tolist()
    conn.close()
    return teams


@st.cache_data(ttl=300)
def load_player_list(db_path: str, team: str = None) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    if team:
        df = pd.read_sql(
            """SELECT DISTINCT p.player_id, p.player_name
               FROM dim_player p JOIN fact_player_game f ON f.player_id = p.player_id
               WHERE f.team = ? ORDER BY p.player_name""",
            conn, params=(team,),
        )
    else:
        df = pd.read_sql(
            "SELECT DISTINCT player_id, player_name FROM dim_player ORDER BY player_name", conn
        )
    conn.close()
    return df


@st.cache_data(ttl=300)
def load_player_games(db_path: str, player_id: int) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    df = pd.read_sql(
        "SELECT * FROM v_dashboard WHERE player_id = ? ORDER BY game_date",
        conn, params=(player_id,),
    )
    conn.close()
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df


@st.cache_data(ttl=300)
def load_next_game_info(db_path: str, team: str, season: int):
    """Next not-yet-played game for `team`, plus context for the display
    line: opponent's record entering that game, opponent's season PPG
    for/against, and this team's own season PPG. Returns None if there's
    no upcoming game left in team_schedule for this season (e.g. season
    over, or team_schedule not yet populated by a run of ingest.py)."""
    season = int(season)  # sqlite3 silently fails to match numpy.int64 against an INTEGER column
    conn = sqlite3.connect(db_path)

    next_game = conn.execute(
        """SELECT game_date, opponent, home_away FROM team_schedule
           WHERE team = ? AND season = ? AND status = 'Scheduled'
           ORDER BY game_date ASC LIMIT 1""",
        (team, season),
    ).fetchone()
    if next_game is None:
        conn.close()
        return None
    game_date, opponent, home_away = next_game

    opp_record = conn.execute(
        """SELECT team_wins, team_losses FROM team_schedule
           WHERE team = ? AND season = ? AND game_date = ? AND opponent = ?""",
        (opponent, season, game_date, team),
    ).fetchone()
    opp_ppg_for, opp_ppg_against = conn.execute(
        """SELECT AVG(team_score), AVG(opp_score) FROM team_schedule
           WHERE team = ? AND season = ? AND status = 'Final'""",
        (opponent, season),
    ).fetchone()
    (team_ppg_for,) = conn.execute(
        """SELECT AVG(team_score) FROM team_schedule
           WHERE team = ? AND season = ? AND status = 'Final'""",
        (team, season),
    ).fetchone()

    conn.close()
    return dict(
        game_date=game_date, opponent=opponent, home_away=home_away,
        opp_wins=opp_record[0] if opp_record else None,
        opp_losses=opp_record[1] if opp_record else None,
        opp_ppg_for=opp_ppg_for, opp_ppg_against=opp_ppg_against,
        team_ppg_for=team_ppg_for,
    )


@st.cache_data(ttl=300)
def load_top_performers(db_path: str, team: str, season: int, n: int = 3):
    """Top n players on `team` this season by average Game Score (played
    games only) -- used to preview the upcoming opponent's best players.
    Also pulls position and PPG so the display line can show more than
    just a name."""
    season = int(season)  # sqlite3 silently fails to match numpy.int64 against an INTEGER column
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        """SELECT p.player_name, p.main_position, AVG(f.game_score) AS avg_gs, AVG(f.pts) AS avg_pts
           FROM fact_player_game f JOIN dim_player p ON p.player_id = f.player_id
           WHERE f.team = ? AND f.season = ? AND f.dnp = 0
           GROUP BY f.player_id
           ORDER BY avg_gs DESC
           LIMIT ?""",
        (team, season, n),
    ).fetchall()
    conn.close()
    return rows


def _leaderboard_player_row(g: pd.DataFrame) -> pd.Series:
    """Per-player aggregation for one stat, given ALL of that player's
    rows this season (played + DNP, needed for the no-DNP streak)."""
    g = g.sort_values("game_date")
    played = g[g["dnp"] == 0]
    n_games = len(played)
    avg = played["stat_val"].mean() if n_games else None
    sd = played["stat_val"].std() if n_games else None  # sample SD (ddof=1): this season is a sample, not the full population
    avg_minutes = played["minutes"].mean() if n_games else None

    # current streak of consecutive games PLAYED (no DNP), counting back
    # from the most recent game of the season -- same convention as the
    # single-player "Current Played Streak" metric above.
    played_streak_val = 0
    for dnp in g["dnp"].iloc[::-1]:
        if dnp == 0:
            played_streak_val += 1
        else:
            break

    # current streak of consecutive PLAYED games at/above (mean - 1 SD),
    # counting back from the most recent played game -- DNPs don't break
    # it, same "carries forward" convention used for hot/cold streaks
    # elsewhere in this dashboard.
    low = high = above_low_streak = None
    if n_games and pd.notna(sd):
        low, high = avg - sd, avg + sd
        above_low_streak = 0
        for val in played["stat_val"].iloc[::-1]:
            if val >= low:
                above_low_streak += 1
            else:
                break

    return pd.Series({
        "player_name": g["player_name"].iloc[0], "position": g["main_position"].iloc[0],
        "team": g["team"].iloc[-1], "games": n_games, "avg_minutes": avg_minutes,
        "played_streak": played_streak_val, "avg": avg, "range_low": low, "range_high": high,
        "above_low_streak": above_low_streak, "sd": sd,
    })


@st.cache_data(ttl=300)
def load_consistency_leaderboard(db_path: str, stat_col: str, season: int,
                                  min_games: int = CONSISTENCY_MIN_GAMES, n: int = 30) -> pd.DataFrame:
    """Top n players league-wide this season ranked by lowest Coefficient
    of Variation (CV = sample SD / mean) for `stat_col` -- CV rather than
    raw SD so a star's spread is measured relative to their own scoring
    level, not against everyone else's absolute numbers (otherwise this
    would just surface bench players who barely play and score near
    zero every night). min_games filters out small-sample noise; a
    player who never records the stat at all (avg == 0) is dropped too,
    since CV is undefined for them (division by zero)."""
    season = int(season)  # sqlite3 silently fails to match numpy.int64 against an INTEGER column
    conn = sqlite3.connect(db_path)
    df = pd.read_sql(
        f"""SELECT f.player_id, p.player_name, p.main_position, f.team, f.game_date, f.dnp,
                   f.minutes, f.{stat_col} AS stat_val
            FROM fact_player_game f JOIN dim_player p ON p.player_id = f.player_id
            WHERE f.season = ?""",
        conn, params=(season,),
    )
    conn.close()
    if df.empty:
        return pd.DataFrame()

    agg = df.groupby("player_id").apply(_leaderboard_player_row, include_groups=False)
    agg = agg[(agg["games"] >= min_games) & (agg["avg"] > 0)].copy()
    agg["cv_pct"] = (agg["sd"] / agg["avg"]) * 100
    return agg.sort_values("cv_pct", ascending=True).head(n).reset_index(drop=True)


CONSISTENCY_STAT_LABELS = {"Points": "Scoring", "Rebounds": "Rebounding", "Assists": "Assists",
                           "Steals": "Steals", "Blocks": "Blocks", "Turnovers": "Turnovers"}


@st.cache_data(ttl=300)
def get_consistency_tags(db_path: str, player_name: str, season: int) -> list:
    """Which of the 6 major-stat top-30 Consistency Leaderboards (see
    render_consistency_leaderboard) this player currently appears on, as
    (friendly_label, cv_pct) pairs -- e.g. [("Scoring", 22.6), ...].
    Matched by name since load_consistency_leaderboard's output doesn't
    retain player_id (dropped as the groupby key); WNBA active rosters
    don't have duplicate names, so this is safe in practice."""
    tags = []
    for stat_label, stat_col in MAJOR_STATS.items():
        board = load_consistency_leaderboard(db_path, stat_col, season)
        if board.empty:
            continue
        match = board[board["player_name"] == player_name]
        if not match.empty:
            tags.append((CONSISTENCY_STAT_LABELS[stat_label], match.iloc[0]["cv_pct"]))
    return tags


@st.cache_data(ttl=300)
def load_player_bio(db_path: str, player_id: int):
    """birthdate + height_in -- not in v_dashboard (which only pulls
    player_id/name/position from dim_player), so a separate lookup."""
    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT birthdate, height_in FROM dim_player WHERE player_id = ?", (player_id,)
    ).fetchone()
    conn.close()
    return row if row else (None, None)


@st.cache_data(ttl=300)
def load_team_id(db_path: str, team: str):
    conn = sqlite3.connect(db_path)
    row = conn.execute("SELECT team_id FROM dim_team WHERE team = ?", (team,)).fetchone()
    conn.close()
    return row[0] if row else None


def team_logo_url(team_id):
    return TEAM_LOGO_URL.format(team_id=team_id) if team_id else None


def compute_age(birthdate_str):
    if not birthdate_str:
        return None
    b = date.fromisoformat(birthdate_str)
    today = date.today()
    return today.year - b.year - ((today.month, today.day) < (b.month, b.day))


def format_height(height_in):
    if height_in is None:
        return None
    feet, inches = divmod(int(height_in), 12)
    return f"{feet}'{inches}\""


def add_hot_cold(played: pd.DataFrame, stat: str) -> pd.DataFrame:
    """Leakage-safe z-score vs. each game's entering (prior-games) mean/SD,
    computed here in the app since the schema only stores prior *averages*,
    not SD. Requires HOT_COLD_MIN_GAMES prior played games before a z-score
    is shown -- otherwise it's too noisy to mean anything."""
    played = played.sort_values("game_date").reset_index(drop=True)
    prior_mean = played[stat].expanding().mean().shift(1)
    prior_std = played[stat].expanding().std().shift(1)
    n_prior = played.index  # row i has i prior played games (0-indexed, all rows are played-only)

    z = (played[stat] - prior_mean) / prior_std
    z[n_prior < HOT_COLD_MIN_GAMES] = pd.NA
    z[prior_std == 0] = pd.NA

    played["hot_cold_z"] = z
    played["is_hot"] = played["hot_cold_z"] >= 1
    played["is_cold"] = played["hot_cold_z"] <= -1
    return played


def add_dnp_markers(fig, dnp_games: pd.DataFrame, y_value=0):
    """Adds a visible marker for each DNP game at a fixed y (default 0) so
    missed games show up on the same timeline as everything else, instead
    of just silently vanishing from the chart."""
    if dnp_games.empty:
        return
    fig.add_scatter(
        x=dnp_games["game_date"], y=[y_value] * len(dnp_games), mode="markers",
        name="DNP", marker=dict(symbol="x", size=11, color="black", line=dict(width=2)),
        text=[f"DNP vs {opp}" for opp in dnp_games["opponent"]], hoverinfo="text+x",
    )


def synced_selectbox(container, label, options, shared_key, widget_key, anchor=None):
    """A selectbox that can be rendered in multiple places at once (e.g.
    the sidebar AND inline in one or two panels) while staying in sync --
    changing any instance updates the others on the next rerun.

    Streamlit normally lets a widget's own session_state (tied to its
    `key`) "win" over anything else once that key has been used, so
    simply passing `index=` from a shared value only works on the very
    first render. Instead, this pushes the current shared value into
    this instance's own key right before creating it, every rerun --
    that's what makes every instance track the shared value, not just
    whichever one the user directly touched.

    `anchor`, if given, renders the label as a clickable in-page link to
    that chart's heading instead of a plain (non-clickable) widget label --
    used for the sidebar copies so clicking the section name jumps to it."""
    if shared_key not in st.session_state:
        st.session_state[shared_key] = options[0]
    st.session_state[widget_key] = st.session_state[shared_key]

    def _sync():
        st.session_state[shared_key] = st.session_state[widget_key]

    if anchor:
        container.markdown(f"[{label}](#{anchor})")
        return container.selectbox(label, options, key=widget_key, on_change=_sync,
                                    label_visibility="collapsed")
    return container.selectbox(label, options, key=widget_key, on_change=_sync)


MAJOR_STATS = {"Points": "pts", "Rebounds": "reb_tot", "Assists": "ast",
               "Steals": "stl", "Blocks": "blk", "Turnovers": "tov"}

THRESHOLD_STAT_OPTIONS = {
    "Points": ("pts", (5, 10, 15, 20, 25, 30), "pts"),
    "Rebounds": ("reb_tot", (2, 4, 6, 8, 10), "reb"),
    "Assists": ("ast", (2, 4, 6, 8), "ast"),
    "Steals": ("stl", (1, 2, 3), "stl"),
    "Blocks": ("blk", (1, 2, 3), "blk"),
    "Turnovers": ("tov", (1, 2, 3, 4), "tov"),
}

HOTCOLD_STAT_OPTIONS = {"Points": "pts", "Rebounds": "reb_tot", "Assists": "ast",
                         "Steals": "stl", "Blocks": "blk", "Game Score": "game_score"}


def rolling_prior_averages(season_df: pd.DataFrame, stat_col: str) -> pd.DataFrame:
    """Leakage-safe L5/L14D averages entering each game (played or DNP),
    computed the same way as compute_features.py's prior-averages: only
    games strictly before the current row count. DNP games are skipped
    when accumulating but still get a value here (using whatever the
    average was heading into that game), so the line continues across
    DNP gaps instead of stopping. Only points/rebounds/assists have this
    precomputed in the database -- this covers every other stat too."""
    df = season_df.sort_values("game_date").reset_index(drop=True)
    played_so_far = []  # list of (date, value), non-DNP only
    l5_vals, l14d_vals = [], []
    for _, row in df.iterrows():
        window_start = row["game_date"] - pd.Timedelta(days=14)
        last5 = played_so_far[-5:]
        l14d = [v for d, v in played_so_far if d >= window_start]
        l5_vals.append(sum(v for _, v in last5) / len(last5) if last5 else None)
        l14d_vals.append(sum(l14d) / len(l14d) if l14d else None)
        if row["dnp"] == 0:
            played_so_far.append((row["game_date"], row[stat_col]))
    df["_l5_prior"] = l5_vals
    df["_l14d_prior"] = l14d_vals
    return df


def streak(flags: pd.Series) -> int:
    """Current (most recent) consecutive-True streak, reading backward from the end."""
    count = 0
    for v in flags.iloc[::-1]:
        if v is True:
            count += 1
        elif v is False:
            break
        else:  # NA -- not enough history yet, streak stops here
            break
    return count


def played_streak(season_df: pd.DataFrame) -> int:
    """Current number of consecutive games PLAYED, counting back from the
    most recent game in the season -- resets to 0 the moment a DNP is hit."""
    ordered = season_df.sort_values("game_date")
    count = 0
    for dnp in ordered["dnp"].iloc[::-1]:
        if dnp == 0:
            count += 1
        else:
            break
    return count


def hit_rate_table(played: pd.DataFrame, stat_col: str, thresholds, label: str) -> pd.DataFrame:
    """Hit-rate for each threshold, split several ways: overall, home/away,
    and recent form (last 5 played games, last 14 calendar days) -- the
    same L5/L14D windows used elsewhere in the dashboard, so this lines up
    with the rolling-average and streak charts. Computed directly from the
    raw stat column (not precomputed flag columns) so any threshold for
    any stat can be added here without a schema change."""
    played = played.sort_values("game_date")
    last5 = played.tail(5)
    most_recent_date = played["game_date"].max()
    last14d = played[played["game_date"] >= most_recent_date - pd.Timedelta(days=14)]

    rows = []
    for t in thresholds:
        hit = played[stat_col] >= t
        home_mask = played.home_away == "H"
        away_mask = played.home_away == "A"
        # NaN, not None -- pandas' Styler.format() crashes trying to apply
        # a numeric format string to a plain None (found via a real
        # low-game-count player who had zero away games)
        rows.append({
            "Threshold": f"{t}+ {label}",
            "Overall": hit.mean() * 100 if len(played) else float("nan"),
            "Home": hit[home_mask].mean() * 100 if home_mask.any() else float("nan"),
            "Away": hit[away_mask].mean() * 100 if away_mask.any() else float("nan"),
            "Last 5": (last5[stat_col] >= t).mean() * 100 if len(last5) else float("nan"),
            "Last 14 Days": (last14d[stat_col] >= t).mean() * 100 if len(last14d) else float("nan"),
        })
    return pd.DataFrame(rows)


def american_odds_to_implied_prob(odds: float) -> float:
    """Standard American-odds-to-probability conversion. Includes the
    sportsbook's vig -- this is the RAW implied probability, not the
    fair one (see devig_two_way)."""
    if odds >= 0:
        return 100.0 / (odds + 100.0)
    return -odds / (-odds + 100.0)


def devig_two_way(over_odds: float, under_odds: float):
    """Removes the vig via the standard proportional method: normalize
    both sides' raw implied probabilities so they sum to 100%. Simpler
    than Shin's method or a power devig, but appropriate for a manual,
    phase-1 tool -- "simpler is better all else equal." Returns
    (fair_over_prob, fair_under_prob), each in [0, 1]."""
    p_over_raw = american_odds_to_implied_prob(over_odds)
    p_under_raw = american_odds_to_implied_prob(under_odds)
    overround = p_over_raw + p_under_raw
    return p_over_raw / overround, p_under_raw / overround


def render_player_panel(db_path, player_id, season, compact=False, panel_key="p1"):
    """Renders one player's full panel: header, all charts, game log.
    `compact=True` (used in compare mode, where two of these sit side by
    side) tucks the game log behind a collapsed expander so two full game
    logs don't turn the page into a wall of scrolling. `panel_key` gives
    every widget a unique key -- needed because compare mode can show the
    same player/season on both sides (e.g. comparing two seasons of the
    same player isn't actually asked for here, but nothing should break
    if two panels happen to render identical content).

    The four stat-picker dropdowns are rendered inline here (same spot
    as before compare mode existed) AND in the sidebar -- both are kept
    in sync via synced_selectbox()."""
    df = load_player_games(db_path, player_id)
    season_df = df[df.season == season].reset_index(drop=True)
    played = season_df[season_df.dnp == 0].copy()
    dnp_games = season_df[season_df.dnp == 1].copy()

    if played.empty:
        st.warning("No played games for this player/season yet.")
        return

    bio = season_df.iloc[0]
    player_team = played.iloc[-1].team

    # ---------------------------------------------------------------
    # Header
    # ---------------------------------------------------------------
    birthdate, height_in = load_player_bio(db_path, player_id)
    age = compute_age(birthdate)
    height_str = format_height(height_in)

    logo_url = team_logo_url(load_team_id(db_path, player_team))
    logo_html = f'<img src="{logo_url}" style="height:2.25rem; margin-left:10px;">' if logo_url else ""

    consistency_tags = get_consistency_tags(db_path, bio.player_name, season)
    consistency_html = ""
    if consistency_tags:
        tags_str = ", ".join(f"{label} ({cv:.1f}%)" for label, cv in consistency_tags)
        consistency_html = (
            f'<span style="font-size:1.1rem; margin-left:18px; opacity:0.8;">'
            f'Consistent in: {tags_str}</span>'
        )

    st.markdown(
        f'<div id="player-panel-{panel_key}" style="display:flex; align-items:center; flex-wrap:wrap;">'
        f'<h1 style="margin:0; padding:0;">{bio.player_name}</h1>{logo_html}{consistency_html}</div>',
        unsafe_allow_html=True,
    )
    caption_parts = [player_team, f"{bio.position_detail} ({bio.main_position})"]
    if age is not None:
        caption_parts.append(f"{age} yrs")
    if height_str:
        caption_parts.append(height_str)
    caption_parts.append(f"{season} season")
    st.caption(" · ".join(caption_parts))

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Season Avg PTS", f"{played.pts.mean():.1f}")

    last5_pts = played.tail(5)["pts"].mean()
    delta5 = last5_pts - played.pts.mean()
    m2.metric("L5 Avg PTS", f"{last5_pts:.1f}", delta=f"{delta5:+.1f} vs season")

    most_recent_date = played["game_date"].max()
    last14d_pts = played.loc[played["game_date"] >= most_recent_date - pd.Timedelta(days=14), "pts"].mean()
    delta14 = last14d_pts - played.pts.mean()
    m3.metric("L14D Avg PTS", f"{last14d_pts:.1f}", delta=f"{delta14:+.1f} vs season")

    m4.metric("Season Avg REB", f"{played.reb_tot.mean():.1f}")
    m5.metric("Season Avg AST", f"{played.ast.mean():.1f}")

    # narrower columns (+ an unused spacer) so these 3 cluster together on
    # the left instead of stretching across the full row width like the
    # rows above/below
    m6, m7, m8, _spacer = st.columns([1, 1, 1, 3])
    m6.metric("Games Played", f"{len(played)}")
    m7.metric("Games Missed (DNP)", f"{(season_df.dnp == 1).sum()}")
    m8.metric("Current Played Streak", f"{played_streak(season_df)}")

    next_game = load_next_game_info(db_path, player_team, season)
    st.markdown("**Next Game**")
    if next_game:
        opp = next_game["opponent"]
        opp_logo_url = team_logo_url(load_team_id(db_path, opp))
        opp_logo_html = (f'<img src="{opp_logo_url}" style="height:1.4rem; margin-left:6px; '
                          f'vertical-align:middle;">' if opp_logo_url else "")
        ng1, ng2, ng3, ng4, ng5 = st.columns(5)
        ng1.markdown(
            f'<div style="font-size:0.875rem; color:rgb(120,120,120);">Opponent</div>'
            f'<div style="font-size:1.75rem; font-weight:600; line-height:1.2;">'
            f'{opp} ({next_game["home_away"]}){opp_logo_html}</div>',
            unsafe_allow_html=True,
        )
        ng2.metric(f"{opp} Record",
                   f"{next_game['opp_wins']}-{next_game['opp_losses']}"
                   if next_game["opp_wins"] is not None else "N/A")
        ng3.metric(f"{opp} PPG",
                   f"{next_game['opp_ppg_for']:.1f}" if next_game["opp_ppg_for"] is not None else "N/A")
        ng4.metric(f"{opp} PPG Allowed",
                   f"{next_game['opp_ppg_against']:.1f}" if next_game["opp_ppg_against"] is not None else "N/A")
        ng5.metric(f"{player_team} PPG",
                   f"{next_game['team_ppg_for']:.1f}" if next_game["team_ppg_for"] is not None else "N/A")
        st.caption(f"Game date: {next_game['game_date']}")

        top_performers = load_top_performers(db_path, opp, season)
        if top_performers:
            names = ", ".join(
                f"{name} ({position}) {avg_pts:.1f} PPG"
                for name, position, avg_gs, avg_pts in top_performers
            )
            st.caption(f"{opp} top performers: {names} *(ranked by Game Score)*")
    else:
        st.caption("No upcoming games scheduled (run ingest.py to refresh the schedule, "
                   "or the season may be over).")

    st.divider()

    # ---------------------------------------------------------------
    # Threshold hit-rates
    # ---------------------------------------------------------------
    st.subheader("Statistical Threshold Hit Rates", anchor=f"threshold-hit-rates-{panel_key}")
    threshold_stat_label = synced_selectbox(
        st, "Stat", list(THRESHOLD_STAT_OPTIONS.keys()),
        "threshold_stat_shared", f"threshold_stat_inline_{panel_key}")
    threshold_col, thresholds, threshold_short_label = THRESHOLD_STAT_OPTIONS[threshold_stat_label]

    hr = hit_rate_table(played, threshold_col, thresholds, threshold_short_label)
    fig = go.Figure()
    fig.add_bar(name="Home", x=hr["Threshold"], y=hr["Home"])
    fig.add_bar(name="Away", x=hr["Threshold"], y=hr["Away"])
    fig.add_bar(name="Overall", x=hr["Threshold"], y=hr["Overall"], marker_color="lightgray", opacity=0.6)
    fig.add_bar(name="Last 5", x=hr["Threshold"], y=hr["Last 5"], marker_color="orange")
    fig.add_bar(name="Last 14 Days", x=hr["Threshold"], y=hr["Last 14 Days"], marker_color="red")
    fig.update_layout(barmode="group", yaxis_title="Hit rate (%)", yaxis_range=[0, 100], height=420,
                       legend=dict(orientation="h", y=1.15))
    st.plotly_chart(fig, width='stretch', key=f"threshold_chart_{panel_key}")

    st.dataframe(
        hr.style.format({c: "{:.0f}%" for c in ["Overall", "Home", "Away", "Last 5", "Last 14 Days"]},
                        na_rep="N/A"),
        hide_index=True, width='stretch', key=f"threshold_table_{panel_key}",
    )

    # ---------------------------------------------------------------
    # Odds calculator (manual entry, phase 1 of the predictive-model
    # project discussed separately) -- devigs a sportsbook line against
    # this player's actual history for whichever stat is selected above.
    # ---------------------------------------------------------------
    st.markdown("**Odds Calculator (manual entry)**")
    st.caption("Enter a sportsbook line and American odds to compare this player's actual history "
               "against the market's no-vig (devigged) fair probability. Uses the same stat as the "
               "table above, so switching the Stat picker updates this too.")

    default_line = round(played[threshold_col].mean() * 2) / 2
    oc1, oc2, oc3 = st.columns(3)
    odds_line = oc1.number_input(
        f"{threshold_stat_label} line", value=float(default_line), step=0.5, format="%.1f",
        # keyed per-stat (not just per-panel): a number_input remembers its
        # own value once created and ignores new `value=` defaults on later
        # runs, so without this a Points-appropriate line would silently
        # stick around after switching to Rebounds via the Stat picker above
        key=f"odds_line_{panel_key}_{threshold_col}")
    over_odds = oc2.number_input(
        "Over odds (American)", value=-110, step=5, key=f"odds_over_{panel_key}")
    under_odds = oc3.number_input(
        "Under odds (American)", value=-110, step=5, key=f"odds_under_{panel_key}")

    fair_over, fair_under = devig_two_way(over_odds, under_odds)
    fm1, fm2 = st.columns(2)
    fm1.metric("Market Fair Prob (Over)", f"{fair_over * 100:.1f}%")
    fm2.metric("Market Fair Prob (Under)", f"{fair_under * 100:.1f}%")

    odds_last5 = played.tail(5)
    odds_most_recent_date = played["game_date"].max()
    odds_last14d = played[played["game_date"] >= odds_most_recent_date - pd.Timedelta(days=14)]

    odds_rows = []
    for window_label, window_df in [("Season", played), ("Last 5", odds_last5), ("Last 14 Days", odds_last14d)]:
        your_over = (window_df[threshold_col] > odds_line).mean() * 100 if len(window_df) else float("nan")
        odds_rows.append({
            "Window": window_label, "Games": len(window_df), "Your Over %": your_over,
            "Market Fair Over %": fair_over * 100, "Edge (pts)": your_over - fair_over * 100,
        })
    st.dataframe(
        pd.DataFrame(odds_rows).style.format({
            "Your Over %": "{:.1f}%", "Market Fair Over %": "{:.1f}%", "Edge (pts)": "{:+.1f}",
        }, na_rep="N/A"),
        hide_index=True, width='stretch', key=f"odds_calc_table_{panel_key}",
    )
    st.caption("Edge = your historical Over rate minus the market's devigged fair Over probability, "
               "in percentage points. Positive means your data says Over hits more often than the "
               "market implies. A whole-number line can push in real betting (not modeled here) -- "
               "use a half-point line like sportsbooks do to avoid that.")

    st.divider()

    # ---------------------------------------------------------------
    # Full game log -- reads the Hot/Cold Markers stat straight from
    # session_state rather than rendering another dropdown here, since
    # that section (further down) already owns the actual "Stat" picker
    # for this; both read/write the same shared key so they always agree.
    # ---------------------------------------------------------------
    hotcold_stat_label_for_log = st.session_state.get("hotcold_stat_shared", "Points")
    hc_for_log = add_hot_cold(played, HOTCOLD_STAT_OPTIONS[hotcold_stat_label_for_log])

    log_source = season_df.merge(
        hc_for_log[["game_id", "hot_cold_z", "is_hot", "is_cold"]], on="game_id", how="left",
    )
    log_df = log_source[[
        "game_date", "opponent", "home_away", "result", "dnp",
        "minutes", "pts", "reb_tot", "ast", "stl", "blk", "tov", "game_score",
        "season_avg_pts_prior", "avg_pts_l5_prior", "avg_pts_l14d_prior",
        "hot_cold_z", "is_hot", "is_cold",
    ]].sort_values("game_date", ascending=False).copy()
    log_df["game_date"] = log_df["game_date"].dt.strftime("%Y-%m-%d")
    log_df["dnp"] = log_df["dnp"].map({1: "Yes", 0: ""})
    log_df["season_avg_pts_prior"] = log_df["season_avg_pts_prior"].round(1)
    log_df["hot_cold_z"] = log_df["hot_cold_z"].round(2)
    log_df = log_df.rename(columns={
        "dnp": "DNP", "hot_cold_z": f"{hotcold_stat_label_for_log} z", "is_hot": "Hot", "is_cold": "Cold",
        "season_avg_pts_prior": "PTS season avg (to date)",
        "avg_pts_l5_prior": "PTS L5 avg", "avg_pts_l14d_prior": "PTS L14d avg",
    })

    if compact:
        # expanders don't support an anchor -- a bare id div gives the
        # sidebar "Game Log" link something to jump to even in compare mode
        st.markdown(f'<div id="game-log-{panel_key}"></div>', unsafe_allow_html=True)
        with st.expander("Game Log", key=f"gamelog_expander_{panel_key}"):
            st.dataframe(log_df, hide_index=True, width='stretch', height=400, key=f"gamelog_{panel_key}")
    else:
        st.subheader("Game Log", anchor=f"game-log-{panel_key}")
        st.caption("Includes DNP games (all box-score stats blank) so you can see missed games "
                   "relative to games actually played.")
        st.dataframe(log_df, hide_index=True, width='stretch', height=400, key=f"gamelog_{panel_key}")

    st.divider()

    # ---------------------------------------------------------------
    # Statistical trend (single stat, selectable). This used to be two
    # separate sections (a line chart, and a bar chart with home/away
    # coloring) that plotted the same underlying data -- raw per-game
    # value, L5/L14D rolling averages, DNP markers -- just styled
    # differently. Merged into one: the bar version, since home/away
    # coloring and the season-average line are strictly more informative
    # than the plain line chart was.
    # ---------------------------------------------------------------
    st.subheader("Statistical Trends", anchor=f"rolling-hot-cold-trend-{panel_key}")
    hotcoldtrend_stat_label = synced_selectbox(
        st, "Stat", list(MAJOR_STATS.keys()),
        "hotcoldtrend_stat_shared", f"hotcoldtrend_stat_inline_{panel_key}")
    hotcoldtrend_stat_col = MAJOR_STATS[hotcoldtrend_stat_label]
    st.caption(f"{hotcoldtrend_stat_label} per game vs. rolling 5-game and trailing 14-day averages "
               f"(entering each game, leakage-safe)")

    hc_rolling_df = rolling_prior_averages(season_df, hotcoldtrend_stat_col)

    trend = go.Figure()
    trend.add_bar(x=played["game_date"], y=played[hotcoldtrend_stat_col],
                  name=f"{hotcoldtrend_stat_label} (that game)",
                  marker_color=["#2ca02c" if h == "H" else "#1f77b4" for h in played["home_away"]])
    trend.add_scatter(x=hc_rolling_df["game_date"], y=hc_rolling_df["_l5_prior"], mode="lines",
                       name="Last 5 games avg (entering)", line=dict(color="orange", width=2))
    trend.add_scatter(x=hc_rolling_df["game_date"], y=hc_rolling_df["_l14d_prior"], mode="lines",
                       name="Last 14 days avg (entering)", line=dict(color="red", width=2, dash="dot"))
    add_dnp_markers(trend, dnp_games)
    trend.add_hline(y=played[hotcoldtrend_stat_col].mean(), line_dash="dash", line_color="gray",
                     annotation_text="Season avg (final)")
    trend.update_layout(height=420, yaxis_title=hotcoldtrend_stat_label, legend=dict(orientation="h", y=1.1))
    st.plotly_chart(trend, width='stretch', key=f"hotcoldtrend_chart_{panel_key}")
    st.caption("Green bar = home game, blue bar = away game. Black X = DNP. Rolling lines reflect "
               "games *entering* each date (leakage-safe) and continue across DNP gaps, since a "
               "missed game doesn't change the average.")

    st.divider()

    # ---------------------------------------------------------------
    # Hot / Cold markers (z-score vs. entering-game mean/SD)
    # ---------------------------------------------------------------
    st.subheader("Hot / Cold Markers", anchor=f"hot-cold-markers-{panel_key}")
    st.caption("Games flagged ≥1 SD above/below the player's own to-date mean, entering that game "
               "(leakage-safe) — plus the current streak of consecutive hot/cold games.")

    hotcold_stat_label = synced_selectbox(
        st, "Stat", list(HOTCOLD_STAT_OPTIONS.keys()),
        "hotcold_stat_shared", f"hotcold_stat_inline_{panel_key}")
    hotcold_stat_col = HOTCOLD_STAT_OPTIONS[hotcold_stat_label]
    hc = add_hot_cold(played, hotcold_stat_col)
    valid = hc[hc["hot_cold_z"].notna()]
    n_gated_out = len(hc) - len(valid)

    hr1, hr2 = st.columns(2)
    hr1.metric("Hit rate: hot (≥1 SD above)", f"{valid['is_hot'].mean()*100:.0f}%" if len(valid) else "N/A")
    hr2.metric("Hit rate: cold (≤1 SD below)", f"{valid['is_cold'].mean()*100:.0f}%" if len(valid) else "N/A")

    if len(valid):
        s1, s2 = st.columns(2)
        s1.metric("Current hot streak", f"{streak(hc['is_hot'])} game(s)")
        s2.metric("Current cold streak", f"{streak(hc['is_cold'])} game(s)")

        zfig = go.Figure()
        zfig.add_scatter(x=hc["game_date"], y=hc["hot_cold_z"], mode="lines+markers",
                          name=f"{hotcold_stat_label} z-score", line=dict(color="#636EFA"))
        add_dnp_markers(zfig, dnp_games)
        zfig.add_hline(y=1, line_dash="dash", line_color="orange")
        zfig.add_hline(y=-1, line_dash="dash", line_color="orange")
        zfig.update_layout(height=340, yaxis_title="Z-score (entering-game mean/SD)")
        st.plotly_chart(zfig, width='stretch', key=f"hotcold_zfig_{panel_key}")
        st.caption(f"Dashed lines = ±1 SD. Black X = DNP (no z-score for a game that didn't happen — "
                   f"streaks simply carry forward across it). First {HOT_COLD_MIN_GAMES} played games "
                   f"of the season show no z-score yet (not enough prior history) — {n_gated_out} "
                   f"game(s) currently gated out.")
    else:
        st.info(f"Not enough played games yet this season to compute hot/cold markers "
                f"(needs {HOT_COLD_MIN_GAMES}+ prior played games).")


def render_consistency_leaderboard(db_path, season):
    """League-wide, not tied to whichever player(s) are selected above --
    rendered once at the bottom of the page (not per-panel), even in
    compare mode, since showing it twice would just be a duplicate."""
    st.subheader("Consistency Leaderboard for Safer Bets", anchor="consistency-leaderboard")
    stat_label = synced_selectbox(
        st, "Stat", list(MAJOR_STATS.keys()),
        "consistency_stat_shared", "consistency_stat_inline")
    stat_col = MAJOR_STATS[stat_label]

    board = load_consistency_leaderboard(db_path, stat_col, season)
    st.caption(f"Top 30 most consistent players by {stat_label.lower()} this season ({season}), "
               f"among players with {CONSISTENCY_MIN_GAMES}+ games played -- sorted by CV%, "
               f"last column. **CV% (Coefficient of Variation)** is the standard deviation "
               f"expressed as a percentage of the mean (SD ÷ mean × 100): it measures spread "
               f"*relative to each player's own average*, so a 25 PPG scorer and a 6 PPG scorer "
               f"can be compared fairly. Lower CV% = more consistent. Click a player's row to "
               f"jump to their full page above.")

    if board.empty:
        st.info(f"Not enough players with {CONSISTENCY_MIN_GAMES}+ played games yet this season "
                f"to build a leaderboard.")
        return

    avg_col = f"Avg {stat_label}"
    display_df = board.copy()
    display_df["range"] = display_df.apply(
        lambda r: f"{r['range_low']:.1f} – {r['range_high']:.1f}", axis=1)
    display_df = display_df.rename(columns={
        "player_name": "Player", "position": "Position", "team": "Team", "games": "Games",
        "avg_minutes": "Avg Minutes", "played_streak": "Games In A Row (No DNP)",
        "avg": avg_col, "range": "1 SD Range", "above_low_streak": "Streak Above Lower Bound",
        "sd": "Std Dev", "cv_pct": "CV %",
    })
    display_df.insert(0, "Rank", range(1, len(display_df) + 1))
    display_df = display_df[[
        "Rank", "Player", "Position", "Team", "Games", "Avg Minutes",
        "Games In A Row (No DNP)", avg_col, "1 SD Range", "Streak Above Lower Bound",
        "Std Dev", "CV %",
    ]]
    event = st.dataframe(
        display_df.style.format({
            "Avg Minutes": "{:.1f}", avg_col: "{:.1f}", "Std Dev": "{:.2f}", "CV %": "{:.1f}%",
            "Streak Above Lower Bound": "{:.0f}",
        }),
        hide_index=True, width='stretch', height=600,
        on_select="rerun", selection_mode="single-row", key="consistency_leaderboard_table",
    )

    selected_rows = event.selection.rows if event and event.selection else []
    if selected_rows:
        clicked_player = display_df.iloc[selected_rows[0]]["Player"]
        if st.session_state.get("player_1") != clicked_player:
            # can't touch team_1/player_1/season_1 here -- those widgets
            # were already instantiated earlier in this same run. Stash
            # the request and apply it at the top of main() on the next
            # rerun, before the sidebar widgets are (re)created.
            st.session_state["_pending_player_jump"] = {"player": clicked_player, "season": season}
            st.rerun()


def sidebar_player_picker(db_path, teams, suffix, label_suffix=""):
    """One Team/Player/Season block in the sidebar. `suffix` disambiguates
    widget keys between the primary and comparison pickers; `label_suffix`
    is just a visual hint (e.g. " (P2)") so the two blocks aren't identical
    text when both are visible."""
    team_choice = st.sidebar.selectbox(f"Team{label_suffix}", ["All Teams"] + teams, key=f"team_{suffix}")
    selected_team = None if team_choice == "All Teams" else team_choice

    players = load_player_list(db_path, team=selected_team)
    if players.empty:
        st.sidebar.error("No players found for that team.")
        return None, None

    player_name = st.sidebar.selectbox(f"Player{label_suffix}", players["player_name"], key=f"player_{suffix}")
    player_id = int(players.loc[players.player_name == player_name, "player_id"].iloc[0])

    df = load_player_games(db_path, player_id)
    seasons = sorted(df["season"].unique(), reverse=True)
    season = st.sidebar.selectbox(f"Season{label_suffix}", seasons, key=f"season_{suffix}")

    return player_id, season


def main():
    st.set_page_config(page_title="WNBA Player Stats Dashboard", layout="wide")
    # center every st.metric's label/value/delta under each other, page-wide
    # (Streamlit left-aligns them by default with no built-in option to change it)
    st.markdown(
        """<style>
        [data-testid="stMetric"] { text-align: center; }
        [data-testid="stMetric"] > div {
            display: flex; flex-direction: column; align-items: center; width: 100%;
        }
        [data-testid="stMetricLabel"], [data-testid="stMetricValue"], [data-testid="stMetricDelta"] {
            display: flex; justify-content: center; width: 100%;
        }
        </style>""",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<h1 style="text-align:center;">WNBA Player-Game Data Dashboard</h1>',
        unsafe_allow_html=True,
    )
    db_path = get_db_path()
    teams = load_teams(db_path)

    # apply a leaderboard row click from the *previous* run now, before the
    # team_1/player_1/season_1 widgets below are (re)created -- setting
    # their session_state after they're instantiated raises an exception,
    # so the click handler just stashes the request and reruns instead
    pending_jump = st.session_state.pop("_pending_player_jump", None)
    if pending_jump:
        st.session_state["team_1"] = "All Teams"
        st.session_state["player_1"] = pending_jump["player"]
        st.session_state["season_1"] = pending_jump["season"]

    last_updated = load_last_updated(db_path)
    if last_updated:
        st.sidebar.caption(f"Updated: {last_updated:%Y-%m-%d %H:%M}")

    st.sidebar.title("Player Selection")
    player_id_1, season_1 = sidebar_player_picker(db_path, teams, "1")
    if player_id_1 is None:
        st.error(f"No players found in {db_path}. Run seed_mock_data.py first.")
        return

    compare = st.sidebar.checkbox("Compare with another player")
    player_id_2 = season_2 = None
    if compare:
        st.sidebar.markdown("**Player 2**")
        player_id_2, season_2 = sidebar_player_picker(db_path, teams, "2", " (P2)")

    st.sidebar.divider()
    st.sidebar.subheader("Chart Settings")
    st.sidebar.caption("Click a name to jump there. Also editable inline, right above each "
                        "chart -- both stay in sync.")
    synced_selectbox(st.sidebar, "Statistical Threshold Hit Rates", list(THRESHOLD_STAT_OPTIONS.keys()),
                      "threshold_stat_shared", "threshold_stat_sidebar", anchor="threshold-hit-rates-p1")
    synced_selectbox(st.sidebar, "Statistical Trends", list(MAJOR_STATS.keys()),
                      "hotcoldtrend_stat_shared", "hotcoldtrend_stat_sidebar", anchor="rolling-hot-cold-trend-p1")
    synced_selectbox(st.sidebar, "Hot / Cold Markers", list(HOTCOLD_STAT_OPTIONS.keys()),
                      "hotcold_stat_shared", "hotcold_stat_sidebar", anchor="hot-cold-markers-p1")

    st.sidebar.divider()
    st.sidebar.markdown("[Game Log](#game-log-p1)")

    st.sidebar.divider()
    synced_selectbox(st.sidebar, "Consistency Leaderboard for Safer Bets", list(MAJOR_STATS.keys()),
                      "consistency_stat_shared", "consistency_stat_sidebar", anchor="consistency-leaderboard")

    if compare and player_id_2 is not None:
        col1, col2 = st.columns(2)
        with col1:
            render_player_panel(db_path, player_id_1, season_1, compact=True, panel_key="p1")
        with col2:
            render_player_panel(db_path, player_id_2, season_2, compact=True, panel_key="p2")
    else:
        render_player_panel(db_path, player_id_1, season_1, compact=False, panel_key="p1")

    if pending_jump:
        # scroll back to the top of the player panel after a leaderboard
        # click -- plain st.markdown doesn't execute <script> tags even
        # with unsafe_allow_html, so this needs st.iframe (raw HTML
        # strings run with JS execution + same-origin access), reaching
        # out to the real page via window.parent
        st.iframe(
            "<script>"
            "window.parent.document.getElementById('player-panel-p1')"
            ".scrollIntoView({behavior: 'instant', block: 'start'});"
            "</script>",
            height=1,
        )

    st.divider()
    render_consistency_leaderboard(db_path, season_1)


if __name__ == "__main__":
    main()
