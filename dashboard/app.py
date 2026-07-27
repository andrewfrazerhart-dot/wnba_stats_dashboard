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

import sqlite3
import sys

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DEFAULT_DB = "../data/wnba.db"
HOT_COLD_MIN_GAMES = 5  # min prior played games before a hot/cold read is shown


def get_db_path() -> str:
    if "--db" in sys.argv:
        return sys.argv[sys.argv.index("--db") + 1]
    return DEFAULT_DB


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


def synced_selectbox(container, label, options, shared_key, widget_key):
    """A selectbox that can be rendered in multiple places at once (e.g.
    the sidebar AND inline in one or two panels) while staying in sync --
    changing any instance updates the others on the next rerun.

    Streamlit normally lets a widget's own session_state (tied to its
    `key`) "win" over anything else once that key has been used, so
    simply passing `index=` from a shared value only works on the very
    first render. Instead, this pushes the current shared value into
    this instance's own key right before creating it, every rerun --
    that's what makes every instance track the shared value, not just
    whichever one the user directly touched."""
    if shared_key not in st.session_state:
        st.session_state[shared_key] = options[0]
    st.session_state[widget_key] = st.session_state[shared_key]

    def _sync():
        st.session_state[shared_key] = st.session_state[widget_key]

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
        rows.append({
            "Threshold": f"{t}+ {label}",
            "Overall": hit.mean() * 100 if len(played) else None,
            "Home": hit[home_mask].mean() * 100 if home_mask.any() else None,
            "Away": hit[away_mask].mean() * 100 if away_mask.any() else None,
            "Last 5": (last5[stat_col] >= t).mean() * 100 if len(last5) else None,
            "Last 14 Days": (last14d[stat_col] >= t).mean() * 100 if len(last14d) else None,
        })
    return pd.DataFrame(rows)


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

    # ---------------------------------------------------------------
    # Header
    # ---------------------------------------------------------------
    st.title(bio.player_name)
    st.caption(f"{played.iloc[-1].team} · {bio.position_detail} ({bio.main_position}) · {season} season")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Season Avg PTS", f"{played.pts.mean():.1f}")
    m2.metric("Season Avg REB", f"{played.reb_tot.mean():.1f}")
    m3.metric("Season Avg AST", f"{played.ast.mean():.1f}")
    m4.metric("Games Played", f"{len(played)}")

    m5, m6, m7, m8 = st.columns(4)
    m5.metric("Games Missed (DNP)", f"{(season_df.dnp == 1).sum()}")
    m6.metric("Current Played Streak", f"{played_streak(season_df)} game(s)")

    last5_pts = played.tail(5)["pts"].mean()
    delta5 = last5_pts - played.pts.mean()
    m7.metric("L5 Avg PTS", f"{last5_pts:.1f}", delta=f"{delta5:+.1f} vs season")

    most_recent_date = played["game_date"].max()
    last14d_pts = played.loc[played["game_date"] >= most_recent_date - pd.Timedelta(days=14), "pts"].mean()
    delta14 = last14d_pts - played.pts.mean()
    m8.metric("L14D Avg PTS", f"{last14d_pts:.1f}", delta=f"{delta14:+.1f} vs season")

    st.divider()

    # ---------------------------------------------------------------
    # Threshold hit-rates
    # ---------------------------------------------------------------
    st.subheader("Threshold Hit-Rates")
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
        hr.style.format({c: "{:.0f}%" for c in ["Overall", "Home", "Away", "Last 5", "Last 14 Days"]}),
        hide_index=True, width='stretch', key=f"threshold_table_{panel_key}",
    )

    st.divider()

    # ---------------------------------------------------------------
    # Stat trend (single stat, selectable)
    # ---------------------------------------------------------------
    st.subheader("Stat Trend")
    trend_stat_label = synced_selectbox(
        st, "Stat", list(MAJOR_STATS.keys()),
        "trend_stat_shared", f"trend_stat_inline_{panel_key}")
    trend_stat_col = MAJOR_STATS[trend_stat_label]

    rolling_df = rolling_prior_averages(season_df, trend_stat_col)
    played_rolling = rolling_df[rolling_df.dnp == 0]

    fig_trend = go.Figure()
    fig_trend.add_scatter(x=played_rolling["game_date"], y=played_rolling[trend_stat_col],
                           name=trend_stat_label, mode="lines+markers")
    fig_trend.add_scatter(x=rolling_df["game_date"], y=rolling_df["_l5_prior"],
                           name="L5 avg (entering)", mode="lines", line=dict(dash="dot", color="orange"))
    fig_trend.add_scatter(x=rolling_df["game_date"], y=rolling_df["_l14d_prior"],
                           name="L14D avg (entering)", mode="lines", line=dict(dash="dot", color="red"))
    add_dnp_markers(fig_trend, dnp_games)
    fig_trend.update_layout(height=380, yaxis_title=trend_stat_label, legend=dict(orientation="h", y=1.15))
    st.plotly_chart(fig_trend, width='stretch', key=f"trend_chart_{panel_key}")
    st.caption("Black X = DNP. Rolling lines are leakage-safe (entering each game) and continue "
               "across DNP gaps, since a missed game doesn't change the average.")

    st.divider()

    # ---------------------------------------------------------------
    # Rolling hot / cold trend
    # ---------------------------------------------------------------
    st.subheader("Rolling Hot / Cold Trend")
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
    st.subheader("Hot / Cold Markers")
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

    st.divider()

    # ---------------------------------------------------------------
    # Full game log
    # ---------------------------------------------------------------
    log_source = season_df.merge(
        hc[["game_id", "hot_cold_z", "is_hot", "is_cold"]], on="game_id", how="left",
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
        "dnp": "DNP", "hot_cold_z": f"{hotcold_stat_label} z", "is_hot": "Hot", "is_cold": "Cold",
        "season_avg_pts_prior": "PTS season avg (to date)",
        "avg_pts_l5_prior": "PTS L5 avg", "avg_pts_l14d_prior": "PTS L14d avg",
    })

    if compact:
        with st.expander("Game Log", key=f"gamelog_expander_{panel_key}"):
            st.dataframe(log_df, hide_index=True, width='stretch', height=400, key=f"gamelog_{panel_key}")
    else:
        st.subheader("Game Log")
        st.caption("Includes DNP games (all box-score stats blank) so you can see missed games "
                   "relative to games actually played.")
        st.dataframe(log_df, hide_index=True, width='stretch', height=400, key=f"gamelog_{panel_key}")


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
    db_path = get_db_path()
    teams = load_teams(db_path)

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
    st.sidebar.caption("Also editable inline, right above each chart -- both stay in sync.")
    synced_selectbox(st.sidebar, "Threshold Hit-Rates stat", list(THRESHOLD_STAT_OPTIONS.keys()),
                      "threshold_stat_shared", "threshold_stat_sidebar")
    synced_selectbox(st.sidebar, "Stat Trend stat", list(MAJOR_STATS.keys()),
                      "trend_stat_shared", "trend_stat_sidebar")
    synced_selectbox(st.sidebar, "Rolling Hot/Cold Trend stat", list(MAJOR_STATS.keys()),
                      "hotcoldtrend_stat_shared", "hotcoldtrend_stat_sidebar")
    synced_selectbox(st.sidebar, "Hot/Cold Markers stat", list(HOTCOLD_STAT_OPTIONS.keys()),
                      "hotcold_stat_shared", "hotcold_stat_sidebar")

    if compare and player_id_2 is not None:
        col1, col2 = st.columns(2)
        with col1:
            render_player_panel(db_path, player_id_1, season_1, compact=True, panel_key="p1")
        with col2:
            render_player_panel(db_path, player_id_2, season_2, compact=True, panel_key="p2")
    else:
        render_player_panel(db_path, player_id_1, season_1, compact=False, panel_key="p1")


if __name__ == "__main__":
    main()
