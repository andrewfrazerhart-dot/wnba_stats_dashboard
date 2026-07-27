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
def load_player_list(db_path: str) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
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


def hit_rate_table(played: pd.DataFrame, thresholds=(10, 15, 20)) -> pd.DataFrame:
    """Overall + home/away split hit-rate for each scoring threshold."""
    rows = []
    for t in thresholds:
        col = f"pts_{t}plus"
        overall = played[col].mean() * 100 if len(played) else None
        home = played.loc[played.home_away == "H", col].mean() * 100 if (played.home_away == "H").any() else None
        away = played.loc[played.home_away == "A", col].mean() * 100 if (played.home_away == "A").any() else None
        rows.append({"Threshold": f"{t}+ pts", "Overall": overall, "Home": home, "Away": away})
    return pd.DataFrame(rows)


def main():
    st.set_page_config(page_title="WNBA Player Stats Dashboard", layout="wide")
    db_path = get_db_path()

    players = load_player_list(db_path)
    if players.empty:
        st.error(f"No players found in {db_path}. Run seed_mock_data.py first.")
        return

    st.sidebar.title("Player Selection")
    player_name = st.sidebar.selectbox("Player", players["player_name"])
    player_id = int(players.loc[players.player_name == player_name, "player_id"].iloc[0])

    df = load_player_games(db_path, player_id)
    seasons = sorted(df["season"].unique(), reverse=True)
    season = st.sidebar.selectbox("Season", seasons)

    season_df = df[df.season == season].reset_index(drop=True)
    played = season_df[season_df.dnp == 0].copy()

    if played.empty:
        st.warning("No played games for this player/season yet.")
        return

    bio = season_df.iloc[0]

    # ---------------------------------------------------------------
    # Header
    # ---------------------------------------------------------------
    st.title(bio.player_name)
    st.caption(f"{played.iloc[-1].team} · {bio.position_detail} ({bio.main_position}) · {season} season")

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Season Avg PTS", f"{played.pts.mean():.1f}")
    m2.metric("Season Avg REB", f"{played.reb_tot.mean():.1f}")
    m3.metric("Season Avg AST", f"{played.ast.mean():.1f}")
    m4.metric("Games Played", f"{len(played)}")
    m5.metric("Games Missed (DNP)", f"{(season_df.dnp == 1).sum()}")
    last5_pts = played.tail(5)["pts"].mean()
    delta = last5_pts - played.pts.mean()
    m6.metric("L5 Avg PTS", f"{last5_pts:.1f}", delta=f"{delta:+.1f} vs season")

    st.divider()

    # ---------------------------------------------------------------
    # Threshold hit-rates
    # ---------------------------------------------------------------
    st.subheader("Scoring Threshold Hit-Rates")
    st.caption("% of played games this season clearing each points threshold, split by home/away")

    hr = hit_rate_table(played)
    fig = go.Figure()
    fig.add_bar(name="Home", x=hr["Threshold"], y=hr["Home"])
    fig.add_bar(name="Away", x=hr["Threshold"], y=hr["Away"])
    fig.add_bar(name="Overall", x=hr["Threshold"], y=hr["Overall"], marker_color="lightgray", opacity=0.6)
    fig.update_layout(barmode="group", yaxis_title="Hit rate (%)", yaxis_range=[0, 100], height=380)
    st.plotly_chart(fig, width='stretch')

    st.dataframe(
        hr.style.format({"Overall": "{:.0f}%", "Home": "{:.0f}%", "Away": "{:.0f}%"}),
        hide_index=True, width='stretch',
    )

    st.divider()

    # ---------------------------------------------------------------
    # Rolling hot / cold trend
    # ---------------------------------------------------------------
    st.subheader("Rolling Hot / Cold Trend")
    st.caption("Points per game vs. rolling 5-game and trailing 14-day averages (entering each game, leakage-safe)")

    trend = go.Figure()
    trend.add_bar(x=played["game_date"], y=played["pts"], name="Points (that game)",
                  marker_color=["#2ca02c" if h == "H" else "#1f77b4" for h in played["home_away"]])
    trend.add_scatter(x=played["game_date"], y=played["avg_pts_l5_prior"], mode="lines",
                       name="Last 5 games avg (entering)", line=dict(color="orange", width=2))
    trend.add_scatter(x=played["game_date"], y=played["avg_pts_l14d_prior"], mode="lines",
                       name="Last 14 days avg (entering)", line=dict(color="red", width=2, dash="dot"))
    trend.add_hline(y=played.pts.mean(), line_dash="dash", line_color="gray",
                     annotation_text="Season avg (final)")
    trend.update_layout(height=420, yaxis_title="Points", legend=dict(orientation="h", y=1.1))
    st.plotly_chart(trend, width='stretch')
    st.caption("Green bar = home game, blue bar = away game. Rolling lines reflect games "
               "*entering* each date (leakage-safe).")

    st.divider()

    # ---------------------------------------------------------------
    # Hot / Cold markers (z-score vs. entering-game mean/SD)
    # ---------------------------------------------------------------
    st.subheader("Hot / Cold Markers")
    st.caption("Games flagged ≥1 SD above/below the player's own to-date mean, entering that game "
               "(leakage-safe) — plus the current streak of consecutive hot/cold games.")

    stat_options = {"Points": "pts", "Rebounds": "reb_tot", "Assists": "ast",
                     "Steals": "stl", "Blocks": "blk", "Game Score": "game_score"}
    stat_label = st.selectbox("Stat", list(stat_options.keys()))
    sk = stat_options[stat_label]

    hc = add_hot_cold(played, sk)
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
                          name=f"{stat_label} z-score", line=dict(color="#636EFA"))
        zfig.add_hline(y=1, line_dash="dash", line_color="orange")
        zfig.add_hline(y=-1, line_dash="dash", line_color="orange")
        zfig.update_layout(height=340, yaxis_title="Z-score (entering-game mean/SD)")
        st.plotly_chart(zfig, width='stretch')
        st.caption(f"Dashed lines = ±1 SD. First {HOT_COLD_MIN_GAMES} played games of the season show no "
                   f"z-score yet (not enough prior history) — {n_gated_out} game(s) currently gated out.")
    else:
        st.info(f"Not enough played games yet this season to compute hot/cold markers "
                f"(needs {HOT_COLD_MIN_GAMES}+ prior played games).")

    st.divider()

    # ---------------------------------------------------------------
    # Other rolling stats
    # ---------------------------------------------------------------
    st.subheader("Rebounds & Assists Trend")
    fig2 = go.Figure()
    fig2.add_scatter(x=played["game_date"], y=played["reb_tot"], name="REB", mode="lines+markers")
    fig2.add_scatter(x=played["game_date"], y=played["avg_reb_l5_prior"], name="REB L5 avg (entering)",
                      mode="lines", line=dict(dash="dot"))
    fig2.add_scatter(x=played["game_date"], y=played["ast"], name="AST", mode="lines+markers")
    fig2.add_scatter(x=played["game_date"], y=played["avg_ast_l5_prior"], name="AST L5 avg (entering)",
                      mode="lines", line=dict(dash="dot"))
    fig2.update_layout(height=360)
    st.plotly_chart(fig2, width='stretch')

    st.divider()

    # ---------------------------------------------------------------
    # Full game log
    # ---------------------------------------------------------------
    st.subheader("Game Log")
    log_df = hc[[
        "game_date", "opponent", "home_away", "result", "started",
        "minutes", "pts", "reb_tot", "ast", "stl", "blk", "tov", "game_score",
        "avg_pts_l5_prior", "avg_pts_l14d_prior", "hot_cold_z", "is_hot", "is_cold",
    ]].sort_values("game_date", ascending=False).copy()
    log_df["game_date"] = log_df["game_date"].dt.strftime("%Y-%m-%d")
    log_df["hot_cold_z"] = log_df["hot_cold_z"].round(2)
    log_df = log_df.rename(columns={
        "hot_cold_z": f"{stat_label} z", "is_hot": "Hot", "is_cold": "Cold",
        "avg_pts_l5_prior": "PTS L5 avg", "avg_pts_l14d_prior": "PTS L14d avg",
    })
    st.dataframe(log_df, hide_index=True, width='stretch', height=400)


if __name__ == "__main__":
    main()
