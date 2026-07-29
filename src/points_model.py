"""
points_model.py

Predicts a player's points distribution for a game, built entirely from
leakage-safe inputs: every quantity used to predict game G is computed
using only games strictly before G's date (same convention as
compute_features.py's *_prior columns).

Pipeline (see PROJECT design discussion for the reasoning):
  1. Per-minute scoring rate, not raw points -- separates "how well she
     scores when on the floor" from "how much she played."
  2. Empirical-Bayes shrinkage of that rate toward a cross-season prior
     (her own prior season if it passes a games-played sanity check,
     else a position-average fallback for rookies / excluded seasons).
     Shrinkage strength (k) is estimated from the data itself, not
     guessed -- see estimate_shrinkage_k().
  3. A small, explicit manual-override table for players where outside
     knowledge (real injury news) says the automatic season-selection
     rule gets it wrong -- see MANUAL_OVERRIDES below.
  4. A position-relative opponent defense index (points allowed to a
     position, normalized to league average), applied as a multiplier.
  5. Expected minutes as a simple shrinkage point-estimate (not a full
     second distribution -- deliberately simplified for v1).

Two competing shapes around that same predicted mean:
  - Negative Binomial (parametric)
  - Empirical / nonparametric residual-ratio pool

...plus a naive flat-average/Normal baseline with none of the above, so
the backtest harness (backtest.py) can check whether the extra
machinery actually earns its complexity.
"""

import math
import sqlite3

import numpy as np
import pandas as pd

# ============================================================
# Manual overrides -- see the injury-watchlist design discussion.
# Keyed by (player_name, season). Small and hand-maintained on purpose:
# only worth the attention for players you'd actually bet on, not the
# full 205-player roster (see points_model design notes).
# ============================================================

# Seasons that should NOT be used as a player's cross-season prior even
# if they clear LOW_GAMES_THRESHOLD (not needed today -- Clark's 2025 is
# already excluded automatically for having too few games -- but kept as
# an explicit escape hatch).
SKIP_SEASON_AS_PRIOR = set()

# Seasons that SHOULD be trusted as a prior even though they fall below
# LOW_GAMES_THRESHOLD -- for players who, when healthy, performed at or
# above their established level despite a short sample (Collier 2026:
# out for a long stretch, but elite in the games she did play).
TRUST_THIN_SEASON = {
    ("Napheesa Collier", 2026),
}

# Predictive-variance inflation for players currently playing through a
# known nagging issue -- widens the distribution rather than lowering
# the mean point estimate (Clark 2026: small ongoing injuries, real
# uncertainty about which version of her shows up night to night).
VARIANCE_INFLATION = {
    ("Caitlin Clark", 2026): 1.5,
}

LOW_GAMES_THRESHOLD = 15          # ~1/3 of a 40-44 game season
OPPONENT_INDEX_CAP = 20           # opponent index clipped to [100-cap, 100+cap] after shrinkage


# ============================================================
# Data loading
# ============================================================

def load_games(db_path):
    con = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        """
        SELECT f.player_id, p.player_name, p.main_position AS position,
               f.season, f.game_id, f.game_date, f.team, f.opponent,
               f.home_away, f.dnp, f.minutes, f.pts
        FROM fact_player_game f
        JOIN dim_player p ON p.player_id = f.player_id
        ORDER BY f.game_date, f.game_id
        """,
        con,
    )
    con.close()
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["rate"] = np.where(
        (df["dnp"] == 0) & (df["minutes"] > 0), df["pts"] / df["minutes"], np.nan
    )
    return df


# ============================================================
# Cross-season prior (per player, per season) with the eligibility
# chain: own prior season (if it passes the games check, or is
# manually trusted) -> position average from the season before ->
# global fallback (documented simplification, only touches the
# earliest season in the dataset, which isn't part of the evaluated
# test period -- see backtest.py).
# ============================================================

def build_cross_season_priors(df):
    played = df[df["dnp"] == 0].dropna(subset=["rate"])

    season_summary = (
        played.groupby(["player_id", "player_name", "season"])
        .agg(games=("game_id", "count"), avg_rate=("rate", "mean"), avg_minutes=("minutes", "mean"))
        .reset_index()
    )

    position_by_season = (
        played.groupby(["season", "position"])
        .agg(avg_rate=("rate", "mean"), avg_minutes=("minutes", "mean"))
        .reset_index()
    )

    all_seasons = sorted(df["season"].unique())
    earliest_season = all_seasons[0]
    global_fallback_rate = played.loc[played["season"] == earliest_season, "rate"].mean()
    global_fallback_minutes = played.loc[played["season"] == earliest_season, "minutes"].mean()

    ss_index = season_summary.set_index(["player_id", "season"])
    pos_index = position_by_season.set_index(["season", "position"])

    player_positions = played.groupby("player_id")["position"].agg(lambda s: s.mode().iat[0])
    player_names = played.groupby("player_id")["player_name"].first()

    results = []
    for player_id in season_summary["player_id"].unique():
        name = player_names.get(player_id)
        position = player_positions.get(player_id)
        player_seasons = sorted(season_summary.loc[season_summary.player_id == player_id, "season"].unique())
        for season in all_seasons:
            candidate_rate, candidate_minutes, source = None, None, None
            for prior_season in sorted([s for s in player_seasons if s < season], reverse=True):
                key = (player_id, prior_season)
                if key not in ss_index.index:
                    continue
                games = ss_index.loc[key, "games"]
                trusted = (name, prior_season) in TRUST_THIN_SEASON
                skipped = (name, prior_season) in SKIP_SEASON_AS_PRIOR
                if skipped:
                    continue
                if trusted or games >= LOW_GAMES_THRESHOLD:
                    candidate_rate = ss_index.loc[key, "avg_rate"]
                    candidate_minutes = ss_index.loc[key, "avg_minutes"]
                    source = f"own_season_{prior_season}"
                    break
            if candidate_rate is None:
                prev_season_idx = all_seasons.index(season) - 1
                if prev_season_idx >= 0 and position is not None:
                    prev_season = all_seasons[prev_season_idx]
                    pkey = (prev_season, position)
                    if pkey in pos_index.index:
                        candidate_rate = pos_index.loc[pkey, "avg_rate"]
                        candidate_minutes = pos_index.loc[pkey, "avg_minutes"]
                        source = f"position_avg_{prev_season}"
            if candidate_rate is None:
                candidate_rate = global_fallback_rate
                candidate_minutes = global_fallback_minutes
                source = "global_fallback"
            results.append(dict(
                player_id=player_id, season=season,
                cross_season_prior_rate=candidate_rate,
                cross_season_prior_minutes=candidate_minutes,
                cross_season_prior_source=source,
            ))
    return pd.DataFrame(results)


# ============================================================
# Within-season expanding ("entering this game") priors
# ============================================================

def add_within_season_priors(df):
    df = df.sort_values(["player_id", "season", "game_date", "game_id"]).copy()
    played_mask = (df["dnp"] == 0) & df["rate"].notna()

    def expanding_prior(group_col, value_col):
        vals = df[value_col].where(played_mask)
        shifted = vals.groupby([df["player_id"], df["season"]]).shift(1)
        return shifted.groupby([df["player_id"], df["season"]]).expanding().mean().reset_index(level=[0, 1], drop=True)

    df["within_season_prior_rate"] = expanding_prior(["player_id", "season"], "rate")
    df["within_season_prior_minutes"] = expanding_prior(["player_id", "season"], "minutes")
    df["within_season_prior_pts"] = expanding_prior(["player_id", "season"], "pts")
    df["n_prior_games"] = (
        played_mask.groupby([df["player_id"], df["season"]]).cumsum()
        - played_mask.astype(int)
    ).clip(lower=0)
    return df


# ============================================================
# Empirically-estimated shrinkage strength k = within-player variance /
# between-player variance, using ONLY the earliest complete season
# (2024) so the estimate doesn't peek at partial in-progress seasons.
# ============================================================

def estimate_shrinkage_k(df, value_col, min_games=10):
    played = df[(df["dnp"] == 0) & df[value_col].notna()]
    earliest_season = df["season"].min()
    season_df = played[played["season"] == earliest_season]

    counts = season_df.groupby("player_id")[value_col].count()
    qualifying = counts[counts >= min_games].index
    sub = season_df[season_df["player_id"].isin(qualifying)]
    if sub["player_id"].nunique() < 5:
        return 8.0  # not enough qualifying players -- fall back to a reasonable default

    player_means = sub.groupby("player_id")[value_col].mean()
    within_var = sub.groupby("player_id")[value_col].var(ddof=1).mean()
    raw_between_var = player_means.var(ddof=1)
    avg_n = sub.groupby("player_id")[value_col].count().mean()

    tau2 = max(raw_between_var - within_var / avg_n, 1e-6)
    k = within_var / tau2
    return float(np.clip(k, 1.0, 40.0))


def estimate_opponent_shrinkage_k(df, min_games=10):
    """Same empirical-Bayes method as estimate_shrinkage_k() above, just
    applied to team-position points-allowed instead of player rate/
    minutes: k = within-entity variance / between-entity variance,
    fit on the earliest complete season only. "Entity" here is a
    (team, position) pair rather than a player -- pooled across all
    three positions together (not fit separately per position) to keep
    the entity count large enough for a stable estimate, since there
    are far fewer teams than players."""
    played = df[(df["dnp"] == 0) & df["pts"].notna()]
    earliest_season = df["season"].min()
    season_df = played[played["season"] == earliest_season]

    allowed = season_df.groupby(["opponent", "position", "game_id"])["pts"].sum().reset_index()
    allowed["entity"] = allowed["opponent"] + "_" + allowed["position"]

    counts = allowed.groupby("entity")["pts"].count()
    qualifying = counts[counts >= min_games].index
    sub = allowed[allowed["entity"].isin(qualifying)]
    if sub["entity"].nunique() < 5:
        return 15.0  # not enough qualifying team-position groups -- reasonable default

    entity_means = sub.groupby("entity")["pts"].mean()
    within_var = sub.groupby("entity")["pts"].var(ddof=1).mean()
    raw_between_var = entity_means.var(ddof=1)
    avg_n = sub.groupby("entity")["pts"].count().mean()

    tau2 = max(raw_between_var - within_var / avg_n, 1e-6)
    k = within_var / tau2
    return float(np.clip(k, 3.0, 60.0))


def apply_shrinkage(df, within_col, cross_col, k, out_col):
    n = df["n_prior_games"].astype(float)
    w = n / (n + k)
    within = df[within_col].fillna(0.0)
    df[out_col] = w * within + (1 - w) * df[cross_col]
    return df


# ============================================================
# Opponent position-defense index, leakage-safe (entering-that-date),
# normalized to 100 = league average, lightly shrunk toward 100 early.
# ============================================================

def build_opponent_index(df, k=None):
    if k is None:
        k = estimate_opponent_shrinkage_k(df)

    played = df[df["dnp"] == 0].dropna(subset=["pts"])

    allowed = (
        played.groupby(["season", "opponent", "game_date", "game_id", "position"])["pts"]
        .sum()
        .reset_index()
        .rename(columns={"opponent": "defending_team"})
    )
    allowed = allowed.sort_values(["defending_team", "position", "game_date", "game_id"])

    allowed["team_games_prior"] = allowed.groupby(["defending_team", "position"]).cumcount()
    allowed["team_avg_allowed_prior"] = (
        allowed.groupby(["defending_team", "position"])["pts"]
        .apply(lambda s: s.shift(1).expanding().mean())
        .reset_index(level=[0, 1], drop=True)
    )
    allowed["league_avg_allowed_prior"] = (
        allowed.groupby(["position"])["pts"]
        .apply(lambda s: s.shift(1).expanding().mean())
        .reset_index(level=0, drop=True)
    )

    raw_index = 100 * allowed["team_avg_allowed_prior"] / allowed["league_avg_allowed_prior"]
    n = allowed["team_games_prior"].astype(float)
    w = n / (n + k)
    allowed["opponent_index"] = w * raw_index.fillna(100.0) + (1 - w) * 100.0
    allowed["opponent_index"] = allowed["opponent_index"].fillna(100.0)
    allowed["opponent_index"] = allowed["opponent_index"].clip(
        100 - OPPONENT_INDEX_CAP, 100 + OPPONENT_INDEX_CAP
    )

    return allowed[["season", "defending_team", "game_id", "game_date", "position", "opponent_index"]].rename(
        columns={"defending_team": "opponent"}
    )


# ============================================================
# Shared prediction pipeline. build_predicted_means() computes each
# row's predicted_mean_pts using only strictly-prior data (leakage-safe
# by construction), so it's identical whether the caller wants an
# honest backtest (fit the two remaining nuisance params -- NB
# dispersion and the empirical pool -- on a train-only slice
# afterward, see backtest.py) or a live prediction (fit them on
# everything, see fit_production_params() below).
# ============================================================

def build_predicted_means(df):
    df = add_within_season_priors(df)
    cross_priors = build_cross_season_priors(df)
    df = df.merge(cross_priors, on=["player_id", "season"], how="left")

    k_opponent = estimate_opponent_shrinkage_k(df)
    opponent_idx = build_opponent_index(df, k=k_opponent)
    df = df.merge(
        opponent_idx.drop(columns="game_date"),
        on=["season", "opponent", "game_id", "position"], how="left",
    )
    df["opponent_index"] = df["opponent_index"].fillna(100.0)

    k_rate = estimate_shrinkage_k(df, "rate")
    k_minutes = estimate_shrinkage_k(df, "minutes")
    df = apply_shrinkage(df, "within_season_prior_rate", "cross_season_prior_rate", k_rate, "shrunk_rate")
    df = apply_shrinkage(df, "within_season_prior_minutes", "cross_season_prior_minutes", k_minutes, "shrunk_minutes")
    df["predicted_mean_pts"] = df["shrunk_rate"] * df["shrunk_minutes"] * (df["opponent_index"] / 100.0)

    return df, cross_priors, opponent_idx, k_rate, k_minutes, k_opponent


def fit_production_params(df):
    """Fit NB dispersion + the empirical residual pool on the FULL
    historical dataset -- appropriate for live predictions, where we
    want the best current estimate and there's no future to leak from.
    For an honest, held-out measure of model quality, see backtest.py,
    which fits these same two parameters on a chronological train-only
    slice instead."""
    df, cross_priors, opponent_idx, k_rate, k_minutes, k_opponent = build_predicted_means(df)
    played = df[(df["dnp"] == 0) & df["pts"].notna() & df["predicted_mean_pts"].notna()]
    r_nb = fit_nb_dispersion(played, "predicted_mean_pts")
    ratio_pool = build_empirical_pool(played, "predicted_mean_pts")
    return dict(
        df=df, cross_priors=cross_priors, opponent_idx=opponent_idx,
        k_rate=k_rate, k_minutes=k_minutes, k_opponent=k_opponent,
        r_nb=r_nb, ratio_pool=ratio_pool,
    )


def predict_upcoming_game(params, player_id, opponent_team):
    """Live prediction for a player's next game against `opponent_team`,
    reusing already-fit production params (see fit_production_params) --
    cheap enough to call once per dashboard render. Returns None if the
    player has no games on record at all."""
    df = params["df"]
    player_rows = df[df["player_id"] == player_id]
    if player_rows.empty:
        return None

    current_season = player_rows["season"].max()
    played_this_season = player_rows[
        (player_rows["season"] == current_season) & (player_rows["dnp"] == 0) & player_rows["rate"].notna()
    ]
    n_prior = len(played_this_season)
    within_rate = played_this_season["rate"].mean() if n_prior else 0.0
    within_minutes = played_this_season["minutes"].mean() if n_prior else 0.0

    cp_row = params["cross_priors"][
        (params["cross_priors"]["player_id"] == player_id) & (params["cross_priors"]["season"] == current_season)
    ]
    if cp_row.empty:
        return None
    cross_rate = cp_row["cross_season_prior_rate"].iloc[0]
    cross_minutes = cp_row["cross_season_prior_minutes"].iloc[0]

    k_rate, k_minutes = params["k_rate"], params["k_minutes"]
    w_rate = n_prior / (n_prior + k_rate)
    w_min = n_prior / (n_prior + k_minutes)
    shrunk_rate = w_rate * within_rate + (1 - w_rate) * cross_rate
    shrunk_minutes = w_min * within_minutes + (1 - w_min) * cross_minutes

    position = player_rows["position"].iloc[-1]
    oi = params["opponent_idx"]
    matches = oi[(oi["season"] == current_season) & (oi["opponent"] == opponent_team) & (oi["position"] == position)]
    opponent_index = matches.sort_values("game_date").iloc[-1]["opponent_index"] if not matches.empty else 100.0

    predicted_mean = shrunk_rate * shrunk_minutes * (opponent_index / 100.0)

    player_name = player_rows["player_name"].iloc[0]
    infl = VARIANCE_INFLATION.get((player_name, int(current_season)), 1.0)

    return dict(
        predicted_mean_pts=predicted_mean,
        opponent_index=opponent_index,
        n_prior_games=n_prior,
        variance_inflation=infl,
        r_nb=params["r_nb"] / infl,
        ratio_pool=params["ratio_pool"],
    )


# ============================================================
# Distribution shapes: P(pts > line) under each model.
# ============================================================

def _nb_log_pmf(k, r, mean):
    p = r / (r + mean)
    return (
        math.lgamma(k + r) - math.lgamma(r) - math.lgamma(k + 1)
        + r * math.log(p) + k * math.log(1 - p)
    )


def prob_over_nb(mean, r, line):
    mean = max(mean, 0.1)
    floor_line = math.floor(line)
    if floor_line < 0:
        return 1.0
    cdf = sum(math.exp(_nb_log_pmf(k, r, mean)) for k in range(0, floor_line + 1))
    return float(np.clip(1 - cdf, 0.0, 1.0))


def _normal_cdf(x, mean, sd):
    sd = max(sd, 0.5)
    return 0.5 * (1 + math.erf((x - mean) / (sd * math.sqrt(2))))


def prob_over_normal(mean, sd, line):
    return float(np.clip(1 - _normal_cdf(line, mean, sd), 0.0, 1.0))


def prob_over_empirical(mean, ratio_pool, line, inflation=1.0):
    mean = max(mean, 0.1)
    scaled = mean * (1 + inflation * (ratio_pool - 1))
    return float(np.mean(scaled > line))


def prob_over_blend(mean, r, ratio_pool, line, inflation=1.0):
    """Straight average of the NB and empirical probabilities -- see
    backtest.py. NB alone assumes more skew than the data shows;
    empirical alone (pooled across all players) washes the skew out
    almost entirely. Averaging the two beat both individually on log
    loss, Brier score, and calibration in the walk-forward backtest.
    (A train-fit blend weight was tried instead of a flat 50/50 average
    and did worse out of sample -- overfit train-fold noise, same as
    the dispersion-bucketing and isotonic-recalibration attempts.)"""
    p_nb = prob_over_nb(mean, r, line)
    p_emp = prob_over_empirical(mean, ratio_pool, line, inflation=inflation)
    return (p_nb + p_emp) / 2


def fit_nb_dispersion(train_df, mean_col, actual_col="pts"):
    resid = train_df[actual_col] - train_df[mean_col]
    mean_pred = train_df[mean_col].mean()
    excess_var = max((resid ** 2).mean() - mean_pred, 1e-3)
    r = (mean_pred ** 2) / excess_var
    return float(np.clip(r, 1.0, 200.0))


def build_empirical_pool(train_df, mean_col, actual_col="pts"):
    valid = train_df[train_df[mean_col] > 1]
    ratios = (valid[actual_col] / valid[mean_col]).to_numpy()
    return ratios[np.isfinite(ratios)]
