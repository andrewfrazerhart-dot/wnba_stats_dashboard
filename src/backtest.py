"""
backtest.py

Walk-forward evaluation of four competing points-prediction models:
  - naive     : flat within-season average, single pooled SD, no
                shrinkage / no opponent adjustment / no minutes split.
                The "did any of the machinery actually help" baseline.
  - nb        : shrunk mean (rate x minutes x opponent index) wrapped
                in a Negative Binomial shape.
  - empirical : the same shrunk mean, but with an empirical/nonparametric
                residual-ratio pool instead of an assumed NB shape.
  - blend     : straight average of nb and empirical's P(over). Added
                after noticing the two disagree in OPPOSITE directions
                from the true rate (NB assumes too much skew, empirical
                pools across all players and washes out skew almost
                entirely) -- averaging cancels most of that bias and
                beat both individual models on log loss/Brier/calibration.

An extensive, deliberately exhaustive round of follow-up attempts to
beat this blend did NOT succeed and are not part of this file:
  - CV-bucketed NB dispersion, residual-based dispersion bucketing
  - Post-hoc isotonic recalibration (made held-out log loss worse --
    overfit train-fold noise)
  - A train-fit blend weight vs. the flat 50/50 average (same failure)
  - Skew-normal (independent skewness parameter) -- beat plain NB but
    not the blend; realized sample skewness (~1.51) exceeds what
    skew-normal can represent at all (max ~0.995)
  - Skew-t (adds independent tail-weight) -- matched the true skewness
    and kurtosis closely via method-of-moments, but still didn't beat
    the blend; more accurate moments didn't translate to better
    calibration, likely because kurtosis (a 4th moment) is itself too
    noisy a statistic to trust at this data size
  - Quantile regression (points ~ predicted_mean_pts) -- alone, similar
    to empirical; blended with NB, nominally beat the blend by 0.0005 in
    log loss, judged noise-level given a single train/test split and
    not adopted

Across nine variations, the plain 50/50 NB+empirical blend won or tied
within noise every time -- treated as a real, well-earned ceiling for
probability-level tuning at this data size, not a failure to try hard
enough. See PROJECT_STATUS.md / session notes for the full reasoning;
not re-adding any of that code since none of it earned its keep.

Split: chronological, not random (train = 2024 + first 70% of 2025 by
date; test = remainder of 2025 + all of 2026). Every per-row prediction
is already leakage-safe by construction (built entirely from *_prior /
entering-game quantities), so this single split is really just where we
fit the few GLOBAL nuisance parameters (shrinkage k's, NB dispersion,
empirical pool, opponent-index baselines) without letting them peek at
the evaluation period -- not a re-derivation of the leakage-safety the
rest of the project already guarantees.

Evaluated at a realistic line per player-game: her own predicted mean,
rounded to the nearest half point (mirrors a real sportsbook line, and
matches the default line already used in the dashboard's Odds
Calculator) -- not one fixed threshold for every player regardless of
role.

Usage:
    python backtest.py --db ../data/wnba.db
"""

import argparse

import numpy as np
import pandas as pd

from points_model import (
    VARIANCE_INFLATION,
    build_empirical_pool,
    build_predicted_means,
    fit_nb_dispersion,
    load_games,
    prob_over_empirical,
    prob_over_nb,
    prob_over_normal,
)

EPS = 1e-6


def build_feature_table(db_path):
    df = load_games(db_path)
    df, cross_priors, opponent_idx, k_rate, k_minutes, k_opponent = build_predicted_means(df)
    df["variance_inflation"] = df.apply(
        lambda r: VARIANCE_INFLATION.get((r["player_name"], r["season"]), 1.0), axis=1
    )
    return df, k_rate, k_minutes, k_opponent


def chronological_split(df):
    season_2025_dates = df.loc[df["season"] == 2025, "game_date"]
    cutoff = season_2025_dates.quantile(0.70)
    train = df[df["game_date"] < cutoff]
    test = df[df["game_date"] >= cutoff]
    return train, test, cutoff


def evaluate(df, db_path):
    played = df[(df["dnp"] == 0) & df["pts"].notna() & df["predicted_mean_pts"].notna()].copy()
    train, test, cutoff = chronological_split(played)

    r_nb = fit_nb_dispersion(train, "predicted_mean_pts")
    ratio_pool = build_empirical_pool(train, "predicted_mean_pts")

    train_global_mean_pts = train["pts"].mean()
    naive_mean = train["within_season_prior_pts"].fillna(train_global_mean_pts)
    naive_resid_train = train["pts"] - naive_mean
    naive_sd = naive_resid_train.std(ddof=1)

    test = test.copy()
    test["naive_mean"] = test["within_season_prior_pts"].fillna(train_global_mean_pts)
    test["line"] = (test["predicted_mean_pts"] * 2).round() / 2
    test["line"] = test["line"].clip(lower=0.5)
    test["actual_over"] = (test["pts"] > test["line"]).astype(int)

    def row_probs(row):
        infl = row["variance_inflation"]
        p_naive = prob_over_normal(row["naive_mean"], naive_sd, row["line"])
        p_nb = prob_over_nb(row["predicted_mean_pts"], r_nb / infl, row["line"])
        p_emp = prob_over_empirical(row["predicted_mean_pts"], ratio_pool, row["line"], inflation=infl)
        return pd.Series({"p_naive": p_naive, "p_nb": p_nb, "p_emp": p_emp})

    probs = test.apply(row_probs, axis=1)
    test = pd.concat([test, probs], axis=1)
    test["p_blend"] = (test["p_nb"] + test["p_emp"]) / 2

    print(f"\nTrain rows: {len(train)}   Test rows: {len(test)}   Split date: {cutoff.date()}")
    print(f"Global NB dispersion r: {r_nb:.2f}")
    print(f"Empirical residual pool size: {len(ratio_pool)}")

    print("\n--- Log loss / Brier score (lower is better) ---")
    for name, col in [("naive", "p_naive"), ("negative binomial", "p_nb"), ("empirical", "p_emp"), ("blend", "p_blend")]:
        p = test[col].clip(EPS, 1 - EPS)
        y = test["actual_over"]
        logloss = -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()
        brier = ((p - y) ** 2).mean()
        print(f"  {name:20s}  log loss={logloss:.4f}   brier={brier:.4f}")

    print("\n--- Calibration (predicted vs. actual 'over' rate, by decile) ---")
    for name, col in [("naive", "p_naive"), ("negative binomial", "p_nb"), ("empirical", "p_emp"), ("blend", "p_blend")]:
        print(f"\n  {name}:")
        bins = pd.qcut(test[col], 10, duplicates="drop")
        cal = test.groupby(bins, observed=True).agg(
            n=("actual_over", "size"),
            predicted=(col, "mean"),
            actual=("actual_over", "mean"),
        )
        for _, row in cal.iterrows():
            print(f"    n={int(row['n']):4d}   predicted={row['predicted']*100:5.1f}%   actual={row['actual']*100:5.1f}%")

    print("\n--- Watchlist sanity check (most recent test-period game) ---")
    for name in ["Caitlin Clark", "Napheesa Collier", "Kayla Thornton"]:
        rows = test[test["player_name"] == name].sort_values("game_date")
        if rows.empty:
            print(f"  {name}: no test-period games")
            continue
        row = rows.iloc[-1]
        print(
            f"  {name} ({row['game_date'].date()}): predicted_mean={row['predicted_mean_pts']:.1f} pts, "
            f"line={row['line']}, actual={row['pts']}, "
            f"P(over)_nb={row['p_nb']*100:.1f}%, P(over)_emp={row['p_emp']*100:.1f}%, "
            f"P(over)_blend={row['p_blend']*100:.1f}%, "
            f"variance_inflation={row['variance_inflation']}"
        )

    return test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="../data/wnba.db")
    args = parser.parse_args()

    df, k_rate, k_minutes, k_opponent = build_feature_table(args.db)
    print(f"Estimated shrinkage k -- rate: {k_rate:.2f} games, minutes: {k_minutes:.2f} games, "
          f"opponent index: {k_opponent:.2f} games (was a fixed 5 before this fix)")
    evaluate(df, args.db)


if __name__ == "__main__":
    main()
