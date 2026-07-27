"""
compute_features.py

Builds player_game_features rows from a list of fact_player_game row
dicts. Shared by seed_mock_data.py (mock data) and ingest.py (real data)
so both paths compute rolling/season-average stats identically.

All *_prior values are computed over games strictly before the current
row -- current row's own box score is never included. DNP games are
excluded from every average but still get a features row (with null
threshold flags).
"""

from datetime import date, timedelta


def compute_features(rows):
    features = []

    by_player_season = {}
    for r in rows:
        key = (r["player_id"], r["season"])
        by_player_season.setdefault(key, []).append(r)

    for (player_id, season), games in by_player_season.items():
        games.sort(key=lambda r: r["game_date"])
        played_so_far = []  # list of (game_date, pts, reb_tot, ast), non-dnp only

        season_final_pts = [g["pts"] for g in games if not g["dnp"]]
        season_final_reb = [g["reb_tot"] for g in games if not g["dnp"]]
        season_final_ast = [g["ast"] for g in games if not g["dnp"]]
        avg_final_pts = sum(season_final_pts) / len(season_final_pts) if season_final_pts else None
        avg_final_reb = sum(season_final_reb) / len(season_final_reb) if season_final_reb else None
        avg_final_ast = sum(season_final_ast) / len(season_final_ast) if season_final_ast else None

        for g in games:
            gdate = date.fromisoformat(g["game_date"])
            n_prior = len(played_so_far)

            def avg(idx):
                vals = [p[idx] for p in played_so_far]
                return sum(vals) / len(vals) if vals else None

            season_avg_pts_prior = avg(1)
            season_avg_reb_prior = avg(2)
            season_avg_ast_prior = avg(3)

            last5 = played_so_far[-5:]
            avg_pts_l5 = sum(p[1] for p in last5) / len(last5) if last5 else None
            avg_reb_l5 = sum(p[2] for p in last5) / len(last5) if last5 else None
            avg_ast_l5 = sum(p[3] for p in last5) / len(last5) if last5 else None

            window_start = gdate - timedelta(days=14)
            l14d = [p for p in played_so_far if p[0] >= window_start]
            avg_pts_l14d = sum(p[1] for p in l14d) / len(l14d) if l14d else None
            avg_reb_l14d = sum(p[2] for p in l14d) / len(l14d) if l14d else None
            avg_ast_l14d = sum(p[3] for p in l14d) / len(l14d) if l14d else None

            if g["dnp"]:
                pts_10plus = pts_15plus = pts_20plus = None
            else:
                pts_10plus = int(g["pts"] >= 10)
                pts_15plus = int(g["pts"] >= 15)
                pts_20plus = int(g["pts"] >= 20)

            features.append(dict(
                game_id=g["game_id"], player_id=player_id, season=season,
                season_avg_pts_prior=season_avg_pts_prior, season_avg_pts_final=avg_final_pts,
                season_avg_reb_prior=season_avg_reb_prior, season_avg_reb_final=avg_final_reb,
                season_avg_ast_prior=season_avg_ast_prior, season_avg_ast_final=avg_final_ast,
                avg_pts_l5_prior=avg_pts_l5, avg_reb_l5_prior=avg_reb_l5, avg_ast_l5_prior=avg_ast_l5,
                avg_pts_l14d_prior=avg_pts_l14d, avg_reb_l14d_prior=avg_reb_l14d, avg_ast_l14d_prior=avg_ast_l14d,
                games_played_season_prior=n_prior, games_played_l14d_prior=len(l14d),
                pts_10plus=pts_10plus, pts_15plus=pts_15plus, pts_20plus=pts_20plus,
            ))

            if not g["dnp"]:
                played_so_far.append((gdate, g["pts"], g["reb_tot"], g["ast"]))

    return features


def insert_features(conn, rows):
    cols = ["game_id", "player_id", "season", "season_avg_pts_prior",
            "season_avg_pts_final", "season_avg_reb_prior", "season_avg_reb_final",
            "season_avg_ast_prior", "season_avg_ast_final", "avg_pts_l5_prior",
            "avg_reb_l5_prior", "avg_ast_l5_prior", "avg_pts_l14d_prior",
            "avg_reb_l14d_prior", "avg_ast_l14d_prior", "games_played_season_prior",
            "games_played_l14d_prior", "pts_10plus", "pts_15plus", "pts_20plus"]
    placeholders = ", ".join(["?"] * len(cols))
    conn.executemany(
        f"INSERT INTO player_game_features ({', '.join(cols)}) VALUES ({placeholders})",
        [tuple(r[c] for c in cols) for r in rows],
    )
    conn.commit()
