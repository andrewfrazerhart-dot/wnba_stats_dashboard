"""
seed_mock_data.py

Generates realistic-but-fake WNBA player game logs for testing the
dashboard/schema before real data is wired up. Builds dim_player,
fact_player_game, and player_game_features exactly per sql/schema.sql.

Run from anywhere -- paths are resolved relative to this file's location:
    python seed_mock_data.py

Expects sql/schema.sql to exist as a sibling of this file's parent dir
(i.e. project_root/sql/schema.sql), and writes to project_root/data/wnba.db.

NOTE: this generates each player's game log independently (random opponent,
random home/away) rather than simulating a shared league schedule where
teammates share game_ids. That's fine for testing hit-rate / rolling-window
logic per player, which is all this dashboard needs -- it is NOT meant to
be internally consistent as a full league schedule (e.g. two players on
the same team won't show the same final score for a shared game).
"""

import random
import sqlite3
from datetime import date, timedelta
from pathlib import Path

from compute_features import compute_features, insert_features

random.seed(42)

BASE_DIR = Path(__file__).resolve().parent.parent
SCHEMA_PATH = BASE_DIR / "sql" / "schema.sql"
DB_PATH = BASE_DIR / "data" / "wnba.db"

TEAMS = ["ATL", "CHI", "CONN", "DAL", "GSV", "IND", "LVA",
         "LA", "MIN", "NYL", "PHO", "SEA", "WAS"]

# season_id -> (regular season start, regular season end, is_current_partial)
SEASONS = {
    2024: (date(2024, 5, 14), date(2024, 9, 15), False),
    2025: (date(2025, 5, 16), date(2025, 9, 14), False),
    2026: (date(2026, 5, 16), date(2026, 7, 20), True),  # partial -- "today" is 2026-07-24
}

# a handful of mock players with a rough "skill profile" used only to
# generate plausible-looking box scores -- these fields are NOT part of
# the real schema, just generation inputs.
PLAYERS = [
    dict(player_id=1, player_name="Aria Thompson", birthdate="1999-03-14",
         draft_year=2021, draft_position=4, height_in=73,
         main_position="Guard", position_detail="SG",
         home_team="ATL", pts_mean=17.5, starter_prob=0.9),
    dict(player_id=2, player_name="Devon Marshall", birthdate="1997-11-02",
         draft_year=2019, draft_position=1, height_in=76,
         main_position="Forward", position_detail="PF",
         home_team="LVA", pts_mean=21.0, starter_prob=0.97),
    dict(player_id=3, player_name="Kayla Ruiz", birthdate="2001-06-22",
         draft_year=2023, draft_position=12, height_in=70,
         main_position="Guard", position_detail="PG",
         home_team="NYL", pts_mean=9.5, starter_prob=0.4),
    dict(player_id=4, player_name="Simone Baker", birthdate="1996-01-30",
         draft_year=2018, draft_position=6, height_in=78,
         main_position="Center", position_detail="C",
         home_team="SEA", pts_mean=13.0, starter_prob=0.8),
    dict(player_id=5, player_name="Nina Petrova", birthdate="2000-09-09",
         draft_year=2022, draft_position=3, height_in=74,
         main_position="Forward", position_detail="SF",
         home_team="CHI", pts_mean=15.5, starter_prob=0.75),
]

# Player 2 gets traded mid-2025 season, to exercise the
# "team varies by game, not fixed on dim_player" design decision.
TRADE = dict(player_id=2, season=2025, new_team="DAL",
             effective_date=date(2025, 7, 15))


def build_schema():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DB_PATH.exists():
        DB_PATH.unlink()
    conn = sqlite3.connect(DB_PATH)
    with open(SCHEMA_PATH) as f:
        conn.executescript(f.read())
    conn.commit()
    return conn


def insert_players(conn):
    rows = [
        (p["player_id"], p["player_name"], p["birthdate"], p["draft_year"],
         p["draft_position"], p["height_in"], p["main_position"],
         p["position_detail"])
        for p in PLAYERS
    ]
    conn.executemany(
        """INSERT INTO dim_player
           (player_id, player_name, birthdate, draft_year, draft_position,
            height_in, main_position, position_detail)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )
    conn.commit()


def team_for(player, season, game_date):
    if (TRADE["player_id"] == player["player_id"]
            and TRADE["season"] == season
            and game_date >= TRADE["effective_date"]):
        return TRADE["new_team"]
    return player["home_team"]


def generate_schedule_dates(start, end, n_games):
    """Pick n_games dates between start/end, spaced 2-4 days apart, with
    occasional longer gaps (all-star break, etc.) so days_rest/b2b vary."""
    dates = []
    cur = start
    while cur <= end and len(dates) < n_games:
        dates.append(cur)
        gap = random.choices([1, 2, 3, 4, 7], weights=[10, 35, 30, 20, 5])[0]
        cur += timedelta(days=gap)
    return dates


def gen_box_score(pts_mean, started):
    """Plausible correlated box score stats given a target points mean."""
    pts = max(0, round(random.gauss(pts_mean, pts_mean * 0.35)))
    minutes = round(random.gauss(28 if started else 14, 5), 1)
    minutes = max(1.0, min(minutes, 40.0))

    fga = max(1, round(pts / 2.1 + random.gauss(0, 2)))
    fgm = max(0, min(fga, round(fga * random.uniform(0.35, 0.55))))
    fg3a = max(0, round(fga * random.uniform(0.15, 0.4)))
    fg3m = max(0, min(fg3a, round(fg3a * random.uniform(0.25, 0.45))))
    remaining_pts = max(0, pts - 2 * (fgm - fg3m) - 3 * fg3m)
    fta = max(0, remaining_pts + random.randint(-1, 2))
    ftm = max(0, min(fta, remaining_pts))

    reb_off = max(0, round(random.gauss(1.5, 1.2)))
    reb_def = max(0, round(random.gauss(4.0, 2.0)))
    ast = max(0, round(random.gauss(3.5, 2.2)))
    stl = max(0, round(random.gauss(1.1, 1.0)))
    blk = max(0, round(random.gauss(0.6, 0.8)))
    tov = max(0, round(random.gauss(2.2, 1.4)))
    pf = max(0, min(6, round(random.gauss(2.3, 1.2))))

    game_score = (pts + 0.4 * fgm - 0.7 * fga - 0.4 * (fta - ftm)
                  + 0.7 * reb_off + 0.3 * reb_def + stl + 0.7 * ast
                  + 0.7 * blk - 0.4 * pf - tov)

    return dict(pts=pts, minutes=minutes, fgm=fgm, fga=fga, fg3m=fg3m,
                fg3a=fg3a, ftm=ftm, fta=fta, reb_off=reb_off, reb_def=reb_def,
                reb_tot=reb_off + reb_def, ast=ast, stl=stl, blk=blk,
                tov=tov, pf=pf, game_score=round(game_score, 1))


def generate_games(player):
    """Returns a list of fact_player_game row dicts for one player, all seasons."""
    rows = []
    game_counter = 0
    prev_game_date_by_season = {}

    for season, (start, end, is_partial) in SEASONS.items():
        n_games = random.randint(14, 20) if is_partial else random.randint(32, 36)
        dates = generate_schedule_dates(start, end, n_games)
        prev_date = None

        for game_date in dates:
            game_counter += 1
            game_id = f"G{player['player_id']}_{season}_{game_counter:03d}"
            team = team_for(player, season, game_date)
            opponent = random.choice([t for t in TEAMS if t != team])
            home_away = random.choice(["H", "A"])
            stage = "Regular"

            if prev_date is None:
                days_rest = None
                b2b = 0
            else:
                days_rest = (game_date - prev_date).days
                b2b = 1 if days_rest == 1 else 0
            prev_date = game_date

            dnp = 1 if random.random() < 0.07 else 0

            team_score = random.randint(68, 96)
            opp_score = random.randint(68, 96)
            result = "W" if team_score > opp_score else "L"

            row = dict(
                game_id=game_id, player_id=player["player_id"], season=season,
                game_date=game_date.isoformat(), stage=stage, team=team,
                opponent=opponent, home_away=home_away, days_rest=days_rest,
                b2b=b2b, dnp=dnp,
                started=None if dnp else (1 if random.random() < player["starter_prob"] else 0),
                team_score=team_score, opp_score=opp_score, result=result,
            )

            if dnp:
                row.update(minutes=None, pts=None, reb_off=None, reb_def=None,
                           reb_tot=None, ast=None, stl=None, blk=None, tov=None,
                           pf=None, fgm=None, fga=None, fg3m=None, fg3a=None,
                           ftm=None, fta=None, game_score=None)
            else:
                row.update(gen_box_score(player["pts_mean"], row["started"]))

            rows.append(row)

    return rows


def insert_games(conn, rows):
    cols = ["game_id", "player_id", "season", "game_date", "stage", "team",
            "opponent", "home_away", "days_rest", "b2b", "dnp", "started",
            "team_score", "opp_score", "result", "minutes", "pts", "reb_off",
            "reb_def", "reb_tot", "ast", "stl", "blk", "tov", "pf", "fgm",
            "fga", "fg3m", "fg3a", "ftm", "fta", "game_score"]
    placeholders = ", ".join(["?"] * len(cols))
    conn.executemany(
        f"INSERT INTO fact_player_game ({', '.join(cols)}) VALUES ({placeholders})",
        [tuple(r[c] for c in cols) for r in rows],
    )
    conn.commit()


def main():
    conn = build_schema()
    insert_players(conn)

    all_games = []
    for player in PLAYERS:
        all_games.extend(generate_games(player))
    insert_games(conn, all_games)

    all_features = compute_features(all_games)
    insert_features(conn, all_features)

    n_players = conn.execute("SELECT COUNT(*) FROM dim_player").fetchone()[0]
    n_games = conn.execute("SELECT COUNT(*) FROM fact_player_game").fetchone()[0]
    n_dnp = conn.execute("SELECT COUNT(*) FROM fact_player_game WHERE dnp = 1").fetchone()[0]
    n_features = conn.execute("SELECT COUNT(*) FROM player_game_features").fetchone()[0]

    print(f"Seeded database at: {DB_PATH}")
    print(f"  players:            {n_players}")
    print(f"  player-game rows:   {n_games}  ({n_dnp} marked DNP)")
    print(f"  feature rows:       {n_features}")

    print("\nSanity check -- hit rates for Aria Thompson (player_id=1), 2025 season:")
    q = """
        SELECT home_away, COUNT(*) AS games,
               ROUND(AVG(CASE WHEN pts >= 10 THEN 1.0 ELSE 0 END), 3) AS pct_10plus,
               ROUND(AVG(CASE WHEN pts >= 15 THEN 1.0 ELSE 0 END), 3) AS pct_15plus,
               ROUND(AVG(CASE WHEN pts >= 20 THEN 1.0 ELSE 0 END), 3) AS pct_20plus
        FROM v_dashboard
        WHERE player_id = 1 AND season = 2025 AND dnp = 0
        GROUP BY home_away
    """
    for row in conn.execute(q).fetchall():
        print(f"  {row}")

    conn.close()


if __name__ == "__main__":
    main()
