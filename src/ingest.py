"""
ingest.py

Pulls real WNBA data via wnba_client.py and loads it into the schema in
sql/schema.sql: dim_player, fact_player_game (including derived DNP
rows), then runs compute_features.py to populate player_game_features.

Tracks the full active roster for the most recent of --seasons (the
"current" season) and backfills each of those players' game logs for
every season in --seasons.

Run from anywhere -- paths are resolved relative to this file's location:
    cd src
    python ingest.py --seasons 2026,2025,2024

Requires real outbound internet access and `playwright install chromium`
having been run once (see wnba_client.py for why a real browser is
needed instead of plain HTTP requests).
"""

import argparse
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

from compute_features import compute_features, insert_features
from wnba_client import WNBAClient

BASE_DIR = Path(__file__).resolve().parent.parent
SCHEMA_PATH = BASE_DIR / "sql" / "schema.sql"
DEFAULT_DB = BASE_DIR / "data" / "wnba.db"

SEASON_TYPE = "Regular Season"  # playoffs deferred, see README limitations


def parse_height(height_str):
    """'5-10' -> 70 (inches). Returns None if missing/unparseable."""
    if not height_str or "-" not in height_str:
        return None
    feet, inches = height_str.split("-")
    return int(feet) * 12 + int(inches)


def parse_bio_date(birthdate_str):
    """'1996-07-07T00:00:00' -> '1996-07-07'."""
    if not birthdate_str:
        return None
    return birthdate_str.split("T")[0]


def parse_game_date(date_str):
    """'Sep 09, 2025' -> '2025-09-09'."""
    return datetime.strptime(date_str, "%b %d, %Y").date().isoformat()


def parse_matchup(matchup):
    """'LAS @ PHX' -> ('LAS', 'PHX', 'A'); 'LAS vs. PHX' -> ('LAS', 'PHX', 'H')."""
    team, rest = matchup.split(" ", 1)
    if rest.startswith("@"):
        return team, rest.replace("@", "").strip(), "A"
    return team, rest.replace("vs.", "").strip(), "H"


def game_score(row):
    """Hollinger Game Score, from box score fields already on the row."""
    return round(
        row["pts"] + 0.4 * row["fgm"] - 0.7 * row["fga"] - 0.4 * (row["fta"] - row["ftm"])
        + 0.7 * row["reb_off"] + 0.3 * row["reb_def"] + row["stl"] + 0.7 * row["ast"]
        + 0.7 * row["blk"] - 0.4 * row["pf"] - row["tov"],
        1,
    )


def build_dim_player_row(bio):
    draft_year = bio.get("DRAFT_YEAR")
    draft_year = int(draft_year) if draft_year and draft_year.isdigit() else None
    draft_number = bio.get("DRAFT_NUMBER")
    draft_position = int(draft_number) if draft_number and draft_number.isdigit() else None
    years_experience = bio.get("SEASON_EXP")
    position = (bio.get("POSITION") or "").split("-")[0] or None

    return dict(
        player_id=bio["PERSON_ID"],
        player_name=bio["DISPLAY_FIRST_LAST"],
        birthdate=parse_bio_date(bio.get("BIRTHDATE")),
        draft_year=draft_year,
        draft_position=draft_position,
        years_experience=years_experience,
        rookie_flag=1 if years_experience == 0 else 0,
        height_in=parse_height(bio.get("HEIGHT")),
        main_position=position,
        position_detail=position,  # API doesn't expose PG/SG/SF/PF/C granularity
    )


def fetch_team_schedules(client, season, team_ids):
    """Returns (schedule_by_team, score_by_game_team):
    schedule_by_team: {team_id: [{game_id, game_date, team_abbrev, opponent_abbrev,
                                   home_away, team_score, wl}, ...]} sorted by date
    score_by_game_team: {game_id: {team_id: pts}} -- for opponent-score lookup
    """
    schedule_by_team = {}
    score_by_game_team = {}

    for team_id in team_ids:
        rows = client.get_team_gamelog(team_id, season, SEASON_TYPE)
        parsed = []
        for r in rows:
            team_abbrev, opponent_abbrev, home_away = parse_matchup(r["MATCHUP"])
            game_date = parse_game_date(r["GAME_DATE"])
            parsed.append(dict(
                game_id=r["Game_ID"], game_date=game_date, team_abbrev=team_abbrev,
                opponent_abbrev=opponent_abbrev, home_away=home_away,
                team_score=r["PTS"], wl=r["WL"],
            ))
            score_by_game_team.setdefault(r["Game_ID"], {})[team_id] = r["PTS"]
        parsed.sort(key=lambda g: g["game_date"])
        schedule_by_team[team_id] = parsed

    return schedule_by_team, score_by_game_team


def build_player_season_rows(player_id, season, gamelog, team_id, schedule_by_team,
                              score_by_game_team, abbrev_to_team_id):
    """Merges playergamelog (played games) with the player's team schedule
    (to derive DNP rows for games the team played but the player didn't
    appear in) into a single, date-ordered list of fact_player_game rows."""
    played_by_game_id = {}
    for r in gamelog:
        game_date = parse_game_date(r["GAME_DATE"])
        team_abbrev, opponent_abbrev, home_away = parse_matchup(r["MATCHUP"])
        row = dict(
            game_id=r["Game_ID"], player_id=player_id, season=season,
            game_date=game_date, stage="Regular",
            team=team_abbrev, opponent=opponent_abbrev, home_away=home_away,
            dnp=0, started=None, result=r["WL"],
            minutes=r["MIN"], pts=r["PTS"], reb_off=r["OREB"], reb_def=r["DREB"],
            reb_tot=r["REB"], ast=r["AST"], stl=r["STL"], blk=r["BLK"], tov=r["TOV"],
            pf=r["PF"], fgm=r["FGM"], fga=r["FGA"], fg3m=r["FG3M"], fg3a=r["FG3A"],
            ftm=r["FTM"], fta=r["FTA"],
        )
        row["game_score"] = game_score(row)
        played_by_game_id[r["Game_ID"]] = row

    team_schedule = schedule_by_team.get(team_id, [])
    all_rows = []
    for g in team_schedule:
        if g["game_id"] in played_by_game_id:
            row = played_by_game_id[g["game_id"]]
        else:
            row = dict(
                game_id=g["game_id"], player_id=player_id, season=season,
                game_date=g["game_date"], stage="Regular",
                team=g["team_abbrev"], opponent=g["opponent_abbrev"], home_away=g["home_away"],
                dnp=1, started=None, result=g["wl"],
                minutes=None, pts=None, reb_off=None, reb_def=None, reb_tot=None,
                ast=None, stl=None, blk=None, tov=None, pf=None, fgm=None, fga=None,
                fg3m=None, fg3a=None, ftm=None, fta=None, game_score=None,
            )

        opp_team_id = abbrev_to_team_id.get(row["opponent"])
        team_score = score_by_game_team.get(row["game_id"], {}).get(team_id)
        opp_score = score_by_game_team.get(row["game_id"], {}).get(opp_team_id) if opp_team_id else None
        row["team_score"] = team_score
        row["opp_score"] = opp_score
        all_rows.append(row)

    all_rows.sort(key=lambda r: r["game_date"])
    prev_date = None
    for row in all_rows:
        gdate = datetime.strptime(row["game_date"], "%Y-%m-%d").date()
        if prev_date is None:
            row["days_rest"] = None
            row["b2b"] = 0
        else:
            days_rest = (gdate - prev_date).days
            row["days_rest"] = days_rest
            row["b2b"] = 1 if days_rest == 1 else 0
        prev_date = gdate

    # a game a player was on a DIFFERENT team for (traded away before it, or
    # not yet on the roster) isn't a real DNP for them -- playergamelog only
    # returns games for the team(s) they were actually on, so this only
    # affects the schedule-derived DNP rows tied to their CURRENT team_id;
    # left as a known approximation for intra-season trades.
    return all_rows


def build_schema(db_path):
    """Always starts fresh -- a full ingest run tracks the whole current
    active roster, so there's no meaningful partial/incremental state to
    preserve, and this guarantees no leftover mock data lingers alongside
    real player IDs."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(db_path)
    with open(SCHEMA_PATH) as f:
        conn.executescript(f.read())
    conn.commit()
    return conn


def insert_players(conn, players_by_id):
    rows = [
        (p["player_id"], p["player_name"], p["birthdate"], p["draft_year"],
         p["draft_position"], p["years_experience"], p["rookie_flag"],
         p["height_in"], p["main_position"], p["position_detail"])
        for p in players_by_id.values()
    ]
    conn.executemany(
        """INSERT OR IGNORE INTO dim_player
           (player_id, player_name, birthdate, draft_year, draft_position,
            years_experience, rookie_flag, height_in, main_position, position_detail)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )
    conn.commit()


def insert_games(conn, rows):
    if not rows:
        return
    cols = ["game_id", "player_id", "season", "game_date", "stage", "team",
            "opponent", "home_away", "days_rest", "b2b", "dnp", "started",
            "team_score", "opp_score", "result", "minutes", "pts", "reb_off",
            "reb_def", "reb_tot", "ast", "stl", "blk", "tov", "pf", "fgm",
            "fga", "fg3m", "fg3a", "ftm", "fta", "game_score"]
    placeholders = ", ".join(["?"] * len(cols))
    conn.executemany(
        f"INSERT OR IGNORE INTO fact_player_game ({', '.join(cols)}) VALUES ({placeholders})",
        [tuple(r[c] for c in cols) for r in rows],
    )
    conn.commit()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument("--seasons", default="2026,2025,2024",
                         help="Comma-separated, most-recent (current) season first.")
    args = parser.parse_args()

    db_path = Path(args.db)
    seasons = [int(s) for s in args.seasons.split(",")]
    current_season = seasons[0]

    conn = build_schema(db_path)

    with WNBAClient() as client:
        print(f"Fetching active roster for {current_season}...")
        roster = client.get_active_players(current_season)
        print(f"  {len(roster)} active players found.")

        # per-season roster snapshot -> player's team_id that season + abbrev map
        team_id_by_player_season = {}
        abbrev_to_team_id_by_season = {}
        team_ids_by_season = {}
        for season in seasons:
            season_roster = roster if season == current_season else client.get_active_players(season)
            abbrev_map = {}
            for p in season_roster:
                if p["TEAM_ID"]:
                    abbrev_map[p["TEAM_ABBREVIATION"]] = p["TEAM_ID"]
                team_id_by_player_season[(p["PERSON_ID"], season)] = p["TEAM_ID"] or None
            abbrev_to_team_id_by_season[season] = abbrev_map
            team_ids_by_season[season] = sorted(set(abbrev_map.values()))

        schedule_and_scores_by_season = {}
        for season in seasons:
            print(f"Fetching team schedules for {season} "
                  f"({len(team_ids_by_season[season])} teams)...")
            schedule_and_scores_by_season[season] = fetch_team_schedules(
                client, season, team_ids_by_season[season]
            )

        players_by_id = {}
        all_fact_rows = []
        n_players = len(roster)
        for i, p in enumerate(roster, start=1):
            player_id = p["PERSON_ID"]
            print(f"[{i}/{n_players}] {p['DISPLAY_FIRST_LAST']} ({p['TEAM_ABBREVIATION']})...", end=" ")
            try:
                bio = client.get_player_bio(player_id)
                if bio:
                    players_by_id[player_id] = build_dim_player_row(bio)

                player_row_count = 0
                for season in seasons:
                    team_id = team_id_by_player_season.get((player_id, season))
                    if team_id is None:
                        continue  # player wasn't rostered this season -- nothing to fetch
                    schedule_by_team, score_by_game_team = schedule_and_scores_by_season[season]
                    gamelog = client.get_player_gamelog(player_id, season, SEASON_TYPE)
                    rows = build_player_season_rows(
                        player_id, season, gamelog, team_id, schedule_by_team,
                        score_by_game_team, abbrev_to_team_id_by_season[season],
                    )
                    all_fact_rows.extend(rows)
                    player_row_count += len(rows)
                print(f"{player_row_count} games.")
            except Exception as e:
                print(f"SKIPPED ({type(e).__name__}: {e})")

    print(f"\nInserting {len(players_by_id)} players and {len(all_fact_rows)} game rows...")
    insert_players(conn, players_by_id)
    insert_games(conn, all_fact_rows)

    print("Computing rolling/season features...")
    conn.execute("DELETE FROM player_game_features")
    all_games_in_db = [
        dict(zip([c[0] for c in conn.execute("SELECT * FROM fact_player_game LIMIT 0").description], row))
        for row in conn.execute("SELECT * FROM fact_player_game").fetchall()
    ]
    features = compute_features(all_games_in_db)
    insert_features(conn, features)

    n_players_total = conn.execute("SELECT COUNT(*) FROM dim_player").fetchone()[0]
    n_games_total = conn.execute("SELECT COUNT(*) FROM fact_player_game").fetchone()[0]
    n_dnp_total = conn.execute("SELECT COUNT(*) FROM fact_player_game WHERE dnp = 1").fetchone()[0]
    print(f"\nDone. Database at: {db_path}")
    print(f"  players (total in db):     {n_players_total}")
    print(f"  player-game rows (total):  {n_games_total}  ({n_dnp_total} DNP)")
    print(f"  feature rows:              {len(features)}")

    conn.close()


if __name__ == "__main__":
    sys.exit(main())
