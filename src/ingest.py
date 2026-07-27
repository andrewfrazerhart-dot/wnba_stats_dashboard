"""
ingest.py

Pulls real WNBA data via wnba_client.py and loads it into the schema in
sql/schema.sql: dim_player, fact_player_game (including derived DNP
rows), then runs compute_features.py to populate player_game_features.

Default mode is INCREMENTAL: past (non-current) seasons are complete and
never change once loaded, so re-running this only re-checks the current
(first-listed) season for new games, and only backfills prior seasons
for players who aren't in the database yet at all (e.g. a new call-up).
Existing rows are never overwritten (INSERT OR IGNORE), so re-running
is always safe. This cuts a routine "what happened since last time"
update from ~1300 API calls down to ~200.

Run from anywhere -- paths are resolved relative to this file's location:
    cd src
    python ingest.py --seasons 2026,2025,2024              # incremental (default)
    python ingest.py --seasons 2026,2025,2024 --full-refresh  # wipe + rebuild everything

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


TEAM_SCHEDULE_DDL = """
CREATE TABLE IF NOT EXISTS team_schedule (
    season          INTEGER NOT NULL,
    game_id         TEXT NOT NULL,
    game_date       TEXT NOT NULL,
    team            TEXT NOT NULL,
    opponent        TEXT NOT NULL,
    home_away       TEXT NOT NULL CHECK (home_away IN ('H', 'A')),
    status          TEXT NOT NULL CHECK (status IN ('Scheduled', 'Final')),
    team_score      INTEGER,
    opp_score       INTEGER,
    team_wins       INTEGER,
    team_losses     INTEGER,
    PRIMARY KEY (season, team, game_id)
);
CREATE INDEX IF NOT EXISTS idx_team_schedule_team_season ON team_schedule(team, season);
"""


def build_schema(db_path, full_refresh=False):
    """full_refresh=True wipes any existing database (e.g. to clear out
    old mock data or start completely clean). Otherwise an existing
    database is reused as-is -- that's what makes incremental updates
    possible -- and a schema is only created if the file doesn't exist
    yet. TEAM_SCHEDULE_DDL runs unconditionally (CREATE ... IF NOT EXISTS)
    as a lightweight migration, so databases created before team_schedule
    existed pick it up without needing a full wipe-and-rebuild."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if full_refresh and db_path.exists():
        db_path.unlink()
    is_new = not db_path.exists()
    conn = sqlite3.connect(db_path)
    if is_new:
        with open(SCHEMA_PATH) as f:
            conn.executescript(f.read())
        conn.commit()
    conn.executescript(TEAM_SCHEDULE_DDL)
    conn.commit()
    return conn


def build_team_schedule_rows(game_dates, season):
    """Flattens the schedule feed into two rows per game (one per team's
    perspective). Regular Season only, matching SEASON_TYPE used
    elsewhere (playoffs/preseason deferred, see README limitations)."""
    rows = []
    for gd in game_dates:
        for g in gd["games"]:
            if g["seasonType"] != SEASON_TYPE:
                continue
            status = "Final" if g["gameStatus"] == 3 else "Scheduled"
            game_date = g["gameDateEst"][:10]
            home, away = g["homeTeam"], g["awayTeam"]
            home_score = home["score"] if status == "Final" else None
            away_score = away["score"] if status == "Final" else None
            rows.append(dict(
                season=season, game_id=g["gameId"], game_date=game_date,
                team=home["teamTricode"], opponent=away["teamTricode"], home_away="H",
                status=status, team_score=home_score, opp_score=away_score,
                team_wins=home["wins"], team_losses=home["losses"],
            ))
            rows.append(dict(
                season=season, game_id=g["gameId"], game_date=game_date,
                team=away["teamTricode"], opponent=home["teamTricode"], home_away="A",
                status=status, team_score=away_score, opp_score=home_score,
                team_wins=away["wins"], team_losses=away["losses"],
            ))
    return rows


def insert_team_schedule(conn, season, rows):
    """Always deletes + reinserts for this season -- cheap (one API call,
    a few hundred rows) and simplest way to pick up newly-final games and
    updated records each run, current-season only."""
    conn.execute("DELETE FROM team_schedule WHERE season = ?", (season,))
    if rows:
        cols = ["season", "game_id", "game_date", "team", "opponent", "home_away",
                "status", "team_score", "opp_score", "team_wins", "team_losses"]
        placeholders = ", ".join(["?"] * len(cols))
        conn.executemany(
            f"INSERT INTO team_schedule ({', '.join(cols)}) VALUES ({placeholders})",
            [tuple(r[c] for c in cols) for r in rows],
        )
    conn.commit()


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


def fetch_prior_season_data(client, season, player_ids_needing_backfill):
    """Only called for seasons/players that actually need first-time
    backfill -- fetches a roster snapshot for `season` (to find each
    player's team that season) and team schedules for just the teams
    those specific players were on, not the whole league."""
    season_roster = client.get_active_players(season)
    abbrev_map = {p["TEAM_ABBREVIATION"]: p["TEAM_ID"] for p in season_roster if p["TEAM_ID"]}
    team_id_by_player = {p["PERSON_ID"]: p["TEAM_ID"] or None for p in season_roster}

    relevant_team_ids = sorted({
        team_id_by_player[pid] for pid in player_ids_needing_backfill
        if team_id_by_player.get(pid)
    })
    schedule_by_team, score_by_game_team = fetch_team_schedules(client, season, relevant_team_ids)
    return team_id_by_player, abbrev_map, schedule_by_team, score_by_game_team


def recompute_features_for_seasons(conn, seasons):
    """Recomputes player_game_features for exactly the seasons touched
    this run -- a pure local operation (no API calls), so it's cheap to
    always do for the current season, but there's no need to touch
    prior seasons unless a new player's history was just backfilled
    into them."""
    if not seasons:
        return 0
    placeholders = ", ".join("?" * len(seasons))
    conn.execute(f"DELETE FROM player_game_features WHERE season IN ({placeholders})", seasons)
    cols = [c[0] for c in conn.execute("SELECT * FROM fact_player_game LIMIT 0").description]
    rows = [
        dict(zip(cols, row))
        for row in conn.execute(
            f"SELECT * FROM fact_player_game WHERE season IN ({placeholders})", seasons
        ).fetchall()
    ]
    features = compute_features(rows)
    insert_features(conn, features)
    return len(features)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument("--seasons", default="2026,2025,2024",
                         help="Comma-separated, most-recent (current) season first.")
    parser.add_argument("--full-refresh", action="store_true",
                         help="Wipe the database and re-fetch everything from scratch. "
                              "Default is incremental: only the current season is "
                              "re-checked, and prior seasons are only backfilled for "
                              "players not already in the database.")
    args = parser.parse_args()

    db_path = Path(args.db)
    seasons = [int(s) for s in args.seasons.split(",")]
    current_season = seasons[0]
    prior_seasons = seasons[1:]

    conn = build_schema(db_path, full_refresh=args.full_refresh)
    existing_player_ids = {row[0] for row in conn.execute("SELECT player_id FROM dim_player")}

    with WNBAClient() as client:
        print(f"Fetching active roster for {current_season}...")
        roster = client.get_active_players(current_season)
        new_player_ids = {p["PERSON_ID"] for p in roster if p["PERSON_ID"] not in existing_player_ids}
        print(f"  {len(roster)} active players found ({len(new_player_ids)} not yet tracked).")

        print(f"Fetching full league schedule for {current_season}...")
        game_dates = client.get_schedule(current_season)
        schedule_rows = build_team_schedule_rows(game_dates, current_season)
        n_final = sum(1 for r in schedule_rows if r["status"] == "Final") // 2
        n_scheduled = sum(1 for r in schedule_rows if r["status"] == "Scheduled") // 2
        print(f"  {n_final} completed games, {n_scheduled} upcoming games.")

        players_by_id = {}
        for p in roster:
            if p["PERSON_ID"] not in new_player_ids:
                continue  # bio barely ever changes -- skip refetching for known players
            try:
                bio = client.get_player_bio(p["PERSON_ID"])
                if bio:
                    players_by_id[p["PERSON_ID"]] = build_dim_player_row(bio)
            except Exception as e:
                print(f"  bio fetch failed for {p['DISPLAY_FIRST_LAST']}: {e}")

        # current season: always refresh for everyone, since new games can
        # appear for any tracked player, not just new ones
        current_abbrev_map = {p["TEAM_ABBREVIATION"]: p["TEAM_ID"] for p in roster if p["TEAM_ID"]}
        current_team_ids = sorted(set(current_abbrev_map.values()))
        print(f"Fetching team schedules for {current_season} ({len(current_team_ids)} teams)...")
        cur_schedule, cur_scores = fetch_team_schedules(client, current_season, current_team_ids)

        # prior seasons: only fetched at all if someone actually needs backfill,
        # and only for the teams those specific players were on
        prior_season_data = {}
        for season in prior_seasons:
            if not new_player_ids:
                continue
            print(f"Fetching backfill data for {season} ({len(new_player_ids)} new player(s))...")
            prior_season_data[season] = fetch_prior_season_data(client, season, new_player_ids)

        all_fact_rows = []
        n_players = len(roster)
        touched_seasons = {current_season}
        for i, p in enumerate(roster, start=1):
            player_id = p["PERSON_ID"]
            print(f"[{i}/{n_players}] {p['DISPLAY_FIRST_LAST']} ({p['TEAM_ABBREVIATION']})...", end=" ")
            try:
                row_count = 0
                if p["TEAM_ID"]:
                    gamelog = client.get_player_gamelog(player_id, current_season, SEASON_TYPE)
                    rows = build_player_season_rows(
                        player_id, current_season, gamelog, p["TEAM_ID"],
                        cur_schedule, cur_scores, current_abbrev_map,
                    )
                    all_fact_rows.extend(rows)
                    row_count += len(rows)

                if player_id in new_player_ids:
                    for season in prior_seasons:
                        if season not in prior_season_data:
                            continue
                        team_id_by_player, abbrev_map, schedule_by_team, score_by_game_team = prior_season_data[season]
                        team_id = team_id_by_player.get(player_id)
                        if team_id is None:
                            continue
                        gamelog = client.get_player_gamelog(player_id, season, SEASON_TYPE)
                        rows = build_player_season_rows(
                            player_id, season, gamelog, team_id,
                            schedule_by_team, score_by_game_team, abbrev_map,
                        )
                        all_fact_rows.extend(rows)
                        row_count += len(rows)
                        touched_seasons.add(season)
                print(f"{row_count} games.")
            except Exception as e:
                print(f"SKIPPED ({type(e).__name__}: {e})")

    print(f"\nInserting {len(players_by_id)} new player(s) and {len(all_fact_rows)} game rows "
          f"(existing rows are skipped automatically)...")
    insert_players(conn, players_by_id)
    insert_games(conn, all_fact_rows)
    insert_team_schedule(conn, current_season, schedule_rows)

    print(f"Recomputing features for season(s): {sorted(touched_seasons)}...")
    n_features = recompute_features_for_seasons(conn, sorted(touched_seasons))

    n_players_total = conn.execute("SELECT COUNT(*) FROM dim_player").fetchone()[0]
    n_games_total = conn.execute("SELECT COUNT(*) FROM fact_player_game").fetchone()[0]
    n_dnp_total = conn.execute("SELECT COUNT(*) FROM fact_player_game WHERE dnp = 1").fetchone()[0]
    n_schedule_total = conn.execute("SELECT COUNT(*) FROM team_schedule").fetchone()[0]
    print(f"\nDone. Database at: {db_path}")
    print(f"  players (total in db):     {n_players_total}")
    print(f"  player-game rows (total):  {n_games_total}  ({n_dnp_total} DNP)")
    print(f"  feature rows recomputed:   {n_features}")
    print(f"  team_schedule rows:        {n_schedule_total}")

    conn.close()


if __name__ == "__main__":
    sys.exit(main())
