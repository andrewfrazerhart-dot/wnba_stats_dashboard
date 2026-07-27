"""
wnba_client.py

Thin wrapper around the (undocumented) stats.wnba.com JSON API.

IMPORTANT: plain HTTP clients (curl, Python `requests`) get silently
stalled by Akamai bot-protection in front of stats.wnba.com/stats.nba.com
-- the TCP connection succeeds but the server never responds. The same
request succeeds instantly when it originates from inside a real browser
session (confirmed empirically). So this client drives a real (headless)
Chromium browser via Playwright: it loads a normal wnba.com page once to
pick up whatever session state the bot-check wants, then issues every
API call as a `fetch()` executed inside that page's JS context.

This means ingest.py must run somewhere with real outbound internet
access (same constraint the original README flagged) and needs
`playwright install chromium` run once after `pip install`.
"""

import json
import time

from playwright.sync_api import sync_playwright

BASE = "https://stats.wnba.com/stats"
WARMUP_URL = "https://www.wnba.com/players"
LEAGUE_ID = "10"  # WNBA, within the shared stats.nba.com/stats.wnba.com schema

RETRIES = 3
RETRY_DELAY_S = 2.0
CALL_DELAY_S = 0.4  # small politeness delay between calls


def _result_set_to_dicts(payload, result_set_name=None):
    """Converts one resultSet (headers + rowSet) into a list of dicts.
    Assumes a single relevant result set unless result_set_name is given --
    if an endpoint ever returns multiple named result sets, pass the name
    explicitly rather than guessing which one is relevant."""
    result_sets = payload["resultSets"]
    if result_set_name:
        rs = next(rs for rs in result_sets if rs["name"] == result_set_name)
    else:
        rs = result_sets[0]
    headers = rs["headers"]
    return [dict(zip(headers, row)) for row in rs["rowSet"]]


class WNBAClient:
    """Use as a context manager so the browser is always cleaned up:
        with WNBAClient() as client:
            players = client.get_active_players(2025)
    """

    def __init__(self, headless=True):
        self._headless = headless
        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None

    def __enter__(self):
        self._playwright = sync_playwright().start()
        # Getting past stats.wnba.com's Akamai bot-check took two things,
        # confirmed empirically by bisecting failures:
        #   1. channel="chrome" -- the bundled Playwright Chromium ships a
        #      stripped-down "headless shell" binary that gets silently
        #      stalled (connection hangs, no response). Real Chrome doesn't.
        #   2. Hiding navigator.webdriver + a realistic UA -- without this,
        #      real Chrome-via-CDP still gets an actual "Access Denied" page.
        self._browser = self._playwright.chromium.launch(
            headless=self._headless, channel="chrome",
            args=["--disable-blink-features=AutomationControlled"],
        )
        self._context = self._browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                       "(KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36",
            viewport={"width": 1280, "height": 800},
        )
        self._page = self._context.new_page()
        self._page.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        self._page.goto(WARMUP_URL, wait_until="domcontentloaded", timeout=30000)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._browser:
            self._browser.close()
        if self._playwright:
            self._playwright.stop()

    def _fetch_json(self, url, error_label):
        last_error = None
        for attempt in range(1, RETRIES + 1):
            result = self._page.evaluate(
                """(url) => fetch(url, {headers: {Accept: 'application/json'}})
                    .then(async r => ({ok: r.ok, status: r.status, text: await r.text()}))
                    .catch(e => ({ok: false, status: 0, text: String(e)}))""",
                url,
            )
            if result["ok"]:
                time.sleep(CALL_DELAY_S)
                return json.loads(result["text"])
            last_error = f"HTTP {result['status']}: {result['text'][:200]}"
            time.sleep(RETRY_DELAY_S * attempt)

        raise RuntimeError(f"{error_label} failed after {RETRIES} attempts: {last_error}")

    def _get(self, endpoint, params, result_set_name=None):
        query = "&".join(f"{k}={v}" for k, v in params.items())
        url = f"{BASE}/{endpoint}?{query}"
        payload = self._fetch_json(url, endpoint)
        return _result_set_to_dicts(payload, result_set_name)

    def get_active_players(self, season):
        """Full active-roster index for a season: PERSON_ID, name, TEAM_ID, etc.

        IsOnlyCurrentSeason=1 looks like it should mean "active during the
        queried season," but empirically it doesn't for past seasons --
        e.g. Season=2025 with IsOnlyCurrentSeason=1 dropped A'ja Wilson
        entirely, despite her having played a full 2025 season. Using 0 and
        filtering on ROSTERSTATUS ourselves gives correct historical
        rosters (confirmed against known players) without changing the
        current-season result at all."""
        rows = self._get("commonallplayers", {
            "LeagueID": LEAGUE_ID, "Season": season, "IsOnlyCurrentSeason": 0,
        })
        return [r for r in rows if r["ROSTERSTATUS"] == 1]

    def get_player_bio(self, player_id):
        """Bio info: birthdate, height, position, draft info."""
        rows = self._get(
            "commonplayerinfo", {"PlayerID": player_id, "LeagueID": LEAGUE_ID},
            result_set_name="CommonPlayerInfo",
        )
        return rows[0] if rows else None

    def get_player_gamelog(self, player_id, season, season_type="Regular Season"):
        """One row per game the player actually played (no DNP rows)."""
        season_type_q = season_type.replace(" ", "+")
        return self._get("playergamelog", {
            "PlayerID": player_id, "Season": season, "SeasonType": season_type_q,
            "LeagueID": LEAGUE_ID,
        })

    def get_team_gamelog(self, team_id, season, season_type="Regular Season"):
        """Full team schedule + team's own box score totals for each game --
        used both to backfill DNP rows and to look up opponent score."""
        season_type_q = season_type.replace(" ", "+")
        return self._get("teamgamelog", {
            "TeamID": team_id, "Season": season, "SeasonType": season_type_q,
            "LeagueID": LEAGUE_ID,
        })

    def get_schedule(self, season):
        """Full-season schedule for every team, played AND upcoming, in a
        single call -- this is wnba.com's own first-party schedule API
        (not stats.wnba.com/stats), the only source here for not-yet-played
        games. Each game already carries both teams' win-loss record as of
        that game, and the final score once played, so this alone covers
        next-game lookups, records, and PPG-for/against without needing
        any additional per-team calls."""
        url = f"https://www.wnba.com/api/schedule?season={season}&regionId=1"
        payload = self._fetch_json(url, "schedule")
        return payload["leagueSchedule"]["gameDates"]
