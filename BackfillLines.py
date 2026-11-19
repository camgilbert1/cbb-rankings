"""
Historical betting lines from ESPN, for honest backtests of the bet rules (#12 / #20).

ESPN's scoreboard lists each season's games; each finished game's summary page keeps its DraftKings line
(pickcenter): spread and over/under. Saved to workspace.default.historical_lines with the final score.

    python BackfillLines.py 2026          # 2025-26 season (Nov 2025 - Apr 2026)
    python BackfillLines.py 2025 --local  # write a CSV only, no Databricks

home_spread uses the same convention as vegas_spread everywhere else: the HOME team's spread, negative =
home favored. Its sign comes from ESPN's favorite flags, not from the raw number, so it can't be flipped.
"""

import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

import pandas as pd
import requests

SB = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard"
SM = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/summary"
TABLE = 'workspace.default.historical_lines'


def get(url, params, tries=3):
    for i in range(tries):
        try:
            r = requests.get(url, params=params, timeout=20)
            if r.ok:
                return r.json()
        except Exception:
            pass
        time.sleep(1 + 2 * i)
    return None


def season_games(season):
    """Finished D1 games of a season (season 2026 = 2025-26), from the daily scoreboards."""
    d, end = date(season - 1, 11, 1), date(season, 4, 10)
    games = []
    while d <= end:
        js = get(SB, {'dates': d.strftime('%Y%m%d'), 'limit': 400, 'groups': 50}) or {}
        for e in js.get('events', []):
            c = e['competitions'][0]
            if e['status']['type']['state'] != 'post':
                continue
            side = {x['homeAway']: x for x in c['competitors']}
            if 'home' not in side or 'away' not in side:
                continue
            games.append({'season': season, 'game_date': d.isoformat(), 'event_id': e['id'],
                          'home_team': side['home']['team']['displayName'], 'away_team': side['away']['team']['displayName'],
                          'home_id': int(side['home']['team']['id']), 'away_id': int(side['away']['team']['id']),
                          'neutral_site': bool(c.get('neutralSite', False)),
                          'home_score': int(side['home']['score']), 'away_score': int(side['away']['score'])})
        d += timedelta(days=1)
    return games


CORE_ODDS = ("https://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/"
             "events/{eid}/competitions/{eid}/odds")


def _parse(game, p):
    """A line dict from one ESPN odds entry (summary pickcenter or core odds item), or None."""
    spread, total = p.get('spread'), p.get('overUnder')
    if spread is None and total is None:
        return None
    home_fav = (p.get('homeTeamOdds') or {}).get('favorite')
    away_fav = (p.get('awayTeamOdds') or {}).get('favorite')
    if spread is not None:
        mag = abs(float(spread))
        # Anchor the sign to ESPN's favorite flags: home favored -> negative home spread
        home_spread = 0.0 if mag == 0 else (-mag if home_fav else mag if away_fav else None)
    else:
        home_spread = None
    return {**game, 'home_spread': home_spread, 'total_line': float(total) if total is not None else None,
            'provider': (p.get('provider') or {}).get('name'), 'details': p.get('details')}


def line_for(game):
    """The game's pre-game line: ESPN summary page (DraftKings, 2025-26 on); for older seasons, ESPN's
    detailed odds feed (ESPN BET in 2024-25), skipping live in-game providers. None if not kept."""
    js = get(SM, {'event': game['event_id']}) or {}
    for p in js.get('pickcenter') or []:
        line = _parse(game, p)
        if line:
            return line
    core = get(CORE_ODDS.format(eid=game['event_id']), {}) or {}
    for ref in core.get('items') or []:
        p = ref if 'spread' in ref or 'overUnder' in ref else (get(ref['$ref'].replace('http://', 'https://'), {}) or {})
        name = ((p.get('provider') or {}).get('name') or '').lower()
        if 'live' in name:
            continue
        line = _parse(game, p)
        if line:
            return line
    return {**game, 'home_spread': None, 'total_line': None, 'provider': None, 'details': None}


def upload(df):
    import TrainModel as T
    conn = T.db_connect(); cur = conn.cursor()
    cur.execute(f"""CREATE TABLE IF NOT EXISTS {TABLE} (
        season INT, game_date DATE, event_id STRING, home_team STRING, away_team STRING, home_id INT, away_id INT,
        neutral_site BOOLEAN, home_score INT, away_score INT, home_spread DOUBLE, total_line DOUBLE,
        provider STRING, details STRING) USING DELTA""")
    seasons = sorted(df['season'].unique())
    cur.execute(f"DELETE FROM {TABLE} WHERE season IN ({', '.join(str(int(s)) for s in seasons)})")
    q = lambda v: 'NULL' if v is None or (isinstance(v, float) and pd.isna(v)) else (
        ('TRUE' if v else 'FALSE') if isinstance(v, bool) else repr(v) if isinstance(v, (int, float)) else "'" + str(v).replace("'", "''") + "'")
    cols = ['season', 'game_date', 'event_id', 'home_team', 'away_team', 'home_id', 'away_id', 'neutral_site',
            'home_score', 'away_score', 'home_spread', 'total_line', 'provider', 'details']
    recs = df[cols].to_dict('records')
    for i in range(0, len(recs), 500):
        vals = ', '.join('(' + ', '.join(q(r[c]) for c in cols) + ')' for r in recs[i:i + 500])
        cur.execute(f"INSERT INTO {TABLE} ({', '.join(cols)}) VALUES {vals}")
    cur.close(); conn.close()


def main():
    season = int(sys.argv[1]) if len(sys.argv) > 1 else 2026
    t0 = time.time()
    games = season_games(season)
    print(f"{season}: {len(games)} finished games ({time.time() - t0:.0f}s)")
    with ThreadPoolExecutor(max_workers=8) as pool:
        rows = list(pool.map(line_for, games))
    df = pd.DataFrame(rows)
    n_sp, n_to = df['home_spread'].notna().sum(), df['total_line'].notna().sum()
    print(f"  lines: spread {n_sp} ({n_sp / len(df):.0%}), O/U {n_to} ({n_to / len(df):.0%}) ({time.time() - t0:.0f}s)")
    df.to_csv(f"historical_lines_{season}.csv", index=False)
    if '--local' not in sys.argv:
        upload(df)
        print(f"  saved to {TABLE}")


if __name__ == '__main__':
    main()
