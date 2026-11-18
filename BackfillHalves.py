"""
First-half scores from ESPN's scoreboard (competitors[].linescores; period 1 = first half).
Same request, dates (US/Eastern) and team names as BackfillHistoricalData.py, so rows join to ml_game_features.

    python BackfillHalves.py 2024 2025 2026     # whole seasons (Nov 1 - Apr 15)
    python BackfillHalves.py --recent 3         # the last 3 days (daily pipeline)

Table workspace.default.game_halves:
    season, game_date, home_team, away_team, neutral_site, home_h1, away_h1, home_final, away_final, periods
periods = number of periods ESPN lists (2 = regulation, 3+ = overtime).
"""

import sys
import time
from datetime import date, datetime, timedelta

import pytz
import requests
from dotenv import load_dotenv

load_dotenv()
TABLE = 'workspace.default.game_halves'
URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard"
COLUMNS = [('season', 'INT'), ('game_date', 'DATE'), ('home_team', 'STRING'), ('away_team', 'STRING'),
           ('neutral_site', 'BOOLEAN'), ('home_h1', 'INT'), ('away_h1', 'INT'), ('home_final', 'INT'),
           ('away_final', 'INT'), ('periods', 'INT')]


def season_of(d):
    return d.year + 1 if d.month >= 7 else d.year


def fetch_date(d, retries=3):
    for attempt in range(retries):
        try:
            r = requests.get(URL, params={'dates': d.strftime('%Y%m%d'), 'limit': 300, 'groups': 50}, timeout=20)
            if r.status_code == 429:
                time.sleep(10 * (attempt + 1)); continue
            r.raise_for_status()
            out = []
            for ev in r.json().get('events', []):
                if ev['status']['type']['state'] != 'post':
                    continue
                comp = ev['competitions'][0]
                side = {c['homeAway']: c for c in comp['competitors']}
                if 'home' not in side or 'away' not in side:
                    continue
                h, a = side['home'], side['away']
                hl, al = h.get('linescores') or [], a.get('linescores') or []
                if not hl or not al:
                    continue
                v = lambda ls, i: int(float(ls[i].get('value', ls[i].get('displayValue'))))
                try:
                    out.append({'season': season_of(d), 'game_date': d.isoformat(),
                                'home_team': h['team']['displayName'], 'away_team': a['team']['displayName'],
                                'neutral_site': bool(comp.get('neutralSite', False)),
                                'home_h1': v(hl, 0), 'away_h1': v(al, 0),
                                'home_final': int(h['score']), 'away_final': int(a['score']),
                                'periods': min(len(hl), len(al))})
                except (TypeError, ValueError, KeyError):
                    continue
            return out
        except Exception:
            time.sleep(5)
    return None          # request failed (distinct from a date with no games)


def upload(rows, delete_where):
    import TrainModel as T
    conn = T.db_connect(); cur = conn.cursor()
    cur.execute(f"CREATE TABLE IF NOT EXISTS {TABLE} ({', '.join(f'{c} {t}' for c, t in COLUMNS)}) USING DELTA")
    cur.execute(f"DELETE FROM {TABLE} WHERE {delete_where}")
    q = lambda v: 'NULL' if v is None else (str(v).upper() if isinstance(v, bool) else
                                           (str(v) if isinstance(v, int) else "'" + str(v).replace("'", "''") + "'"))
    cols = [c for c, _ in COLUMNS]
    for i in range(0, len(rows), 300):
        vals = ', '.join('(' + ', '.join(q(r[c]) for c in cols) + ')' for r in rows[i:i + 300])
        cur.execute(f"INSERT INTO {TABLE} ({', '.join(cols)}) VALUES {vals}")
    cur.close(); conn.close()


def run(start, end, delete_where):
    rows, failed, d = [], [], start
    while d <= end:
        got = fetch_date(d)
        if got is None:
            failed.append(d.isoformat())
        else:
            rows.extend(got)
        d += timedelta(days=1)
        time.sleep(0.2)
    if failed:
        print(f"  ESPN failed for {len(failed)} dates: {failed[:10]}")
    ot = sum(1 for r in rows if r['periods'] > 2)
    print(f"  {start} to {end}: {len(rows)} games with first-half scores ({ot} went to overtime)")
    if rows:
        upload(rows, delete_where)
    return rows


def main():
    today = datetime.now(pytz.timezone('US/Eastern')).date()
    if '--recent' in sys.argv:
        n = int(sys.argv[sys.argv.index('--recent') + 1])
        start, end = today - timedelta(days=n), today - timedelta(days=1)
        run(start, end, f"game_date BETWEEN '{start}' AND '{end}'")
        return
    for s in [int(x) for x in sys.argv[1:]]:
        start, end = date(s - 1, 11, 1), min(date(s, 4, 15), today - timedelta(days=1))
        print(f"Season {s}:")
        run(start, end, f"season = {s}")


if __name__ == '__main__':
    main()
