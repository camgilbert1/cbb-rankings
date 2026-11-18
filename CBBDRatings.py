"""
Daily snapshot of CollegeBasketballData.com (CBBD) team ratings: SRS, Elo and adjusted efficiency.

CBBD only serves each rating as it stands today, so the only way to train/backtest on them later is to save
a copy every day. This script ONLY records; nothing reads the table into a model until a backtest earns it.

    python CBBDRatings.py            # today's snapshot for the current season

Table workspace.default.cbbd_ratings_daily:
    snapshot_date, season, team (ESPN display name, as in d1_teams.py), espn_id, srs, elo, adj_o, adj_d, adj_net
Teams are matched by ESPN id (CBBD's team sourceId) through d1_teams.ESPN_IDS; unmatched teams are skipped.
Uses 4 API calls per run (free tier: 1,000 per month).
"""

import os
from datetime import datetime

import pytz
import requests
from dotenv import load_dotenv

import d1_teams as D

load_dotenv()
API = 'https://api.collegebasketballdata.com'
TABLE = 'workspace.default.cbbd_ratings_daily'
SEASON = 2027


def get(path, **params):
    r = requests.get(API + path, params=params, timeout=60,
                     headers={'Authorization': 'Bearer ' + os.environ['CBBD_API_KEY']})
    r.raise_for_status()
    return r.json()


def snapshot(season=SEASON):
    espn_name = {v: k for k, v in D.ESPN_IDS.items()}
    id2espn = {t['id']: int(t['sourceId']) for t in get('/teams', season=season) if t.get('sourceId')}
    rows = {}

    def row(team_id):
        eid = id2espn.get(team_id)
        if eid not in espn_name:
            return None
        return rows.setdefault(eid, {'team': espn_name[eid], 'espn_id': eid})

    for x in get('/ratings/srs', season=season):
        if (r := row(x['teamId'])) is not None:
            r['srs'] = x['rating']
    for x in get('/ratings/elo', season=season):
        if (r := row(x['teamId'])) is not None:
            r['elo'] = x['elo']
    for x in get('/ratings/adjusted', season=season):
        if (r := row(x['teamId'])) is not None:
            r.update(adj_o=x['offensiveRating'], adj_d=x['defensiveRating'], adj_net=x['netRating'])
    return list(rows.values())


def upload(rows, snap_date, season=SEASON):
    import TrainModel as T
    conn = T.db_connect(); cur = conn.cursor()
    cur.execute(f"""CREATE TABLE IF NOT EXISTS {TABLE} (snapshot_date DATE, season INT, team STRING, espn_id INT,
                    srs DOUBLE, elo DOUBLE, adj_o DOUBLE, adj_d DOUBLE, adj_net DOUBLE) USING DELTA""")
    cur.execute(f"DELETE FROM {TABLE} WHERE snapshot_date = '{snap_date}'")      # rerun-safe
    num = lambda v: 'NULL' if v is None else repr(float(v))
    vals = ', '.join(
        f"('{snap_date}', {season}, '{r['team'].replace(chr(39), chr(39) * 2)}', {r['espn_id']}, "
        f"{num(r.get('srs'))}, {num(r.get('elo'))}, {num(r.get('adj_o'))}, {num(r.get('adj_d'))}, {num(r.get('adj_net'))})"
        for r in rows)
    cur.execute(f"INSERT INTO {TABLE} (snapshot_date, season, team, espn_id, srs, elo, adj_o, adj_d, adj_net) VALUES {vals}")
    cur.close(); conn.close()


def main():
    if not os.getenv('CBBD_API_KEY'):
        print("CBBD_API_KEY not set; skipping CBBD ratings snapshot")
        return
    snap_date = datetime.now(pytz.timezone('US/Eastern')).strftime('%Y-%m-%d')
    rows = snapshot()
    full = sum(1 for r in rows if all(r.get(k) is not None for k in ('srs', 'elo', 'adj_net')))
    print(f"CBBD ratings {snap_date}: {len(rows)} D1 teams matched ({full} with SRS, Elo and adjusted all present)")
    if len(rows) < 300:
        raise SystemExit(f"Only {len(rows)} teams matched; not saving (check CBBD / team mapping)")
    upload(rows, snap_date)
    print(f"  saved to {TABLE}")


if __name__ == '__main__':
    main()
