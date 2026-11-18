"""
Day-of KenPom ratings for a past season, from kenpom.com/archive.php?d=YYYY-MM-DD (subscriber page).
Used for honest backtests: predict each game with the ratings KenPom showed that morning, exactly as the
live pipeline does, instead of end-of-season ratings that already "know" how the season went.

    python BackfillKenPomDaily.py 2026          # 2025-26 season, every date with a game
    python BackfillKenPomDaily.py 2026 --local  # CSV only

Saved to workspace.default.kenpom_daily: season, rating_date, team_name, adj_em, adj_o, adj_d, tempo.
Team names are KenPom's (the same names d1_teams.py maps ESPN schools to).
"""

import os
import re
import sys
import time

import pandas as pd
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from kenpompy.utils import get_html, login

load_dotenv()
TABLE = 'workspace.default.kenpom_daily'
SEED_SUFFIX = re.compile(r'\d+[A-Za-z0-9]*$')   # tournament tags: "Michigan1CH", "Iowa St.2S16", runner-up "Purdue12nd"


def parse_archive(html, rating_date, season):
    txt = html.decode('utf-8', 'replace') if isinstance(html, (bytes, bytearray)) else str(html)
    tbl = BeautifulSoup(txt, 'html.parser').find('table', id='ratings-table')
    if tbl is None:
        return []
    out = []
    for tr in tbl.find('tbody').find_all('tr'):
        td = [x.get_text(strip=True) for x in tr.find_all('td')]
        if len(td) < 9 or not td[0].isdigit():
            continue
        # columns: Rk, Team, Conf, NetRtg (AdjEM on that date), ORtg, ORk, DRtg, DRk, AdjT, ... final-season columns
        out.append({'season': season, 'rating_date': rating_date, 'team_name': SEED_SUFFIX.sub('', td[1]).strip(),
                    'adj_em': float(td[3]), 'adj_o': float(td[4]), 'adj_d': float(td[6]), 'tempo': float(td[8])})
    return out


def upload(df):
    import TrainModel as T
    conn = T.db_connect(); cur = conn.cursor()
    cur.execute(f"""CREATE TABLE IF NOT EXISTS {TABLE} (season INT, rating_date DATE, team_name STRING,
                    adj_em DOUBLE, adj_o DOUBLE, adj_d DOUBLE, tempo DOUBLE) USING DELTA""")
    for s in sorted(df['season'].unique()):
        cur.execute(f"DELETE FROM {TABLE} WHERE season = {int(s)}")
    recs = df.to_dict('records')
    q = lambda v: repr(float(v)) if isinstance(v, float) else (str(v) if isinstance(v, int) else "'" + str(v).replace("'", "''") + "'")
    for i in range(0, len(recs), 1000):
        vals = ', '.join(f"({r['season']}, '{r['rating_date']}', {q(r['team_name'])}, {r['adj_em']}, {r['adj_o']}, {r['adj_d']}, {r['tempo']})"
                         for r in recs[i:i + 1000])
        cur.execute(f"INSERT INTO {TABLE} VALUES {vals}")
    cur.close(); conn.close()


def main():
    season = int(sys.argv[1]) if len(sys.argv) > 1 else 2026
    lines = f"historical_lines_{season}.csv"
    if os.path.exists(lines):
        dates = sorted(pd.read_csv(lines)['game_date'].unique())          # only dates with games
    else:
        dates = [d.strftime('%Y-%m-%d') for d in pd.date_range(f"{season - 1}-11-01", f"{season}-04-10")]
    out_csv = f"kenpom_daily_{season}.csv"
    done = pd.read_csv(out_csv) if os.path.exists(out_csv) else pd.DataFrame()
    have = set(done['rating_date']) if not done.empty else set()
    todo = [d for d in dates if d not in have]
    print(f"  {len(have)} dates already saved, {len(todo)} to fetch", flush=True)

    def safe_login():
        for wait in (0, 15, 45, 120):            # KenPom sometimes resets the connection; back off and retry
            time.sleep(wait)
            try:
                return login(os.getenv('KENPOM_EMAIL'), os.getenv('KENPOM_PASSWORD'))
            except Exception as e:
                print(f"    login failed ({type(e).__name__}); retrying", flush=True)
        raise SystemExit("KenPom login kept failing; rerun later to resume from the saved dates")

    br = safe_login()
    missing = []
    for i, d in enumerate(todo):
        r = []
        for attempt in range(3):
            try:
                r = parse_archive(get_html(br, f"https://kenpom.com/archive.php?d={d}"), d, season)
                if r:
                    break
            except Exception:
                pass
            time.sleep(10 * (attempt + 1))
            br = safe_login()
        if r:
            pd.DataFrame(r).to_csv(out_csv, mode='a', header=not os.path.exists(out_csv), index=False)   # resume point
        else:
            missing.append(d)
        if i % 25 == 0:
            print(f"  {d}: {len(r)} teams ({i + 1}/{len(todo)})", flush=True)
        time.sleep(2)
    df = pd.read_csv(out_csv)
    print(f"{season}: {df['rating_date'].nunique()} dates, {len(df)} team-days | dates with no ratings: {missing}")
    if '--local' not in sys.argv:
        upload(df)
        print(f"  saved to {TABLE}")


if __name__ == '__main__':
    main()
