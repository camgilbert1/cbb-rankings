"""
First-half lines from SportsGameOdds (SGO) for the 1H tab: fills the games (and the parts: spread / total) that
Kalshi (KalshiLines.py, run first) did not cover. Display + grading only; nothing bets on them.

Line used per game: DraftKings when it has the market, else SGO's book consensus ("bookSpread"/"bookOverUnder").
Spread is stored as the expected HOME 1H margin (DraftKings home -4.5 -> 4.5).

Budget (free Amateur tier: 2,500 events per month, charged per event returned, not per market):
  * games that already have both 1H lines (Kalshi or SGO) are never requested again;
  * after the first run of a day, only the SGO events still missing a line are requested (by eventID);
  * no request when this month's usage is above MONTHLY_STOP.
Games that had already tipped when captured are flagged after_tip (in-game lines): shown, not graded.

    python SGOLines.py          (needs SGO_API_KEY)

Table workspace.default.sgo_1h_lines, one row per SGO event seen (lines may be NULL when no book has them):
    game_date, home_team, away_team, sgo_event_id, src_spread, h1_margin_line, src_total, h1_total_line,
    after_tip, captured_at
"""

import os
from datetime import datetime, timezone

import pandas as pd
import pytz
import requests
from dotenv import load_dotenv

import d1_teams as D
from KalshiLines import candidates, norm

load_dotenv()
API = 'https://api.sportsgameodds.com/v2'
TABLE = 'workspace.default.sgo_1h_lines'
KALSHI_TABLE = 'workspace.default.kalshi_1h_lines'
SPREAD_ID, TOTAL_ID = 'points-home-1h-sp-home', 'points-all-1h-ou-over'
BOOK = 'draftkings'
MONTHLY_LIMIT, MONTHLY_STOP = 2500, 2300
ET = pytz.timezone('America/New_York')


def headers():
    return {'x-api-key': os.environ['SGO_API_KEY']}


def month_usage():
    d = requests.get(f"{API}/account/usage", headers=headers(), timeout=30).json()['data']
    return int(d['rateLimits']['per-month'].get('current-entities') or 0)


def fetch(params):
    out, cursor = [], None
    for _ in range(10):
        p = dict(params, limit=100, oddID=f"{SPREAD_ID},{TOTAL_ID}", includeOpposingOdds='false')
        if cursor:
            p['cursor'] = cursor
        js = requests.get(f"{API}/events", headers=headers(), params=p, timeout=60).json()
        out += js.get('data', [])
        cursor = js.get('nextCursor')
        if not cursor:
            break
    return out


def line(odd, field):
    """(source, value) from DraftKings, else the book consensus; (None, None) if neither."""
    if not odd:
        return None, None
    dk = (odd.get('byBookmaker') or {}).get(BOOK) or {}
    if dk.get('available') and dk.get(field) not in (None, ''):
        return 'DraftKings', float(dk[field])
    cons = odd.get('bookSpread' if field == 'spread' else 'bookOverUnder')
    return ('consensus', float(cons)) if cons not in (None, '') else (None, None)


def team_names(t):
    n = t.get('names') or {}
    return [x for x in (n.get('long'), n.get('medium'), n.get('short')) if x]


def match(events, games, season):
    """SGO events -> rows keyed to our games (date in ET + both team names)."""
    kp = D.D1_TEAMS.get(season, {})
    games = games.copy()
    games['ch'] = games['home_team'].map(lambda n: candidates(n, kp))
    games['ca'] = games['away_team'].map(lambda n: candidates(n, kp))
    rows = []
    for e in events:
        st = e.get('status') or {}
        if st.get('ended') or not st.get('startsAt'):
            continue
        d = pd.Timestamp(st['startsAt']).tz_convert(ET).date()
        hn = {norm(x) for x in team_names(e['teams']['home'])}
        an = {norm(x) for x in team_names(e['teams']['away'])}
        hit, flipped = None, False
        for r in games[games['game_date'] == d].itertuples():
            if hn & r.ch and an & r.ca:
                hit = r; break
            if hn & r.ca and an & r.ch:
                hit, flipped = r, True; break
        if hit is None:
            continue
        odds = e.get('odds') or {}
        ss, sv = line(odds.get(SPREAD_ID), 'spread')
        ts, tv = line(odds.get(TOTAL_ID), 'overUnder')
        margin = None if sv is None else -sv                 # SGO home spread -> expected SGO-home margin
        if margin is not None and flipped:
            margin = -margin
        rows.append({'game_date': d, 'home_team': hit.home_team, 'away_team': hit.away_team,
                     'sgo_event_id': e['eventID'], 'src_spread': ss, 'h1_margin_line': margin,
                     'src_total': ts, 'h1_total_line': tv, 'after_tip': bool(st.get('started'))})
    return pd.DataFrame(rows)


def main():
    if not os.environ.get('SGO_API_KEY'):
        print("SGO_API_KEY not set; skipping."); return
    import TrainModel as T
    conn = T.db_connect(); cur = conn.cursor()
    try:
        run(cur, T)
    finally:
        cur.close(); conn.close()


def run(cur, T):
    cur.execute(f"""CREATE TABLE IF NOT EXISTS {TABLE} (game_date DATE, home_team STRING, away_team STRING,
                    sgo_event_id STRING, src_spread STRING, h1_margin_line DOUBLE, src_total STRING, h1_total_line DOUBLE,
                    after_tip BOOLEAN, captured_at TIMESTAMP) USING DELTA""")
    today = datetime.now(ET).date()
    cur.execute(f"""SELECT DISTINCT game_date, home_team, away_team FROM workspace.default.ml_predictions
                    WHERE model_family = 'v3' AND game_date = '{today}'""")
    games = pd.DataFrame(cur.fetchall(), columns=['game_date', 'home_team', 'away_team'])
    if games.empty:
        print(f"No v3 games for {today}; nothing to do."); return

    # what's already covered (either source), per game and part
    have = {}
    for tbl in (KALSHI_TABLE, TABLE):
        try:
            cur.execute(f"SELECT home_team, away_team, h1_margin_line, h1_total_line FROM {tbl} WHERE game_date = '{today}'")
            for h, a, m, t in cur.fetchall():
                k = have.setdefault((h, a), [False, False])
                k[0] |= m is not None; k[1] |= t is not None
        except Exception:
            pass
    missing = [(h, a) for h, a in zip(games.home_team, games.away_team) if not all(have.get((h, a), [False, False]))]
    print(f"{len(games)} v3 games today; {len(games) - len(missing)} already have both 1H lines; {len(missing)} missing")
    if not missing:
        return

    used = month_usage()
    print(f"SGO usage this month: {used} of {MONTHLY_LIMIT} events")
    if used >= MONTHLY_STOP:
        print(f"::warning::SGO monthly budget nearly used ({used}); skipping to stay on the free tier"); return

    # known SGO events for today's still-missing games -> request just those; first run of the day -> whole slate
    cur.execute(f"SELECT home_team, away_team, sgo_event_id FROM {TABLE} WHERE game_date = '{today}'")
    known = {(h, a): ev for h, a, ev in cur.fetchall()}
    if known:
        ids = sorted({known[g] for g in missing if g in known})
        if not ids:
            print("Missing games aren't listed on SGO; nothing to request."); return
        events = fetch({'eventIDs': ','.join(ids)})
    else:
        events = fetch({'leagueID': 'NCAAB', 'oddsAvailable': 'true'})
    print(f"SGO returned {len(events)} events")
    df = match(events, games, T.CURRENT_SEASON)
    if df.empty:
        print("No SGO events matched today's games."); return

    now = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
    q = lambda v: 'NULL' if v is None or (isinstance(v, float) and pd.isna(v)) else (
        ('TRUE' if v else 'FALSE') if isinstance(v, bool) else repr(float(v)) if isinstance(v, float)
        else "'" + str(v).replace("'", "''") + "'")
    ids = ', '.join(q(x) for x in df['sgo_event_id'])
    cur.execute(f"DELETE FROM {TABLE} WHERE sgo_event_id IN ({ids})")
    vals = ', '.join(f"({q(str(r.game_date))}, {q(r.home_team)}, {q(r.away_team)}, {q(r.sgo_event_id)}, {q(r.src_spread)}, "
                     f"{q(r.h1_margin_line)}, {q(r.src_total)}, {q(r.h1_total_line)}, {q(bool(r.after_tip))}, '{now}')"
                     for r in df.itertuples())
    cur.execute(f"INSERT INTO {TABLE} VALUES {vals}")
    n_sp, n_to = df['h1_margin_line'].notna().sum(), df['h1_total_line'].notna().sum()
    print(f"Saved {len(df)} SGO games ({n_sp} spreads, {n_to} totals; {int(df['after_tip'].sum())} captured after tip)")


if __name__ == '__main__':
    main()
