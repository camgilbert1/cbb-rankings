"""
Kalshi first-half lines for the 1H tab (display + grading only; nothing bets on them).

Kalshi lists men's college basketball 1st-half markets as ladders of yes/no contracts:
    1H total:   "1st Half: Over 67.5 points scored"          (one ladder per game)
    1H spread:  "Michigan wins the 1H by over 7.5 points"     (a ladder per team)
Each contract's price (cents) is the market's probability. The implied line is where the price crosses 50c,
interpolated between strikes, using the mid of the current bid/ask (two-sided quotes within 20c only).
Spread lines are stored as the market's expected HOME 1H margin (home minus away).

Market data is public: no account or key. Games are matched to today's v3 predictions (ml_predictions) by the
date in Kalshi's event code plus both team names (Kalshi's own game-winner markets give the names).

    python KalshiLines.py       # snapshot for the dates that have v3 predictions today/tomorrow

Table workspace.default.kalshi_1h_lines: one row per game, replaced on every run (latest snapshot wins):
    game_date, home_team, away_team, kalshi_code, h1_margin_line, h1_total_line, n_spread, n_total, captured_at
"""

import re
import time
from datetime import datetime, timezone

import pandas as pd
import requests

import d1_teams as D

API = "https://api.elections.kalshi.com/trade-api/v2"
TABLE = 'workspace.default.kalshi_1h_lines'
SERIES_SPREAD, SERIES_TOTAL, SERIES_GAME = 'KXNCAAMB1HSPREAD', 'KXNCAAMB1HTOTAL', 'KXNCAAMBGAME'
MAX_QUOTE_WIDTH = 0.20
MONTHS = {m: i for i, m in enumerate(['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'], 1)}
# Kalshi names that don't reduce to an ESPN/KenPom name on their own
ALIASES = {'university at albany': 'albany', 'umass': 'massachusetts', 'southern university': 'southern',
           'fdu': 'fairleigh dickinson', 'st francis pa': 'st francis', 'north carolina st': 'nc st',
           'uconn': 'connecticut', 'ole miss': 'mississippi'}


def markets(series):
    out, cursor = [], None
    for _ in range(20):
        params = {'series_ticker': series, 'status': 'open', 'limit': 1000}
        if cursor:
            params['cursor'] = cursor
        for attempt in range(4):
            r = requests.get(f"{API}/markets", params=params, timeout=30)
            if r.status_code == 200:
                break
            time.sleep(2 + 3 * attempt)
        r.raise_for_status()
        js = r.json()
        out += js.get('markets', [])
        cursor = js.get('cursor')
        if not cursor or not js.get('markets'):
            break
    return out


def mid(m):
    """Mid of a two-sided quote, or None."""
    try:
        bid, ask = float(m.get('yes_bid_dollars')), float(m.get('yes_ask_dollars'))
    except (TypeError, ValueError):
        return None
    return (bid + ask) / 2 if 0 < bid < ask < 1 and ask - bid <= MAX_QUOTE_WIDTH else None


def implied(points):
    """points: [(strike, P(value > strike))]. The strike where P crosses 0.5, interpolated; None if no crossing."""
    pts = sorted(points)
    for (s0, p0), (s1, p1) in zip(pts, pts[1:]):
        if p0 >= 0.5 >= p1 and p0 != p1:
            return s0 + (p0 - 0.5) / (p0 - p1) * (s1 - s0)
    return None


def norm(s):
    s = (s or '').lower().replace('&', ' and ')
    for ch in ".'-()":
        s = s.replace(ch, ' ' if ch != "'" else '')
    s = re.sub(r'\bst\b', 'state', s)
    s = re.sub(r'\bsaint\b', 'st', s)
    s = re.sub(r'\bstate\b', 'st', s)
    s = ' '.join(s.split())
    return ALIASES.get(s, s)


def candidates(espn, kenpom_names):
    out = {norm(espn), norm(kenpom_names.get(espn, ''))}
    toks = espn.split()
    for k in range(1, len(toks)):
        out.add(norm(' '.join(toks[:k])))            # location prefixes: 'Michigan', 'West Virginia'
    return {x for x in out if x}


def code_date(code):
    try:
        return pd.Timestamp(2000 + int(code[:2]), MONTHS[code[2:5]], int(code[5:7]))
    except (KeyError, ValueError):
        return None


def build(games, season):
    """games: DataFrame game_date, home_team, away_team (ESPN names). Returns one row per matched Kalshi game."""
    kp = D.D1_TEAMS.get(season, {})
    games = games.copy()
    games['game_date'] = pd.to_datetime(games['game_date'])
    games['ch'] = games['home_team'].map(lambda n: candidates(n, kp))
    games['ca'] = games['away_team'].map(lambda n: candidates(n, kp))
    names = {}
    for m in markets(SERIES_GAME):
        sub = m.get('yes_sub_title') or ''
        if sub and sub.lower() != 'tie':
            names.setdefault(m['event_ticker'].split('-', 1)[1], set()).add(sub)
    spread, total = markets(SERIES_SPREAD), markets(SERIES_TOTAL)
    by_code = {}
    for m in spread + total:
        by_code.setdefault(m['event_ticker'].split('-', 1)[1], []).append(m)
    for code, ms in by_code.items():          # spread titles also name teams
        for m in ms:
            if m['event_ticker'].startswith(SERIES_SPREAD):
                names.setdefault(code, set()).add(str(m.get('yes_sub_title', '')).split(' wins')[0])
    rows, unmatched = [], []
    for code, ms in by_code.items():
        d0, nm = code_date(code), [n for n in names.get(code, ()) if n]
        if d0 is None or len(nm) != 2:
            unmatched.append(code); continue
        a, b = norm(nm[0]), norm(nm[1])
        hit = None
        for d in (d0, d0 - pd.Timedelta(days=1), d0 + pd.Timedelta(days=1)):
            for r in games[games.game_date == d].itertuples():
                if (a in r.ch and b in r.ca) or (a in r.ca and b in r.ch):
                    hit = r; break
            if hit is not None:
                break
        if hit is None:
            unmatched.append(code); continue
        k_home = nm[0] if a in hit.ch else nm[1]
        k_away = nm[1] if k_home == nm[0] else nm[0]
        sp_pts, to_pts = [], []
        for m in ms:
            p, strike = mid(m), m.get('floor_strike')
            if p is None or strike is None:
                continue
            if m['event_ticker'].startswith(SERIES_TOTAL):
                to_pts.append((float(strike), p))
            else:
                team = str(m.get('yes_sub_title', '')).split(' wins')[0]
                if team == k_home:
                    sp_pts.append((float(strike), p))                 # P(home margin > k)
                elif team == k_away:
                    sp_pts.append((-float(strike), 1 - p))            # P(away margin > k) -> P(home margin > -k)
        m_line, t_line = implied(sp_pts), implied(to_pts)
        if m_line is None and t_line is None:
            continue
        rows.append({'game_date': hit.game_date.date(), 'home_team': hit.home_team, 'away_team': hit.away_team,
                     'kalshi_code': code, 'h1_margin_line': None if m_line is None else round(m_line, 2),
                     'h1_total_line': None if t_line is None else round(t_line, 2),
                     'n_spread': len(sp_pts), 'n_total': len(to_pts)})
    return pd.DataFrame(rows), unmatched


ESPN_SB = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard"


def not_started(games):
    """Keep only games ESPN still shows as pre-game, so a later run never overwrites a pre-tip line with
    in-game prices (the earlier snapshot for a started game stays in the table)."""
    keep = set()
    for d in sorted(set(pd.to_datetime(games['game_date']).dt.strftime('%Y%m%d'))):
        try:
            js = requests.get(ESPN_SB, params={'dates': d, 'limit': 400, 'groups': 50}, timeout=30).json()
        except Exception:
            return games                       # ESPN unavailable: don't filter
        for e in js.get('events', []):
            if e['status']['type']['state'] != 'pre':
                continue
            side = {x['homeAway']: x['team']['displayName'] for x in e['competitions'][0]['competitors']}
            keep.add((side.get('home'), side.get('away'))); keep.add((side.get('away'), side.get('home')))
    return games[[(h, a) in keep for h, a in zip(games['home_team'], games['away_team'])]]


def main():
    import TrainModel as T
    conn = T.db_connect(); cur = conn.cursor()
    cur.execute("""SELECT DISTINCT game_date, home_team, away_team FROM workspace.default.ml_predictions
                   WHERE model_family = 'v3' AND game_date >= date_sub(current_date(), 1)""")
    games = pd.DataFrame(cur.fetchall(), columns=['game_date', 'home_team', 'away_team'])
    if games.empty:
        print("No v3 games today; nothing to match."); return
    n_all = len(games)
    games = not_started(games)
    print(f"{len(games)} of {n_all} v3 games not started yet")
    if games.empty:
        return
    df, unmatched = build(games, T.CURRENT_SEASON)
    print(f"Kalshi 1H lines: {len(df)} games matched "
          f"({df['h1_margin_line'].notna().sum() if len(df) else 0} spreads, {df['h1_total_line'].notna().sum() if len(df) else 0} totals); "
          f"{len(unmatched)} Kalshi games not matched{': ' + ', '.join(unmatched[:10]) if unmatched else ''}")
    cur.execute(f"""CREATE TABLE IF NOT EXISTS {TABLE} (game_date DATE, home_team STRING, away_team STRING,
                    kalshi_code STRING, h1_margin_line DOUBLE, h1_total_line DOUBLE, n_spread INT, n_total INT,
                    captured_at TIMESTAMP) USING DELTA""")
    if len(df):
        now = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
        q = lambda v: 'NULL' if v is None or (isinstance(v, float) and pd.isna(v)) else (
            repr(float(v)) if isinstance(v, float) else str(v) if isinstance(v, int) else "'" + str(v).replace("'", "''") + "'")
        codes = ', '.join(q(c) for c in df['kalshi_code'])
        cur.execute(f"DELETE FROM {TABLE} WHERE kalshi_code IN ({codes})")
        vals = ', '.join(f"({q(str(r.game_date))}, {q(r.home_team)}, {q(r.away_team)}, {q(r.kalshi_code)}, "
                         f"{q(r.h1_margin_line)}, {q(r.h1_total_line)}, {int(r.n_spread)}, {int(r.n_total)}, '{now}')"
                         for r in df.itertuples())
        cur.execute(f"INSERT INTO {TABLE} VALUES {vals}")
        for r in df.itertuples():
            print(f"  {r.away_team} @ {r.home_team}: 1H home margin {r.h1_margin_line}  total {r.h1_total_line}")
    cur.close(); conn.close()


if __name__ == '__main__':
    main()
