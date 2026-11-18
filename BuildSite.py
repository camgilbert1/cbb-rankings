"""
Build the data files for the static website (site/). Runs at the end of the daily workflow;
the workflow then publishes site/ to GitHub Pages. Read-only on Databricks.

Writes site/data/:
  meta.json     build time, slate date, model notes
  teams.json    D1 teams: ESPN id/logo, KenPom ratings, box-score ratings, record
  games.json    every 2027 game we predicted or that has a result: formula / v2 / v3 picks,
                Vegas line, final score and grading

Preview locally:  python BuildSite.py  then  python -m http.server -d site 8000
"""

import json
import os
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytz
import requests

import BoxRatings as B
import Betting
import TrainModel as T
from team_names import find_team_match

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'site', 'data')
SEASON_START = T.CURRENT_SEASON_START
LOGO = 'https://a.espncdn.com/i/teamlogos/ncaa/500/{}.png'
EASTERN = pytz.timezone('US/Eastern')
nk = T.name_key


def clean(v):
    """JSON-safe scalar."""
    if v is None:
        return None
    if isinstance(v, (np.floating, float)):
        return None if np.isnan(v) else round(float(v), 2)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (pd.Timestamp, datetime)):
        return v.isoformat()
    return v


def query(cur, sql):
    cur.execute(sql)
    return cur.fetchall_arrow().to_pandas()


def load_tables():
    conn = T.db_connect(); cur = conn.cursor()
    s = SEASON_START
    t = {
        'history': query(cur, f"SELECT * FROM workspace.default.prediction_history WHERE game_date >= '{s}'"),
        'outcomes': query(cur, f"SELECT * FROM workspace.default.prediction_outcomes WHERE game_date >= '{s}'"),
        'results': query(cur, f"SELECT * FROM workspace.default.game_results WHERE game_date >= '{s}'"),
        'ml': query(cur, f"""SELECT * FROM workspace.default.ml_predictions
                             WHERE game_date >= '{s}' AND model_family IN ('v2', 'v3')"""),
        'gold': query(cur, "SELECT * FROM workspace.default_gold.gold_team_rankings"),
    }
    try:     # first-half scores (BackfillHalves.py) for grading the 1H tab; the site builds without them
        t['halves'] = query(cur, f"SELECT game_date, home_team, away_team, home_h1, away_h1 FROM workspace.default.game_halves WHERE game_date >= '{s}'")
        t['halves']['game_date'] = pd.to_datetime(t['halves']['game_date']).dt.strftime('%Y-%m-%d')
    except Exception as e:
        print(f"  (first-half scores unavailable: {e})")
        t['halves'] = pd.DataFrame(columns=['game_date', 'home_team', 'away_team', 'home_h1', 'away_h1'])
    cur.close(); conn.close()
    for k in ('history', 'outcomes', 'results', 'ml'):
        t[k]['game_date'] = pd.to_datetime(t[k]['game_date']).dt.strftime('%Y-%m-%d')
    return t


def espn_slate(date):
    """ESPN scoreboard for one date: ids, start time, status, neutral site, records."""
    url = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard"
    try:
        resp = requests.get(url, params={'dates': date.replace('-', ''), 'limit': 400, 'groups': 50}, timeout=20)
        resp.raise_for_status()
    except Exception as e:
        print(f"  ESPN scoreboard unavailable ({e})")
        return []
    games = []
    for ev in resp.json().get('events', []):
        comp = ev['competitions'][0]
        side = {}
        for c in comp['competitors']:
            rec = next((r.get('summary') for r in c.get('records', []) if r.get('type') == 'total'), None)
            side[c['homeAway']] = {'id': int(c['team']['id']), 'name': c['team']['displayName'],
                                   'short': c['team'].get('shortDisplayName'), 'record': rec,
                                   'score': c.get('score')}
        if 'home' in side and 'away' in side:
            st = ev['status']['type']
            games.append({'date': date, 'time': ev.get('date'), 'state': st.get('state'), 'status': st.get('shortDetail'),
                          'neutral': bool(comp.get('neutralSite', False)), 'home': side['home'], 'away': side['away']})
    return games


def team_index():
    """ESPN names/ids from the box-score files: name_key(name) -> id, id -> name, id -> games, W-L."""
    paths = B.download_box([2026, 2027], current_season=T.CURRENT_SEASON, datasets=('games',))
    key2id, id2name, counts = {}, {}, defaultdict(int)
    neutral, finals, record = {}, {}, defaultdict(lambda: [0, 0])
    for s in (2026, 2027):
        g = pd.read_parquet(paths[('games', s)])
        for r in g.itertuples(index=False):
            for tid, name in ((r.home_team_id, r.home_team_name), (r.away_team_id, r.away_team_name)):
                key2id[nk(name)] = int(tid); id2name[int(tid)] = name
                if s == 2027:
                    counts[int(tid)] += 1
            if s == 2027:
                d = pd.Timestamp(r.game_date).strftime('%Y-%m-%d')
                neutral[(d, int(r.home_team_id), int(r.away_team_id))] = bool(r.neutral_site) if pd.notna(r.neutral_site) else False
                if bool(r.completed) and pd.notna(r.home_score):
                    finals[(d, int(r.home_team_id), int(r.away_team_id))] = (int(r.home_score), int(r.away_score))
                    hw = r.home_score > r.away_score
                    record[int(r.home_team_id)][0 if hw else 1] += 1
                    record[int(r.away_team_id)][1 if hw else 0] += 1
    return key2id, id2name, counts, neutral, finals, record


def build_teams(gold, key2id, id2name, counts, record, box):
    """D1 teams keyed by ESPN id. Each KenPom team goes to the ESPN school that matches it with the most games."""
    kp = gold.rename(columns={'adj_offensive_efficiency': 'adj_o', 'adj_defensive_efficiency': 'adj_d',
                              'adj_efficiency_margin': 'adj_em', 'adj_tempo': 'tempo'})
    best = {}
    for tid, name in id2name.items():
        m = find_team_match(name, kp)
        if m and counts.get(tid, 0) > best.get(m, (None, -1))[1]:
            best[m] = (tid, counts.get(tid, 0))
    kp = kp.assign(em_rank=kp['adj_em'].rank(ascending=False, method='min'),
                   o_rank=kp['adj_o'].rank(ascending=False, method='min'),
                   d_rank=kp['adj_d'].rank(ascending=True, method='min'),
                   t_rank=kp['tempo'].rank(ascending=False, method='min'))
    # Box ratings are solved against an average that includes non-D1 opponents; re-center each stat
    # on the D1 average for display (D1 offense avg = D1 defense avg, D1 average margin = 0, like KenPom).
    box_df = pd.DataFrame(box.get('teams', []))
    box_by_id, box_rank = {}, {}
    if not box_df.empty:
        d1_ids = {tid for tid, _ in best.values()}
        bd = box_df[box_df['id'].isin(d1_ids)].copy()
        for m in B.METRICS:
            mo, md = bd[f'o_{m}'].mean(), bd[f'd_{m}'].mean()
            c = (mo + md) / 2
            bd[f'o_{m}'] = (c + bd[f'o_{m}'] - mo).round(1)
            bd[f'd_{m}'] = (c + bd[f'd_{m}'] - md).round(1)
        bd['em'] = (bd['o_eff'] - bd['d_eff']).round(1)
        bd['prior_em'] = (bd['prior_em'] - bd['prior_em'].mean()).round(1)
        box_rank = dict(zip(bd['id'], bd['em'].rank(ascending=False, method='min')))
        box_by_id = {int(r['id']): r for r in bd.to_dict('records')}
    teams = {}
    for r in kp.itertuples(index=False):
        if r.team_name not in best:
            continue
        tid = best[r.team_name][0]
        w, l = record.get(tid, [0, 0])
        b = box_by_id.get(tid)
        teams[str(tid)] = {
            'id': tid, 'name': id2name[tid], 'kp_name': r.team_name, 'conf': r.conference, 'logo': LOGO.format(tid),
            'record': f'{w}-{l}',
            'kenpom': {'em': clean(r.adj_em), 'o': clean(r.adj_o), 'd': clean(r.adj_d), 'tempo': clean(r.tempo),
                       'rank': int(r.em_rank), 'o_rank': int(r.o_rank), 'd_rank': int(r.d_rank), 't_rank': int(r.t_rank)},
            'box': ({**{k: clean(v) for k, v in b.items() if k not in ('id', 'name')},
                     'rank': int(box_rank.get(tid, 0)) or None} if b else None),
        }
    return teams


def bet_of(r):
    """A model's bet on one game from its stored bet columns, or None."""
    flag = getattr(r, 'is_bet', None)
    if flag is None or pd.isna(flag) or not bool(flag):
        return None
    retro = getattr(r, 'bet_retro', None)
    return {'side': r.bet_side, 'label': r.bet_label, 'edge': clean(r.bet_edge), 'prob': clean(r.bet_prob),
            'units': clean(r.bet_units), 'retro': bool(retro) if retro is not None and not pd.isna(retro) else False}


def ou_bet_of(r):
    """A model's over/under bet on one game from its stored O/U columns, or None."""
    flag = getattr(r, 'ou_is_bet', None)
    if flag is None or pd.isna(flag) or not bool(flag):
        return None
    retro = getattr(r, 'ou_retro', None)
    return {'side': r.ou_side, 'label': r.ou_label, 'edge': clean(r.ou_edge), 'prob': clean(r.ou_prob),
            'units': clean(r.ou_units), 'retro': bool(retro) if retro is not None and not pd.isna(retro) else False}


def build_games(t, key2id, id2name, neutral, finals, slates):
    """One record per game, keyed by (date, home key, away key)."""
    games = {}

    def get(date, home, away):
        k = (date, nk(home), nk(away))
        if k not in games:
            hid, aid = key2id.get(nk(home)), key2id.get(nk(away))
            games[k] = {'date': date, 'time': None, 'state': None, 'status': None,
                        'home': {'id': hid, 'name': id2name.get(hid, home)}, 'away': {'id': aid, 'name': id2name.get(aid, away)},
                        'neutral': neutral.get((date, hid, aid), False), 'spread': None, 'total_line': None,
                        'formula': None, 'v2': None, 'v3': None, 'final': None, 'h1_final': None}
        return games[k]

    hist = t['history'].sort_values('prediction_date').drop_duplicates(['game_date', 'home_team', 'away_team'], keep='last')
    for r in hist.itertuples(index=False):
        if r.cover_pick == 'N/A' and not r.predicted_home_score:   # non-D1 opponent placeholder row
            continue
        g = get(r.game_date, r.home_team, r.away_team)
        g['time'] = r.game_time
        g['spread'] = clean(r.vegas_spread)
        g['total_line'] = clean(getattr(r, 'vegas_total', None))   # over/under (saved from Nov 18, 2026 on)
        g['formula'] = {'home': clean(r.predicted_home_score), 'away': clean(r.predicted_away_score),
                        'margin': clean(r.predicted_home_score - r.predicted_away_score),
                        'win_prob': clean(r.win_probability), 'confidence': r.confidence,
                        'cover_pick': r.cover_pick, 'cover_confidence': r.cover_confidence, 'ats': None,
                        'bet': bet_of(r), 'ou': None, 'ou_bet': ou_bet_of(r)}
    for r in t['outcomes'].itertuples(index=False):
        g = get(r.game_date, r.home_team, r.away_team)
        if g['formula'] is not None:
            g['formula']['ats'] = clean(r.ats_correct)
            g['formula']['ou'] = clean(getattr(r, 'ou_correct', None))
        if r.actual_home_score is not None and not pd.isna(r.actual_home_score):
            g['final'] = {'home': int(r.actual_home_score), 'away': int(r.actual_away_score)}
    for r in t['ml'].itertuples(index=False):
        g = get(r.game_date, r.home_team, r.away_team)
        g[r.model_family] = {'margin': clean(r.ml_predicted_margin), 'total': clean(r.ml_predicted_total),
                             'home': clean(r.ml_predicted_home_score), 'away': clean(r.ml_predicted_away_score),
                             'win_prob': clean(r.ml_win_probability), 'confidence': r.ats_confidence,
                             'ats': clean(r.ml_ats_correct), 'box': clean(r.box_margin), 'kp': clean(r.kp_margin),
                             'box_share': clean(r.box_share), 'bet': bet_of(r),
                             'ou': clean(getattr(r, 'ml_ou_correct', None)), 'ou_bet': ou_bet_of(r),
                             'box_total': clean(getattr(r, 'box_total', None)), 'kp_total': clean(getattr(r, 'kp_total', None)),
                             'h1_home': clean(getattr(r, 'h1_home_score', None)), 'h1_away': clean(getattr(r, 'h1_away_score', None))}
        if g['total_line'] is None:
            g['total_line'] = clean(getattr(r, 'vegas_total', None))
        if g['spread'] is None:
            g['spread'] = clean(r.vegas_spread)
    # ESPN can relabel home/away after we predict (neutral sites). Each lookup below tries the game as
    # stored first, then the two teams the other way round, and reads each team's score by name.
    for r in t['results'].itertuples(index=False):
        same, flipped = (r.game_date, nk(r.home_team), nk(r.away_team)), (r.game_date, nk(r.away_team), nk(r.home_team))
        if same in games:
            g = games[same]
            g['final'] = g['final'] or {'home': int(r.home_score), 'away': int(r.away_score)}
        elif flipped in games:
            g = games[flipped]
            g['final'] = g['final'] or {'home': int(r.away_score), 'away': int(r.home_score)}
    # Actual first-half scores (graded on the 1H tab), matched the same way: as stored, then flipped
    for r in t['halves'].itertuples(index=False):
        same, flipped = (r.game_date, nk(r.home_team), nk(r.away_team)), (r.game_date, nk(r.away_team), nk(r.home_team))
        if same in games:
            games[same]['h1_final'] = {'home': int(r.home_h1), 'away': int(r.away_h1)}
        elif flipped in games:
            games[flipped]['h1_final'] = {'home': int(r.away_h1), 'away': int(r.home_h1)}
    # ESPN slate data (ids, times, live status, records) for the dates being shown live
    for sg in slates:
        k = (sg['date'], nk(sg['home']['name']), nk(sg['away']['name']))
        sides = {'home': 'home', 'away': 'away'}            # our side -> ESPN's side
        if k not in games:
            k = (sg['date'], nk(sg['away']['name']), nk(sg['home']['name']))
            sides = {'home': 'away', 'away': 'home'}
            if k not in games:
                continue
        g = games[k]
        g.update({'time': sg['time'] or g['time'], 'state': sg['state'], 'status': sg['status'], 'neutral': sg['neutral']})
        for side in ('home', 'away'):
            src = sg[sides[side]]
            g[side].update({'id': src['id'], 'record': src['record'], 'short': src['short']})
        if sg['state'] == 'post' and g['final'] is None and sg['home']['score'] not in (None, ''):
            g['final'] = {'home': int(sg[sides['home']]['score']), 'away': int(sg[sides['away']]['score'])}
    for g in games.values():
        for side in ('home', 'away'):
            if g[side]['id'] is not None:
                g[side]['logo'] = LOGO.format(g[side]['id'])
        if g['final'] is None and g['home']['id'] is not None:
            f = finals.get((g['date'], g['home']['id'], g['away']['id']))
            if f:
                g['final'] = {'home': f[0], 'away': f[1]}
            else:
                f = finals.get((g['date'], g['away']['id'], g['home']['id']))   # relabelled home/away
                if f:
                    g['final'] = {'home': f[1], 'away': f[0]}
    return sorted(games.values(), key=lambda g: (g['date'], g['time'] or '', g['home']['name']))


def main():
    print("Building website data...")
    os.makedirs(OUT, exist_ok=True)
    now = datetime.now(EASTERN)
    today = now.strftime('%Y-%m-%d')

    t = load_tables()
    key2id, id2name, counts, neutral, finals, record = team_index()

    box_path = os.path.join(OUT, 'box_ratings.json')
    if os.path.exists(box_path) and (now.timestamp() - os.path.getmtime(box_path)) < 12 * 3600:
        box = json.load(open(box_path))
    else:
        print("  box ratings file missing/stale — building box ratings")
        _, bstate = B.build_asof([2023, 2024, 2025, 2026, 2027], current_season=T.CURRENT_SEASON, verbose=False)
        box = {'through': None, 'teams': B.export_team_ratings(bstate)}

    teams = build_teams(t['gold'], key2id, id2name, counts, record, box)
    yesterday = (now - timedelta(days=1)).strftime('%Y-%m-%d')
    slates = espn_slate(today) + espn_slate(yesterday)
    games = build_games(t, key2id, id2name, neutral, finals, slates)

    v3_dates = sorted(set(t['ml'].loc[t['ml']['model_family'] == 'v3', 'game_date']))
    meta = {'generated_at': now.isoformat(), 'today': today, 'season': T.CURRENT_SEASON,
            'box_through': box.get('through'), 'trust': box.get('trust'),
            'breakeven': 52.4, 'n_games': len(games), 'n_teams': len(teams),
            'bet_threshold': Betting.BET_THRESHOLD, 'bet_units': Betting.BET_UNITS,
            'v3_dates': v3_dates}
    for name, obj in (('meta', meta), ('teams', teams), ('games', games)):
        with open(os.path.join(OUT, f'{name}.json'), 'w') as fh:
            json.dump(obj, fh, separators=(',', ':'), default=clean)
    print(f"  wrote {len(teams)} teams, {len(games)} games "
          f"({sum(1 for g in games if g['date'] == today)} today) to {OUT}")
    if os.getenv('GITHUB_ACTIONS'):
        bust_cache(now.strftime('%Y%m%d%H%M%S'))


def bust_cache(version):
    """
    GitHub Pages lets browsers keep files for 10 minutes, so a deploy's new app.js / styles.css could be
    ignored on refresh. Tag them with this build's version in index.html (app.js?v=...) so every deploy
    is fetched fresh. Runs only in GitHub Actions, so the repo's index.html is never changed locally.
    """
    import re
    path = os.path.join(os.path.dirname(OUT), 'index.html')
    html = open(path, encoding='utf-8').read()
    html = re.sub(r'(href="styles\.css)(\?v=[^"]*)?"', rf'\1?v={version}"', html)
    html = re.sub(r'(src="app\.js)(\?v=[^"]*)?"', rf'\1?v={version}"', html)
    open(path, 'w', encoding='utf-8', newline='\n').write(html)
    print(f"  cache-busted index.html assets (v={version})")


if __name__ == '__main__':
    main()
