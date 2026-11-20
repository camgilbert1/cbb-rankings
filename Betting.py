"""
Bet selection shared by every model (Formula in GamePredictions.py, v2/v3 in TrainModel.py).

Each model predicts every game; it BETS a game only when it is confident enough against the line:

    edge (home side, points)  = model margin (home - away) + home spread
    scatter (points)          = a + b * |home spread|   (how far results land from the line)
    P(home covers)            = Phi(edge / scatter)
    bet the side whose cover probability >= BET_THRESHOLD; otherwise no bet

Scatter depends on games and lines, not on the model, so one daily fit serves all models. It is fit
by maximum likelihood on graded games whose line was taken BEFORE tip-off (21 Nov 11 2026 games were
predicted after tip with live in-game lines), over the last SCATTER_WINDOW_DAYS days, else the season
to date, else the default. Bet records count only games where is_bet is true; every game is still
predicted, stored and graded for training and accuracy checks.

Stage 2 (after ~150 graded bets per model): replace the fixed threshold with a track-record-adjusted
cover probability vs the price's break-even (52.4% at -110).
"""

import numpy as np
import pandas as pd
from statistics import NormalDist

BET_THRESHOLD = 0.60
BET_UNITS = 1.0        # every bet is 1 unit for now; sizing by confidence comes later

# Columns added to prediction_history (Formula), prediction_outcomes and ml_predictions (v2/v3)
BET_COLUMNS = [('is_bet', 'BOOLEAN'), ('bet_side', 'STRING'), ('bet_label', 'STRING'),
               ('bet_edge', 'DOUBLE'), ('bet_prob', 'DOUBLE'), ('bet_units', 'DOUBLE'),
               ('bet_retro', 'BOOLEAN')]   # True = applied after the fact to stored pre-game numbers

# Over/under bets: the same rule on totals. P(over) = Phi((predicted total - O/U line) / ou_scatter).
OU_COLUMNS = [('ou_is_bet', 'BOOLEAN'), ('ou_side', 'STRING'), ('ou_label', 'STRING'),
              ('ou_edge', 'DOUBLE'), ('ou_prob', 'DOUBLE'), ('ou_units', 'DOUBLE'),
              ('ou_retro', 'BOOLEAN')]
OU_SCATTER_DEFAULT = 16.0               # points; used until OU_SCATTER_MIN_GAMES graded totals exist
OU_SCATTER_RANGE = (10.0, 24.0)         # safe range for the fit
OU_SCATTER_MIN_GAMES = 150

SCATTER_DEFAULT = (11.1, 0.06)          # fit on Nov 2026 games: a + b * |line|
SCATTER_A_RANGE = (8.0, 16.0)           # safe ranges for the fit (also the search grid)
SCATTER_B_RANGE = (0.0, 0.25)
SCATTER_WINDOW_DAYS = 30
SCATTER_MIN_GAMES = 150

_phi = NormalDist().cdf


def load_graded_lines(cursor, season_start):
    """
    Graded games with a pre-game line, for fit_scatter: game_date, home_spread, actual_margin.
    Excludes 0-0 "finals" (postponed games) and lines captured after tip-off.
    """
    cursor.execute(f"""
        SELECT o.game_date,
               o.vegas_spread AS home_spread,
               o.actual_home_score - o.actual_away_score AS actual_margin,
               h.prediction_date, h.game_time
        FROM workspace.default.prediction_outcomes o
        JOIN workspace.default.prediction_history h
          ON h.game_date = o.game_date AND h.home_team = o.home_team AND h.away_team = o.away_team
        WHERE o.game_date >= '{season_start}'
          AND o.vegas_spread IS NOT NULL
          AND NOT (o.actual_home_score = 0 AND o.actual_away_score = 0)
    """)
    df = cursor.fetchall_arrow().to_pandas()
    df = df[predicted_before_tip(df['prediction_date'], df['game_time'])]
    df['game_date'] = pd.to_datetime(df['game_date'])
    return df[['game_date', 'home_spread', 'actual_margin']].reset_index(drop=True)


def predicted_before_tip(prediction_date, game_time):
    """
    Boolean Series: the prediction (and its line) was made before tip-off. prediction_date is a UTC
    timestamp; game_time is ESPN's ISO tip time ('2026-11-17T01:30Z'). Unknown tip time counts as before.
    """
    made = pd.to_datetime(prediction_date, utc=True)
    tip = pd.to_datetime(game_time, utc=True, errors='coerce', format='ISO8601')
    return (tip.isna() | (made <= tip)).to_numpy()


def fit_scatter(graded, day):
    """
    Scatter parameters (a, b) for games on `day`, fit on games graded before it.
    graded: DataFrame with game_date (datetime64), home_spread, actual_margin.
    Returns ((a, b), description).
    """
    day = pd.Timestamp(day)
    hist = graded[graded['game_date'] < day]
    recent = hist[hist['game_date'] >= day - pd.Timedelta(days=SCATTER_WINDOW_DAYS)]
    if len(recent) >= SCATTER_MIN_GAMES:
        sample, source = recent, f"{len(recent)} games, last {SCATTER_WINDOW_DAYS} days"
    elif len(hist) >= SCATTER_MIN_GAMES:
        sample, source = hist, f"{len(hist)} games, season to date"
    else:
        return SCATTER_DEFAULT, f"default ({len(hist)} graded games < {SCATTER_MIN_GAMES})"

    miss = (sample['actual_margin'] + sample['home_spread']).to_numpy(dtype=float)
    line = sample['home_spread'].abs().to_numpy(dtype=float)
    a_grid = np.arange(SCATTER_A_RANGE[0], SCATTER_A_RANGE[1] + 1e-9, 0.1)
    b_grid = np.arange(SCATTER_B_RANGE[0], SCATTER_B_RANGE[1] + 1e-9, 0.005)
    sd = a_grid[:, None, None] + b_grid[None, :, None] * line[None, None, :]       # (a, b, game)
    loglik = -(np.log(sd) + 0.5 * (miss[None, None, :] / sd) ** 2).sum(axis=2)
    i, j = np.unravel_index(np.argmax(loglik), loglik.shape)
    a, b = round(float(a_grid[i]), 1), round(float(b_grid[j]), 3)
    edge_note = ''
    if i in (0, len(a_grid) - 1) or j in (0, len(b_grid) - 1):
        edge_note = ', at the edge of the safe range'
    return (a, b), f"{source}{edge_note}"


def scatter_for(home_spread, params):
    a, b = params
    return a + b * abs(home_spread)


# Big lines (v3 only, Nov 20 2026): a line of BIG_LINE+ points needs BIG_LINE_THRESHOLD to bet. Backtest on
# 30+ lines: at 60% 18-15 (+1.4u) 2025-26 and 8-12 (-4.7u) 2024-25; at 65% 13-6 (+5.8u) and 3-3 (-0.3u).
BIG_LINE = 30.0
BIG_LINE_THRESHOLD = 0.65


def make_bet(home_team, away_team, model_margin, home_spread, params, threshold=BET_THRESHOLD, big_line_rule=False):
    """
    The model's bet on one game, or None.

    model_margin: predicted home - away. home_spread: Vegas home line (None = no line, 0 = pick 'em).
    Returns {'side': 'home'|'away', 'label', 'edge' (points toward the bet side), 'prob'} or None.
    """
    if home_spread is None or model_margin is None or pd.isna(home_spread) or pd.isna(model_margin):
        return None
    home_spread = float(home_spread) + 0.0
    edge_home = float(model_margin) + home_spread
    if big_line_rule and abs(home_spread) >= BIG_LINE:
        threshold = max(threshold, BIG_LINE_THRESHOLD)
    p_home = _phi(edge_home / scatter_for(home_spread, params))
    if p_home >= threshold:
        side, prob, team, line = 'home', p_home, home_team, home_spread
    elif 1 - p_home >= threshold:
        side, prob, team, line = 'away', 1 - p_home, away_team, -home_spread + 0.0
    else:
        return None
    label = f"{team} PK" if line == 0 else f"{team} {line:+.1f}"
    return {'side': side, 'label': label, 'edge': round(abs(edge_home), 1), 'prob': round(prob, 4)}


def bet_fields(bet, retro=False):
    """Column values for one prediction row (a no-bet row stores is_bet = False and NULLs)."""
    if bet is None:
        return {'is_bet': False, 'bet_side': None, 'bet_label': None,
                'bet_edge': None, 'bet_prob': None, 'bet_units': None, 'bet_retro': None}
    return {'is_bet': True, 'bet_side': bet['side'], 'bet_label': bet['label'],
            'bet_edge': bet['edge'], 'bet_prob': bet['prob'], 'bet_units': BET_UNITS, 'bet_retro': bool(retro)}


def sql_value(v):
    """SQL literal for a bet field (strings escaped the same way as team names elsewhere)."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return 'NULL'
    if isinstance(v, (bool, np.bool_)):
        return 'TRUE' if v else 'FALSE'
    if isinstance(v, str):
        return "'" + v.replace("'", "''") + "'"
    return repr(float(v))


def make_ou_bet(predicted_total, total_line, ou_scatter, threshold=BET_THRESHOLD):
    """
    The model's over/under bet on one game, or None.
    Returns {'side': 'over'|'under', 'label': 'OVER 146.5', 'edge' (points), 'prob'} or None.
    """
    if predicted_total is None or total_line is None or pd.isna(predicted_total) or pd.isna(total_line):
        return None
    total_line = float(total_line)
    edge = float(predicted_total) - total_line
    p_over = _phi(edge / ou_scatter)
    if p_over >= threshold:
        side, prob = 'over', p_over
    elif 1 - p_over >= threshold:
        side, prob = 'under', 1 - p_over
    else:
        return None
    return {'side': side, 'label': f"{side.upper()} {total_line:g}", 'edge': round(abs(edge), 1), 'prob': round(prob, 4)}


def ou_fields(bet, retro=False):
    """O/U column values for one prediction row (no bet -> ou_is_bet False and NULLs)."""
    if bet is None:
        return {'ou_is_bet': False, 'ou_side': None, 'ou_label': None,
                'ou_edge': None, 'ou_prob': None, 'ou_units': None, 'ou_retro': None}
    return {'ou_is_bet': True, 'ou_side': bet['side'], 'ou_label': bet['label'], 'ou_edge': bet['edge'],
            'ou_prob': bet['prob'], 'ou_units': BET_UNITS, 'ou_retro': bool(retro)}


def load_graded_totals(cursor, season_start):
    """Graded games with a pre-game O/U line, for fit_ou_scatter: game_date, total_line, actual_total."""
    cursor.execute(f"""
        SELECT o.game_date, h.vegas_total AS total_line,
               o.actual_home_score + o.actual_away_score AS actual_total,
               h.prediction_date, h.game_time
        FROM workspace.default.prediction_outcomes o
        JOIN workspace.default.prediction_history h
          ON h.game_date = o.game_date AND h.home_team = o.home_team AND h.away_team = o.away_team
        WHERE o.game_date >= '{season_start}'
          AND h.vegas_total IS NOT NULL
          AND NOT (o.actual_home_score = 0 AND o.actual_away_score = 0)
    """)
    df = cursor.fetchall_arrow().to_pandas()
    df = df[predicted_before_tip(df['prediction_date'], df['game_time'])]
    df['game_date'] = pd.to_datetime(df['game_date'])
    return df[['game_date', 'total_line', 'actual_total']].reset_index(drop=True)


def fit_ou_scatter(graded, day):
    """
    How far final totals land from the O/U line (one number, points), fit on games graded before `day`:
    last SCATTER_WINDOW_DAYS days, else the season, else OU_SCATTER_DEFAULT. Returns (scatter, description).
    """
    day = pd.Timestamp(day)
    hist = graded[graded['game_date'] < day]
    recent = hist[hist['game_date'] >= day - pd.Timedelta(days=SCATTER_WINDOW_DAYS)]
    if len(recent) >= OU_SCATTER_MIN_GAMES:
        sample, source = recent, f"{len(recent)} games, last {SCATTER_WINDOW_DAYS} days"
    elif len(hist) >= OU_SCATTER_MIN_GAMES:
        sample, source = hist, f"{len(hist)} games, season to date"
    else:
        return OU_SCATTER_DEFAULT, f"default ({len(hist)} graded totals < {OU_SCATTER_MIN_GAMES})"
    miss = (sample['actual_total'] - sample['total_line']).to_numpy(dtype=float)
    fit = float(np.sqrt(np.mean(miss ** 2)))          # maximum-likelihood width for a curve centred on the line
    scatter = round(min(OU_SCATTER_RANGE[1], max(OU_SCATTER_RANGE[0], fit)), 1)
    return scatter, f"{source}{'' if scatter == round(fit, 1) else f', clamped from {fit:.1f}'}"


def fit_ou_for_run(cursor, day):
    """Load graded totals and fit the O/U scatter for a prediction run on `day`. Never raises."""
    try:
        scatter, source = fit_ou_scatter(load_graded_totals(cursor, season_start_for(day)), day)
    except Exception as e:
        scatter, source = OU_SCATTER_DEFAULT, f"default (fit failed: {type(e).__name__}: {e})"
    print(f"  O/U scatter: {scatter:.1f} pts ({source}); betting O/U at {BET_THRESHOLD:.0%}, {BET_UNITS:g}u each")
    return scatter


def ensure_ou_columns(cursor, table):
    """Add the O/U line and O/U bet columns to `table` if missing."""
    for name, kind in [('vegas_total', 'DOUBLE')] + OU_COLUMNS:
        try:
            cursor.execute(f"ALTER TABLE {table} ADD COLUMNS ({name} {kind})")
        except Exception:
            pass


def ensure_bet_columns(cursor, table):
    """Add the bet columns to `table` if missing (existing rows get NULL)."""
    for name, kind in BET_COLUMNS:
        try:
            cursor.execute(f"ALTER TABLE {table} ADD COLUMNS ({name} {kind})")
        except Exception:
            pass   # already exists


def season_start_for(day):
    day = pd.Timestamp(day)
    return f"{day.year if day.month >= 7 else day.year - 1}-11-01"


def fit_for_run(cursor, day):
    """Load graded lines and fit the scatter for a prediction run on `day`. Never raises."""
    try:
        params, source = fit_scatter(load_graded_lines(cursor, season_start_for(day)), day)
    except Exception as e:
        params, source = SCATTER_DEFAULT, f"default (fit failed: {type(e).__name__}: {e})"
    print(f"  Bet scatter: {params[0]:.1f} + {params[1]:.3f} x |line| ({source}); "
          f"betting at {BET_THRESHOLD:.0%} cover probability, {BET_UNITS:g}u each")
    return params
