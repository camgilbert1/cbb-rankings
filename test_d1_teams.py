"""
Offline checks for d1_teams.py (run: python test_d1_teams.py). Fails loudly if the D1 table could let
two schools share one KenPom team, which is how Delaware State got Delaware's ratings.
"""
from collections import defaultdict

import pandas as pd

from d1_teams import D1_TEAMS
from team_names import _name_key, find_team_match


def main():
    problems = []
    for season, table in D1_TEAMS.items():
        by_kp = defaultdict(list)
        for espn, kp in table.items():
            by_kp[kp].append(espn)
        shared = {kp: v for kp, v in by_kp.items() if len(v) > 1}
        if shared:
            problems.append(f"{season}: KenPom teams with more than one ESPN school: {shared}")
        keys = defaultdict(set)
        for espn, kp in table.items():
            keys[_name_key(espn)].add(kp)
        clash = {k: v for k, v in keys.items() if len(v) > 1}
        if clash:
            problems.append(f"{season}: ESPN names that normalize to the same key but different teams: {clash}")
        # Round trip: matching each school against that season's KenPom list returns its own team
        stats = pd.DataFrame({'team_name': sorted(set(table.values()))})
        wrong = {espn: (kp, find_team_match(espn, stats)) for espn, kp in table.items() if find_team_match(espn, stats) != kp}
        if wrong:
            problems.append(f"{season}: round-trip mismatches: {wrong}")
        print(f"{season}: {len(table)} D1 schools checked")

    # Known traps from the old fuzzy matcher must stay correct / non-D1
    stats = pd.DataFrame({'team_name': sorted(set(D1_TEAMS[max(D1_TEAMS)].values()))})
    expect = {'Delaware State Hornets': 'Delaware St.', 'Delaware Blue Hens': 'Delaware', 'UConn Huskies': 'Connecticut',
              'Central Connecticut Blue Devils': 'Central Connecticut', "Saint Mary's Gaels": 'Saint Marys',
              'Saint Marys Gaels': 'Saint Marys', 'Colorado Christian Cougars': None, 'Northwest University Eagles': None,
              'Colorado Buffaloes': 'Colorado', 'Kansas City Roos': 'Kansas City', 'Kansas Jayhawks': 'Kansas'}
    for espn, want in expect.items():
        got = find_team_match(espn, stats)
        if got != want:
            problems.append(f"{espn!r}: expected {want!r}, got {got!r}")

    # Master table: ESPN ids and Odds API names are one-to-one and cover every current D1 school
    from d1_teams import ESPN_IDS, ODDS_NAMES
    current = set(D1_TEAMS[max(D1_TEAMS)])
    if set(ODDS_NAMES) != current:
        problems.append(f"ODDS_NAMES vs current D1 schools: missing {sorted(current - set(ODDS_NAMES))}, extra {sorted(set(ODDS_NAMES) - current)}")
    if len(set(ODDS_NAMES.values())) != len(ODDS_NAMES):
        problems.append("an Odds API name is used by two schools")
    if not current <= set(ESPN_IDS) or len({ESPN_IDS[n] for n in current}) != len(current):
        problems.append("ESPN_IDS missing a current school or reusing an id")
    from team_names import odds_name_for
    for espn, want in {'Delaware State Hornets': 'Delaware St Hornets', 'Miami Hurricanes': 'Miami Hurricanes',
                       'Miami (OH) RedHawks': None if 'Miami (OH) RedHawks' not in ODDS_NAMES else ODDS_NAMES['Miami (OH) RedHawks'],
                       'Saint Marys Gaels': ODDS_NAMES["Saint Mary's Gaels"], 'Colorado Christian Cougars': None}.items():
        if odds_name_for(espn) != want:
            problems.append(f"odds_name_for({espn!r}) = {odds_name_for(espn)!r}, expected {want!r}")
    print(f"master table: {len(ODDS_NAMES)} Odds API names, {len(ESPN_IDS)} ESPN ids")

    if problems:
        raise SystemExit("D1 TABLE CHECK FAILED:\n  " + "\n  ".join(problems))
    print("All D1 table checks passed")


if __name__ == '__main__':
    main()
