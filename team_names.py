"""
Centralized team name mapping utility.
All ESPN→KenPom, ESPN→Odds API, and BracketMatrix→KenPom mappings live here.
"""

# Non-D1 school names (without mascot) that the fuzzy matcher incorrectly maps
# to D1 KenPom entries. Return None immediately for these.
NON_D1_SCHOOLS = {
    'Lincoln (CA)',              # → incorrectly matches 'Oakland' via substring
    'Campbellsville',            # → incorrectly matches 'Campbell' via substring
    'Oklahoma Panhandle',        # → incorrectly matches 'Oklahoma' via containment
    'Thomas College',            # → incorrectly matches 'St. Thomas' (only KenPom "Thomas" team)
    'Bryant & Stratton-Albany',  # → incorrectly matches 'Bryant' via first-word
    'William Paterson',          # → incorrectly matches 'William & Mary' via first-word
    'Virginia Lynchburg',        # → incorrectly matches 'Virginia' via containment
}

# ESPN school name (without mascot) → KenPom name
# Used when the automatic matching logic can't resolve the name correctly
ESPN_TO_KENPOM = {
    'UT Martin': 'Tennessee Martin',
    'Southeast Missouri State': 'Southeast Missouri St.',
    'SIU Edwardsville': 'SIU Edwardsville',
    'UL Monroe': 'Louisiana Monroe',
    'Ole Miss': 'Mississippi',
    'Utah State': 'Utah St.',
    'Washington State': 'Washington St.',
    'Mississippi State': 'Mississippi St.',
    'Illinois State': 'Illinois St.',
    'Missouri State': 'Missouri St.',
    'Jacksonville State': 'Jacksonville St.',
    'Loyola Maryland': 'Loyola MD',
    'Colorado State': 'Colorado St.',
    'Indiana State': 'Indiana St.',
    'Oklahoma State': 'Oklahoma St.',
    'South Dakota State': 'South Dakota St.',
    'North Dakota State': 'North Dakota St.',
    'Kansas State': 'Kansas St.',
    'Michigan State': 'Michigan St.',
    'Oregon State': 'Oregon St.',
    'Penn State': 'Penn St.',
    'Iowa State': 'Iowa St.',
    'Ohio State': 'Ohio St.',
    'Arizona State': 'Arizona St.',
    'Fresno State': 'Fresno St.',
    'Boise State': 'Boise St.',
    'Wichita State': 'Wichita St.',
    'San Diego State': 'San Diego St.',
    'San Jose State': 'San Jose St.',
    'Appalachian State': 'Appalachian St.',
    'Georgia State': 'Georgia St.',
    'Ball State': 'Ball St.',
    'Kent State': 'Kent St.',
    'Wright State': 'Wright St.',
    'Youngstown State': 'Youngstown St.',
    'Norfolk State': 'Norfolk St.',
    'Coppin State': 'Coppin St.',
    'Morgan State': 'Morgan St.',
    'Alcorn State': 'Alcorn St.',
    'Grambling State': 'Grambling St.',
    'Texas State': 'Texas St.',
    'Arkansas State': 'Arkansas St.',
    'Cleveland State': 'Cleveland St.',
    'Portland State': 'Portland St.',
    'Weber State': 'Weber St.',
    'Idaho State': 'Idaho St.',
    'Montana State': 'Montana St.',
    'Kennesaw State': 'Kennesaw St.',
    'Murray State': 'Murray St.',
    'Nicholls State': 'Nicholls St.',
    'McNeese State': 'McNeese St.',
    'Northwestern State': 'Northwestern St.',
    "Saint Joseph's": 'Saint Josephs',
    "Saint Mary's": 'Saint Marys',
    "St. John's": 'St. Johns',
    "Hawai'i": 'Hawaii',
    'South Carolina Upstate': 'USC Upstate',
    'Florida International': 'FIU',
    'Alabama State': 'Alabama St.',
    'Mississippi Valley State': 'Mississippi Valley St.',
    'Tennessee State': 'Tennessee St.',
    'California Baptist': 'Cal Baptist',
    'Cal State Fullerton': 'Cal St. Fullerton',
    'Cal State Bakersfield': 'Cal St. Bakersfield',
    'Cal State Northridge': 'CSUN',  # KenPom uses 'CSUN', not 'Cal St. Northridge'
    'Florida State': 'Florida St.',
    'Chicago State': 'Chicago St.',
    'South Carolina State': 'South Carolina St.',
    'San José State': 'San Jose St.',
    'SE Louisiana': 'Southeastern Louisiana',
    'New Mexico State': 'New Mexico St.',
    'Texas A&M-Corpus Christi': 'Texas A&M Corpus Chris',
    'Alabama A&M': 'Alabama A&M',
    'Jackson State': 'Jackson St.',
    'Towson State': 'Towson St.',
}

# Common abbreviation mappings (ESPN first word → KenPom name)
ESPN_ABBREV_TO_KENPOM = {
    'UConn': 'Connecticut',
    'UNLV': 'Nevada-Las Vegas',
    'USC': 'Southern California',
    'SMU': 'Southern Methodist',
    'TCU': 'Texas Christian',
    'BYU': 'Brigham Young',
    'UCF': 'Central Florida',
    'VCU': 'Virginia Commonwealth',
    'LSU': 'Louisiana State',
    'UTEP': 'Texas-El Paso',
    'UTSA': 'Texas-San Antonio',
    'UAB': 'Alabama-Birmingham',
    'UIC': 'Illinois Chicago',
}

# ESPN team name (with mascot) → Odds API team name
ESPN_TO_ODDS = {
    'IU Indianapolis Jaguars': 'IUPUI Jaguars',
    'Purdue Fort Wayne Mastodons': 'Fort Wayne Mastodons',
    'Gardner-Webb Runnin Bulldogs': 'Gardner-Webb Bulldogs',
    'Florida International Panthers': "Florida Int'l Golden Panthers",
    'Northern Colorado Bears': 'N Colorado Bears',
    'SIU Edwardsville Cougars': 'SIU-Edwardsville Cougars',
    'Cal State Bakersfield Roadrunners': 'CSU Bakersfield Roadrunners',
    'Long Beach State Beach': 'Long Beach St 49ers',
    'Southeast Missouri State Redhawks': 'SE Missouri St Redhawks',
    'Sacramento State Hornets': 'Sacramento St Hornets',
    'Hawaii Rainbow Warriors': "Hawai'i Rainbow Warriors",
    'Cal State Fullerton Titans': 'CSU Fullerton Titans',
    'Loyola Chicago Ramblers': 'Loyola (Chi) Ramblers',
    'UT Martin Skyhawks': 'Tenn-Martin Skyhawks',
    'Seattle U Redhawks': 'Seattle Redhawks',
    'Cal State Northridge Matadors': 'CSU Northridge Matadors',
    'Mississippi Valley State Delta Devils': 'Miss Valley St Delta Devils',
    'Prairie View A&M Panthers': 'Prairie View Panthers',
    'UT Arlington Mavericks': 'UT-Arlington Mavericks',
    'California Baptist Lancers': 'Cal Baptist Lancers',
    'Little Rock Trojans': 'Arkansas-Little Rock Trojans',
    'Kansas City Roos': 'UMKC Kangaroos',
    'Grand Canyon Lopes': 'Grand Canyon Antelopes',
    'Texas A&M-Corpus Christi Islanders': 'Texas A&M-CC Islanders',
}

# BracketMatrix team name → KenPom name
BRACKETMATRIX_TO_KENPOM = {
    'NC-Wilmington': 'UNC Wilmington',
    'Miami (FLA.)': 'Miami FL',
    'Miami (Ohio)': 'Miami OH',
    'E. Tennessee State': 'East Tennessee St.',
    'Tennessee-Martin': 'Tennessee Martin',
    'Long Island': 'LIU',
    "St. Mary's (CA)": "Saint Marys",
    'North Carolina State': 'NC State',
    'Bethune-Cookman': 'Bethune Cookman',
}


def find_team_match(espn_name, team_stats):
    """
    Match ESPN team name to KenPom team name.

    Args:
        espn_name (str): Full team name from ESPN (e.g., "Michigan Wolverines")
        team_stats (pd.DataFrame): Team statistics with KenPom names

    Returns:
        str: Matched KenPom team name, or None if no match found
    """
    words = espn_name.split()
    school_name = ' '.join(words[:-1]) if len(words) > 1 else espn_name

    # Reject known non-D1 schools before any fuzzy matching
    if school_name in NON_D1_SCHOOLS or any(school_name.startswith(s) for s in NON_D1_SCHOOLS):
        return None

    # Check explicit school name mapping first
    mapped = None
    if school_name in ESPN_TO_KENPOM:
        mapped = ESPN_TO_KENPOM[school_name]
    else:
        # Prefix match — check longest keys first to prefer specific over general
        for key in sorted(ESPN_TO_KENPOM.keys(), key=len, reverse=True):
            if school_name.startswith(key):
                mapped = ESPN_TO_KENPOM[key]
                break

    if mapped:
        if mapped in team_stats['team_name'].values:
            return mapped
        if mapped + '.' in team_stats['team_name'].values:
            return mapped + '.'
        contains = team_stats[team_stats['team_name'].str.contains(mapped, case=False, na=False)]
        if len(contains) == 1:
            return contains.iloc[0]['team_name']

    # Try exact match on full ESPN name
    if espn_name in team_stats['team_name'].values:
        return espn_name

    # Try school name (without mascot) as exact match
    if school_name in team_stats['team_name'].values:
        return school_name

    # Try with hyphens replaced by spaces
    school_name_no_hyphen = school_name.replace('-', ' ')
    if school_name_no_hyphen != school_name and school_name_no_hyphen in team_stats['team_name'].values:
        return school_name_no_hyphen

    # Try common abbreviation mappings (first word)
    first_word = words[0]
    search_term = ESPN_ABBREV_TO_KENPOM.get(first_word, first_word)

    matches = team_stats[team_stats['team_name'].str.contains(search_term, case=False, na=False)]

    if len(matches) == 1:
        return matches.iloc[0]['team_name']
    elif len(matches) > 1:
        # Exact name match
        for team_name in matches['team_name']:
            if team_name.lower() in (school_name.lower(), school_name_no_hyphen.lower()):
                return team_name
        # Containment match — prefer most specific (longest)
        containment_matches = []
        for team_name in matches['team_name']:
            if school_name.lower() in team_name.lower() or team_name.lower() in school_name.lower():
                containment_matches.append(team_name)
        if containment_matches:
            return max(containment_matches, key=len)
        # Word boundary match on first word
        for team_name in matches['team_name']:
            if search_term.lower() in team_name.lower().split():
                return team_name
        return matches.iloc[0]['team_name']

    # Try hyphen variant in search
    if '-' in search_term:
        search_no_hyphen = search_term.replace('-', ' ')
        matches = team_stats[team_stats['team_name'].str.contains(search_no_hyphen, case=False, na=False)]
        if len(matches) >= 1:
            return matches.iloc[0]['team_name']

    # Last resort: check if any KenPom name is a substring of the ESPN name
    # Only allow if the KenPom name is at least 6 chars to avoid short false matches
    for team_name in team_stats['team_name']:
        if len(team_name) >= 6 and team_name.lower() in espn_name.lower():
            return team_name

    return None


def normalize_team_name_for_odds(espn_name, vegas_spreads=None):
    """
    Normalize ESPN team name to match Odds API team name.

    Args:
        espn_name (str): Team name from ESPN API
        vegas_spreads (dict): Optional spreads dict to validate against

    Returns:
        str: Normalized team name for Odds API lookup
    """
    if espn_name in ESPN_TO_ODDS:
        return ESPN_TO_ODDS[espn_name]

    if vegas_spreads is not None:
        if espn_name in vegas_spreads:
            return espn_name

        if ' State ' in espn_name:
            abbreviated = espn_name.replace(' State ', ' St ')
            if abbreviated in vegas_spreads:
                return abbreviated

        words = espn_name.split()
        if len(words) > 1:
            without_mascot = ' '.join(words[:-1])
            if without_mascot in vegas_spreads:
                return without_mascot
            if ' State ' in without_mascot:
                abbreviated_no_mascot = without_mascot.replace(' State ', ' St ')
                if abbreviated_no_mascot in vegas_spreads:
                    return abbreviated_no_mascot

    return espn_name
