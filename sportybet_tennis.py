"""
Sportybet tennis scraper.
Covers: ATP, Challenger, ITF, WTA, UTR — 100-200 matches/day
Primary: Sportybet internal API
Fallback: SofaScore (free, no key)
"""

import requests
import time
import logging
import json
from datetime import datetime
import pytz

log = logging.getLogger(__name__)

TENNIS_SPORT_ID = "sr:sport:5"

HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) '
        'AppleWebKit/605.1.15 (KHTML, like Gecko) '
        'Version/17.0 Mobile/15E148 Safari/604.1'
    ),
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'en-US,en;q=0.9',
    'Origin': 'https://www.sportybet.com',
    'Referer': 'https://www.sportybet.com/ng/m/sport/tennis/today',
    'x-requested-with': 'XMLHttpRequest',
}

SPORTYBET_ENDPOINTS = [
    "https://www.sportybet.com/api/ng/factsCenter/sports/sr:sport:5/events",
    "https://www.sportybet.com/api/ng/factsCenter/sports/sr:sport:5/todayEvents",
    "https://www.sportybet.com/api/ng/factsCenter/queryEvents",
    "https://www.sportybet.com/api/ng/factsCenter/popularSports/events",
    "https://www.sportybet.com/api/ng/factsCenter/sports/sr:sport:5/schedule",
    "https://sportybet.com/api/ng/factsCenter/sports/sr:sport:5/events",
]

PARAM_SETS = [
    {'sportId': 'sr:sport:5', 'pageSize': 500, 'pageNum': 1},
    {'sportId': 'sr:sport:5', 'marketId': '186', 'pageSize': 500},
    {'sport': 'tennis', 'size': 500},
    {'sportId': 'sr:sport:5', 'type': 'today', 'pageSize': 500},
    {'sportId': 'sr:sport:5'},
]


def safe_get(url: str, params: dict = None,
             extra_headers: dict = None) -> dict:
    h = {**HEADERS, **(extra_headers or {})}
    for attempt in range(3):
        try:
            r = requests.get(url, params=params, headers=h, timeout=15)
            if r.status_code == 200:
                try:    return r.json()
                except: return {}
            if r.status_code == 429:
                time.sleep(4 * (attempt + 1))
                continue
            if r.status_code in (403, 401):
                HEADERS['User-Agent'] = (
                    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                    'AppleWebKit/537.36 Chrome/122.0.0.0 Safari/537.36'
                )
                time.sleep(2)
                continue
            return {}
        except Exception as e:
            log.debug(f"safe_get {url}: {e}")
            time.sleep(2)
    return {}


def _extract_events(data: dict) -> list:
    """Try all known JSON paths to find events list."""
    if not data:
        return []
    return (
        data.get('data', {}).get('events', []) or
        data.get('data', {}).get('records', []) or
        data.get('events', []) or
        data.get('records', []) or
        (data.get('data', []) if isinstance(data.get('data'), list) else []) or
        []
    )


def _sportybet_fetch(date_str: str) -> list:
    """Try all Sportybet endpoints and param combinations."""
    for url in SPORTYBET_ENDPOINTS:
        for params in PARAM_SETS:
            try:
                data   = safe_get(url, {**params, 'date': date_str})
                events = _extract_events(data)
                if events:
                    log.info(f"Sportybet: {len(events)} events from {url}")
                    return events
                # Also try without date
                data   = safe_get(url, params)
                events = _extract_events(data)
                if events:
                    log.info(f"Sportybet (no date): {len(events)} from {url}")
                    return events
            except Exception as e:
                log.debug(f"Endpoint {url}: {e}")
    return []


def _parse_event(event: dict, timezone: str) -> dict:
    """Parse Sportybet event into standard match dict."""
    TZ = pytz.timezone(timezone)
    try:
        event_id = str(
            event.get('eventId') or event.get('id') or
            event.get('matchId') or ''
        )
        p1 = (
            event.get('homeTeamName') or event.get('homeName') or
            event.get('home', {}).get('name') or
            (event.get('competitors', [{}])[0].get('name', '')
             if event.get('competitors') else '') or ''
        ).strip()
        p2 = (
            event.get('awayTeamName') or event.get('awayName') or
            event.get('away', {}).get('name') or
            (event.get('competitors', [{}]*2)[1].get('name', '')
             if len(event.get('competitors', [])) > 1 else '') or ''
        ).strip()

        if not p1 or not p2:
            return None

        tourney = (
            event.get('tournamentName') or
            event.get('tournament', {}).get('name') or
            event.get('leagueName') or 'Tennis'
        )
        t_lower = tourney.lower()

        if 'wta' in t_lower or 'women' in t_lower:   tour = 'WTA'
        elif 'itf' in t_lower:                         tour = 'ITF'
        elif 'challenger' in t_lower:                  tour = 'Challenger'
        elif 'utr' in t_lower:                         tour = 'UTR'
        else:                                           tour = 'ATP'

        surface = 'Hard'
        ctx_str = t_lower + str(event.get('venue', '')).lower()
        if 'clay' in ctx_str:    surface = 'Clay'
        elif 'grass' in ctx_str: surface = 'Grass'

        is_gs   = any(gs in t_lower for gs in
                      ['australian', 'french', 'wimbledon', 'us open'])
        t_level = 'G' if is_gs else 'C' if 'challenger' in t_lower else 'S'
        best_of = 5 if (is_gs and tour == 'ATP') else 3

        time_local = 'TBD'
        ts = (event.get('estimateStartTime') or
              event.get('startTime') or event.get('kickoff') or 0)
        if ts:
            try:
                ts = int(ts)
                if ts > 1e10: ts = ts / 1000
                utc_dt = pytz.utc.localize(datetime.utcfromtimestamp(ts))
                time_local = utc_dt.astimezone(TZ).strftime('%I:%M %p')
            except Exception:
                pass

        # Extract odds
        p1_odds = p2_odds = None
        for market in (event.get('markets', []) or
                       event.get('odds', []) or []):
            m_id = str(market.get('id') or market.get('marketId') or '')
            if m_id in ('186', '1', 'winner', 'match_winner'):
                for o in (market.get('outcomes', []) or
                          market.get('odds', []) or []):
                    desc = str(
                        o.get('desc') or o.get('name') or ''
                    ).lower()
                    try:
                        odd = float(
                            o.get('odds') or o.get('oddValue') or
                            o.get('price') or 0
                        )
                        if odd > 100: odd = odd / 1000
                        if odd <= 1.0: odd = None
                    except Exception:
                        odd = None
                    if 'home' in desc or desc in ('1', 'w1'):
                        p1_odds = odd
                    elif 'away' in desc or desc in ('2', 'w2'):
                        p2_odds = odd
                if p1_odds or p2_odds:
                    break

        return {
            'fixture_id':    event_id or f"sb_{p1}_{p2}",
            'tour':          tour,
            'tourney_name':  tourney,
            'surface':       surface,
            'round':         str(event.get('round') or
                                 event.get('roundName') or 'R32'),
            'best_of':       best_of,
            'tourney_level': t_level,
            'p1_name':       p1,
            'p2_name':       p2,
            'p1_rank':       int(event.get('homeRank') or 200),
            'p2_rank':       int(event.get('awayRank') or 200),
            'p1_rank_pts':   0,
            'p2_rank_pts':   0,
            'p1_age':        25.0,
            'p2_age':        25.0,
            'p1_odds':       p1_odds,
            'p2_odds':       p2_odds,
            'time_local':    time_local,
            'status':        str(event.get('status') or ''),
            'source':        'sportybet',
        }
    except Exception as e:
        log.debug(f"_parse_event: {e}")
        return None


def _sofascore_fetch(timezone: str) -> list:
    """SofaScore fallback — very reliable for all tennis tours."""
    TZ      = pytz.timezone(timezone)
    today   = datetime.now().strftime('%Y-%m-%d')
    url     = f"https://api.sofascore.com/api/v1/sport/tennis/scheduled-events/{today}"
    matches = []
    seen    = set()

    data   = safe_get(url, extra_headers={
        'Referer': 'https://www.sofascore.com/',
        'Accept': 'application/json',
    })
    events = data.get('events', [])
    log.info(f"SofaScore: {len(events)} tennis events")

    for event in events:
        try:
            status = event.get('status', {}).get('type', '')
            if status in ('finished', 'canceled', 'postponed'):
                continue

            p1 = event.get('homeTeam', {}).get('name', '').strip()
            p2 = event.get('awayTeam', {}).get('name', '').strip()
            if not p1 or not p2:
                continue

            key = f"{p1}_{p2}"
            if key in seen: continue
            seen.add(key)

            tourney = event.get('tournament', {}).get('name', '')
            categ   = event.get('tournament', {}).get('category', {}).get('name', '')
            t_lower = (tourney + categ).lower()

            if 'wta' in t_lower or 'women' in t_lower: tour = 'WTA'
            elif 'itf' in t_lower:                       tour = 'ITF'
            elif 'challenger' in t_lower:                tour = 'Challenger'
            elif 'utr' in t_lower:                       tour = 'UTR'
            else:                                         tour = 'ATP'

            surface = 'Hard'
            ground  = str(event.get('groundType', '')).lower()
            if 'clay' in ground:    surface = 'Clay'
            elif 'grass' in ground: surface = 'Grass'

            time_local = 'TBD'
            ts = event.get('startTimestamp', 0)
            if ts:
                try:
                    utc_dt = pytz.utc.localize(
                        datetime.utcfromtimestamp(int(ts))
                    )
                    time_local = utc_dt.astimezone(TZ).strftime('%I:%M %p')
                except Exception:
                    pass

            is_gs   = any(gs in t_lower for gs in
                          ['australian', 'french', 'wimbledon', 'us open'])
            t_level = 'G' if is_gs else 'C' if 'challenger' in t_lower else 'S'
            best_of = 5 if (is_gs and tour == 'ATP') else 3

            matches.append({
                'fixture_id':    f"sofa_{event.get('id','')}",
                'tour':          tour,
                'tourney_name':  tourney,
                'surface':       surface,
                'round':         str(event.get('roundInfo', {}).get('name', 'R32')),
                'best_of':       best_of,
                'tourney_level': t_level,
                'p1_name':       p1,
                'p2_name':       p2,
                'p1_rank':       int(event.get('homeTeam', {}).get('ranking', 200) or 200),
                'p2_rank':       int(event.get('awayTeam', {}).get('ranking', 200) or 200),
                'p1_rank_pts':   0,
                'p2_rank_pts':   0,
                'p1_age':        25.0,
                'p2_age':        25.0,
                'p1_odds':       None,
                'p2_odds':       None,
                'time_local':    time_local,
                'status':        status,
                'source':        'sofascore',
            })
        except Exception as e:
            log.debug(f"SofaScore parse: {e}")

    return matches


def get_tennis_matches(timezone: str = 'Africa/Lagos',
                        tour_filter: str = None) -> list:
    """
    Main entry point.
    Returns all today's tennis matches from Sportybet -> SofaScore.
    """
    today   = datetime.now().strftime('%Y-%m-%d')
    matches = []
    seen    = set()

    # Try Sportybet first
    raw = _sportybet_fetch(today)
    for event in raw:
        m = _parse_event(event, timezone)
        if not m: continue
        key = f"{m['p1_name']}_{m['p2_name']}"
        if key in seen: continue
        seen.add(key)
        matches.append(m)

    # Fallback to SofaScore if Sportybet empty
    if not matches:
        log.warning("Sportybet empty — using SofaScore")
        matches = _sofascore_fetch(timezone)

    # Tour filter
    if tour_filter and matches:
        filtered = [
            m for m in matches
            if tour_filter.upper() in m.get('tour', '').upper() or
               tour_filter.upper() in m.get('tourney_name', '').upper()
        ]
        matches = filtered or matches  # fallback to all if filter returns 0

    log.info(f"Tennis matches: {len(matches)} ({tour_filter or 'all tours'})")
    return matches
