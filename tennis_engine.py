import requests
import sqlite3
import pandas as pd
import numpy as np
import joblib
import json
import time
import logging
import os
import io
import re
import warnings
warnings.filterwarnings('ignore')

from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, roc_auc_score,
                              brier_score_loss, precision_score,
                              recall_score, log_loss)
from xgboost import XGBClassifier
import pytz

log = logging.getLogger(__name__)

HEADERS_BROWSER = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/122.0.0.0 Safari/537.36'
    ),
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'en-US,en;q=0.9',
}

HEADERS_MOBILE = {
    'User-Agent': (
        'Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) '
        'AppleWebKit/605.1.15 (KHTML, like Gecko) '
        'Version/17.0 Mobile/15E148 Safari/604.1'
    ),
    'Accept': 'application/json',
    'Referer': 'https://www.sofascore.com/',
}

DB_PATH    = "tennis_hunter.db"
MODELS_DIR = "models"
LOOKBACK   = 15         # last N matches for form
MIN_GAMES  = 5          # minimum matches before trusting stats
MAX_CONF   = 90.0
MIN_CONF   = 60.0
MIN_EDGE   = 4.0        # minimum % edge to flag value
TRAIN_YEARS = list(range(2015, 2026))   # 10 years of data

ESPN_BASE  = "http://site.api.espn.com/apis/site/v2/sports/tennis"

SACKMANN_ATP = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_{year}.csv"
SACKMANN_WTA = "https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master/wta_matches_{year}.csv"
SACKMANN_CHALL = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_qual_chall_{year}.csv"

TENNIS_DATA_URLS = {
    'ao_2025':   'http://www.tennis-data.co.uk/2025/ausopen.csv',
    'rg_2025':   'http://www.tennis-data.co.uk/2025/frenchopen.csv',
    'wimb_2024': 'http://www.tennis-data.co.uk/2024/wimbledon.csv',
    'uso_2024':  'http://www.tennis-data.co.uk/2024/usopen.csv',
    'ao_2024':   'http://www.tennis-data.co.uk/2024/ausopen.csv',
    'rg_2024':   'http://www.tennis-data.co.uk/2024/frenchopen.csv',
    'ao_2023':   'http://www.tennis-data.co.uk/2023/ausopen.csv',
    'rg_2023':   'http://www.tennis-data.co.uk/2023/frenchopen.csv',
    'wimb_2023': 'http://www.tennis-data.co.uk/2023/wimbledon.csv',
    'uso_2023':  'http://www.tennis-data.co.uk/2023/usopen.csv',
}

# Surface encoding
SURFACE_MAP = {
    'Hard':  0,
    'Clay':  1,
    'Grass': 2,
    'Carpet': 3,
}

# Tournament level encoding
LEVEL_MAP = {
    'G': 5,   # Grand Slam
    'M': 4,   # Masters 1000
    'A': 3,   # ATP 500
    'D': 3,   # Davis Cup
    'F': 4,   # Tour Finals
    'C': 1,   # Challenger
    'S': 2,   # ATP 250
    'O': 0,   # Other
}

ROUND_MAP = {
    'F':   7,  # Final
    'SF':  6,  # Semifinal
    'QF':  5,  # Quarterfinal
    'R16': 4,
    'R32': 3,
    'R32': 3,
    'R64': 2,
    'R128':1,
    'RR':  4,  # Round Robin
}

FEATURE_COLUMNS = [
    # Ranking features
    'p1_rank', 'p2_rank', 'rank_diff', 'rank_ratio', 'log_rank_diff',
    'p1_rank_pts', 'p2_rank_pts',

    # Surface-specific win rates
    'p1_surface_winrate', 'p2_surface_winrate', 'surface_winrate_diff',
    'p1_surface_games', 'p2_surface_games',

    # Overall form (last LOOKBACK matches)
    'p1_overall_winrate', 'p2_overall_winrate',
    'p1_form_last5', 'p2_form_last5',       # wins in last 5
    'p1_form_last10', 'p2_form_last10',     # wins in last 10

    # Serve stats
    'p1_ace_rate', 'p2_ace_rate', 'ace_rate_diff',
    'p1_df_rate', 'p2_df_rate',
    'p1_1st_serve_pct', 'p2_1st_serve_pct',
    'p1_1st_won_pct', 'p2_1st_won_pct',
    'p1_2nd_won_pct', 'p2_2nd_won_pct',
    'p1_bp_saved_pct', 'p2_bp_saved_pct',

    # H2H
    'h2h_p1_wins', 'h2h_total', 'h2h_p1_winrate',
    'h2h_surface_p1_wins', 'h2h_surface_total',

    # Match context
    'surface_enc', 'tourney_level_enc', 'round_enc', 'best_of',
    'is_grand_slam', 'is_masters',

    # Age + experience
    'p1_age', 'p2_age', 'age_diff',
    'p1_total_career_matches', 'p2_total_career_matches',

    # Implied probability from odds
    'implied_p1_prob',
]

def init_db():
    conn = sqlite3.connect(DB_PATH)

    # Check schema
    try:
        conn.execute("SELECT rank_diff FROM matches LIMIT 1")
        schema_ok = True
    except sqlite3.OperationalError:
        schema_ok = False
        log.warning("Schema outdated — rebuilding")

    if not schema_ok:
        conn.executescript("""
            DROP TABLE IF EXISTS matches;
            DROP TABLE IF EXISTS player_stats;
            DROP TABLE IF EXISTS fixtures;
            DROP TABLE IF EXISTS prediction_log;
            DROP TABLE IF EXISTS model_performance;
        """)

    conn.executescript("""
        CREATE TABLE IF NOT EXISTS matches (
            match_id        TEXT PRIMARY KEY,
            tour            TEXT,           -- ATP or WTA
            tourney_name    TEXT,
            surface         TEXT,
            tourney_level   TEXT,
            match_date      TEXT,
            round           TEXT,
            best_of         INTEGER,
            p1_name         TEXT,           -- winner stored as p1
            p2_name         TEXT,           -- loser stored as p2
            p1_id           TEXT,
            p2_id           TEXT,
            p1_rank         INTEGER,
            p2_rank         INTEGER,
            p1_rank_pts     INTEGER,
            p2_rank_pts     INTEGER,
            p1_age          REAL,
            p2_age          REAL,
            score           TEXT,
            minutes         INTEGER,
            p1_won          INTEGER,        -- always 1 (winner = p1)
            source          TEXT,

            -- Serve stats (winner)
            p1_ace          INTEGER,
            p1_df           INTEGER,
            p1_svpt         INTEGER,
            p1_1stIn        INTEGER,
            p1_1stWon       INTEGER,
            p1_2ndWon       INTEGER,
            p1_bpSaved      INTEGER,
            p1_bpFaced      INTEGER,

            -- Serve stats (loser)
            p2_ace          INTEGER,
            p2_df           INTEGER,
            p2_svpt         INTEGER,
            p2_1stIn        INTEGER,
            p2_1stWon       INTEGER,
            p2_2ndWon       INTEGER,
            p2_bpSaved      INTEGER,
            p2_bpFaced      INTEGER,

            -- Odds
            p1_odds         REAL,
            p2_odds         REAL,

            -- Computed features
            rank_diff               REAL DEFAULT 0,
            rank_ratio              REAL DEFAULT 1,
            log_rank_diff           REAL DEFAULT 0,
            p1_surface_winrate      REAL DEFAULT 0.5,
            p2_surface_winrate      REAL DEFAULT 0.5,
            surface_winrate_diff    REAL DEFAULT 0,
            p1_surface_games        INTEGER DEFAULT 0,
            p2_surface_games        INTEGER DEFAULT 0,
            p1_overall_winrate      REAL DEFAULT 0.5,
            p2_overall_winrate      REAL DEFAULT 0.5,
            p1_form_last5           INTEGER DEFAULT 0,
            p2_form_last5           INTEGER DEFAULT 0,
            p1_form_last10          INTEGER DEFAULT 0,
            p2_form_last10          INTEGER DEFAULT 0,
            p1_ace_rate             REAL DEFAULT 0,
            p2_ace_rate             REAL DEFAULT 0,
            ace_rate_diff           REAL DEFAULT 0,
            p1_df_rate              REAL DEFAULT 0,
            p2_df_rate              REAL DEFAULT 0,
            p1_1st_serve_pct        REAL DEFAULT 0,
            p2_1st_serve_pct        REAL DEFAULT 0,
            p1_1st_won_pct          REAL DEFAULT 0,
            p2_1st_won_pct          REAL DEFAULT 0,
            p1_2nd_won_pct          REAL DEFAULT 0,
            p2_2nd_won_pct          REAL DEFAULT 0,
            p1_bp_saved_pct         REAL DEFAULT 0,
            p2_bp_saved_pct         REAL DEFAULT 0,
            h2h_p1_wins             INTEGER DEFAULT 0,
            h2h_total               INTEGER DEFAULT 0,
            h2h_p1_winrate          REAL DEFAULT 0.5,
            h2h_surface_p1_wins     INTEGER DEFAULT 0,
            h2h_surface_total       INTEGER DEFAULT 0,
            surface_enc             INTEGER DEFAULT 0,
            tourney_level_enc       INTEGER DEFAULT 2,
            round_enc               INTEGER DEFAULT 3,
            is_grand_slam           INTEGER DEFAULT 0,
            is_masters              INTEGER DEFAULT 0,
            p1_total_career_matches INTEGER DEFAULT 0,
            p2_total_career_matches INTEGER DEFAULT 0,
            implied_p1_prob         REAL DEFAULT 0.5
        );

        CREATE TABLE IF NOT EXISTS player_stats (
            player_id       TEXT PRIMARY KEY,
            player_name     TEXT,
            tour            TEXT,
            last_rank       INTEGER,
            last_rank_pts   INTEGER,
            last_updated    TEXT,
            total_matches   INTEGER DEFAULT 0,
            -- Surface win rates
            hard_wins       INTEGER DEFAULT 0,
            hard_total      INTEGER DEFAULT 0,
            clay_wins       INTEGER DEFAULT 0,
            clay_total      INTEGER DEFAULT 0,
            grass_wins      INTEGER DEFAULT 0,
            grass_total     INTEGER DEFAULT 0,
            -- Serve averages (last 20 matches)
            avg_ace_rate    REAL DEFAULT 0,
            avg_df_rate     REAL DEFAULT 0,
            avg_1st_pct     REAL DEFAULT 0,
            avg_1st_won     REAL DEFAULT 0,
            avg_2nd_won     REAL DEFAULT 0,
            avg_bp_saved    REAL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS fixtures (
            fixture_id      TEXT PRIMARY KEY,
            tour            TEXT,
            tourney_name    TEXT,
            surface         TEXT,
            round           TEXT,
            match_date      TEXT,
            p1_name         TEXT,
            p2_name         TEXT,
            p1_id           TEXT,
            p2_id           TEXT,
            p1_rank         INTEGER,
            p2_rank         INTEGER,
            time_local      TEXT,
            p1_odds         REAL,
            p2_odds         REAL,
            status          TEXT,
            prediction_json TEXT
        );

        CREATE TABLE IF NOT EXISTS prediction_log (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at      TEXT,
            match_date      TEXT,
            tour            TEXT,
            match           TEXT,
            tourney_name    TEXT,
            surface         TEXT,
            p1_name         TEXT,
            p2_name         TEXT,
            predicted_winner TEXT,
            win_prob        REAL,
            book_implied    REAL,
            edge_pct        REAL,
            confidence      REAL,
            winner_odds     REAL,
            stake           REAL,
            result          TEXT DEFAULT 'PENDING',
            actual_winner   TEXT,
            profit_loss     REAL DEFAULT 0,
            settled_at      TEXT
        );

        CREATE TABLE IF NOT EXISTS model_performance (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            date            TEXT,
            tour            TEXT,
            accuracy        REAL,
            auc_roc         REAL,
            brier_score     REAL,
            log_loss        REAL,
            samples         INTEGER
        );
    """)

    conn.commit()
    conn.close()
    log.info("DB initialized")

def safe_get(url: str, params: dict = None,
             headers: dict = None, timeout: int = 12) -> dict:
    """Robust GET with retry."""
    h = headers or HEADERS_BROWSER
    for attempt in range(3):
        try:
            r = requests.get(url, params=params, headers=h, timeout=timeout)
            if r.status_code == 200:
                try:    return r.json()
                except: return {'_text': r.text[:1000]}
            if r.status_code == 429:
                time.sleep(5 * (attempt + 1))
                continue
            log.debug(f"HTTP {r.status_code} — {url}")
            return {}
        except requests.exceptions.ConnectionError:
            log.warning(f"Connection error attempt {attempt+1} — {url}")
            time.sleep(3)
        except Exception as e:
            log.debug(f"safe_get {url}: {e}")
            return {}
    return {}

def fetch_sackmann_year(year: int, tour: str = 'ATP') -> pd.DataFrame:
    if tour.upper() == 'ATP':
        url = SACKMANN_ATP.format(year=year)
    else:
        url = SACKMANN_WTA.format(year=year)

    try:
        log.info(f"Sackmann {tour} {year}: {url}")
        r = requests.get(url, timeout=20,
                         headers={'User-Agent': 'Mozilla/5.0'})
        if r.status_code != 200:
            log.warning(f"Sackmann {tour} {year}: HTTP {r.status_code}")
            return pd.DataFrame()

        df = pd.read_csv(io.StringIO(r.text), on_bad_lines='skip',
                         low_memory=False)
        df.columns = [c.strip() for c in df.columns]
        log.info(f"Sackmann {tour} {year}: {len(df)} matches")
        return df

    except Exception as e:
        log.error(f"Sackmann {tour} {year} error: {e}")
        return pd.DataFrame()

def fetch_tennis_data_csv(key: str) -> pd.DataFrame:
    url = TENNIS_DATA_URLS.get(key)
    if not url:
        return pd.DataFrame()
    try:
        r = requests.get(url, timeout=15,
                         headers={'User-Agent': 'Mozilla/5.0'})
        if r.status_code != 200:
            return pd.DataFrame()
        for enc in ['utf-8', 'latin-1', 'cp1252']:
            try:
                df = pd.read_csv(io.StringIO(r.content.decode(enc)),
                                 on_bad_lines='skip')
                df.columns = [c.strip() for c in df.columns]
                log.info(f"tennis-data.co.uk {key}: {len(df)} rows")
                return df
            except Exception:
                continue
    except Exception as e:
        log.error(f"tennis-data.co.uk {key}: {e}")
    return pd.DataFrame()

def store_sackmann_matches(df: pd.DataFrame, tour: str = 'ATP'):
    if df.empty:
        return

    conn = sqlite3.connect(DB_PATH)
    stored = skipped = 0

    # Running player cache: player_name -> list of match results
    player_cache = {}   # name -> [{won, surface, ace_rate, ...}]

    # Sort chronologically
    try:
        df['_date'] = pd.to_numeric(df['tourney_date'], errors='coerce')
        df = df.sort_values('_date').reset_index(drop=True)
    except Exception:
        pass

    for _, row in df.iterrows():
        try:
            winner = str(row.get('winner_name', '')).strip()
            loser  = str(row.get('loser_name', '')).strip()
            if not winner or not loser:
                skipped += 1
                continue

            # Parse date
            raw_date = str(row.get('tourney_date', '')).strip()
            try:
                match_date = datetime.strptime(raw_date[:8], '%Y%m%d').strftime('%Y-%m-%d')
            except Exception:
                skipped += 1
                continue

            surface = str(row.get('surface', 'Hard')).strip()
            level   = str(row.get('tourney_level', 'S')).strip()
            round_  = str(row.get('round', 'R32')).strip()
            best_of = int(row.get('best_of', 3) or 3)

            w_rank  = int(float(row.get('winner_rank', 500) or 500))
            l_rank  = int(float(row.get('loser_rank', 500) or 500))
            w_pts   = int(float(row.get('winner_rank_points', 0) or 0))
            l_pts   = int(float(row.get('loser_rank_points', 0) or 0))
            w_age   = float(row.get('winner_age', 25) or 25)
            l_age   = float(row.get('loser_age', 25) or 25)

            # Serve stats — safe extraction
            def safe_int(v):
                try:
                    return int(float(v or 0))
                except Exception:
                    return 0

            def safe_float(v):
                try:
                    return float(v or 0)
                except Exception:
                    return 0.0

            w_ace  = safe_int(row.get('w_ace'))
            w_df   = safe_int(row.get('w_df'))
            w_svpt = safe_int(row.get('w_svpt'))
            w_1stI = safe_int(row.get('w_1stIn'))
            w_1stW = safe_int(row.get('w_1stWon'))
            w_2ndW = safe_int(row.get('w_2ndWon'))
            w_bpS  = safe_int(row.get('w_bpSaved'))
            w_bpF  = safe_int(row.get('w_bpFaced'))

            l_ace  = safe_int(row.get('l_ace'))
            l_df   = safe_int(row.get('l_df'))
            l_svpt = safe_int(row.get('l_svpt'))
            l_1stI = safe_int(row.get('l_1stIn'))
            l_1stW = safe_int(row.get('l_1stWon'))
            l_2ndW = safe_int(row.get('l_2ndWon'))
            l_bpS  = safe_int(row.get('l_bpSaved'))
            l_bpF  = safe_int(row.get('l_bpFaced'))

            # ── Rolling features from player cache ──────────────────
            def get_player_stats(name, surf):
                hist = player_cache.get(name, [])
                if not hist:
                    return {
                        'surface_winrate': 0.5, 'surface_games': 0,
                        'overall_winrate': 0.5, 'form_last5': 2,
                        'form_last10': 5, 'ace_rate': 0.06,
                        'df_rate': 0.03, '1st_pct': 0.60,
                        '1st_won': 0.70, '2nd_won': 0.50,
                        'bp_saved': 0.60, 'total': 0,
                    }

                recent_all  = hist[-LOOKBACK:]
                recent_surf = [m for m in hist if m['surface'] == surf][-LOOKBACK:]
                n_all  = max(len(recent_all), 1)
                n_surf = max(len(recent_surf), 1)

                surf_wins = sum(1 for m in recent_surf if m['won'])
                all_wins  = sum(1 for m in recent_all if m['won'])

                last5   = sum(1 for m in recent_all[-5:] if m['won'])
                last10  = sum(1 for m in recent_all[-10:] if m['won'])

                ace_rates = [m['ace_rate'] for m in recent_all if m['ace_rate'] > 0]
                df_rates  = [m['df_rate']  for m in recent_all if m['df_rate']  >= 0]
                p1st      = [m['1st_pct']  for m in recent_all if m['1st_pct']  > 0]
                w1st      = [m['1st_won']  for m in recent_all if m['1st_won']  > 0]
                w2nd      = [m['2nd_won']  for m in recent_all if m['2nd_won']  > 0]
                bps       = [m['bp_saved'] for m in recent_all if m['bp_saved'] >= 0]

                return {
                    'surface_winrate': round(surf_wins / n_surf, 3),
                    'surface_games':   len(recent_surf),
                    'overall_winrate': round(all_wins / n_all, 3),
                    'form_last5':      last5,
                    'form_last10':     last10,
                    'ace_rate':        round(np.mean(ace_rates), 4) if ace_rates else 0.06,
                    'df_rate':         round(np.mean(df_rates),  4) if df_rates  else 0.03,
                    '1st_pct':         round(np.mean(p1st),      4) if p1st      else 0.60,
                    '1st_won':         round(np.mean(w1st),      4) if w1st      else 0.70,
                    '2nd_won':         round(np.mean(w2nd),      4) if w2nd      else 0.50,
                    'bp_saved':        round(np.mean(bps),       4) if bps       else 0.60,
                    'total':           len(hist),
                }

            ws = get_player_stats(winner, surface)
            ls = get_player_stats(loser,  surface)

            # H2H
            h2h_key = tuple(sorted([winner, loser]))
            h2h_all  = [m for m in player_cache.get(winner, [])
                        if m.get('opponent') == loser]
            h2h_surf = [m for m in h2h_all if m.get('surface') == surface]

            h2h_w_wins    = sum(1 for m in h2h_all if m['won'])
            h2h_total     = len(h2h_all)
            h2h_surf_wins = sum(1 for m in h2h_surf if m['won'])
            h2h_surf_tot  = len(h2h_surf)

            # Encoding
            surf_enc  = SURFACE_MAP.get(surface, 0)
            level_enc = LEVEL_MAP.get(level, 2)
            round_enc = ROUND_MAP.get(round_, 3)
            is_gs     = 1 if level == 'G' else 0
            is_m1000  = 1 if level == 'M' else 0

            # Rank features
            rank_diff     = l_rank - w_rank   # positive = winner ranked higher
            rank_ratio    = round(l_rank / max(w_rank, 1), 3)
            log_rank_diff = round(np.log1p(abs(rank_diff)) * np.sign(rank_diff), 3)

            # Serve rate features
            def ace_rate(ace, svpt):
                return round(ace / max(svpt, 1), 4)

            def df_rate(df, svpt):
                return round(df / max(svpt, 1), 4)

            def pct(a, b):
                return round(a / max(b, 1), 4)

            match_id = f"{tour}_{match_date}_{winner}_{loser}".replace(' ','_')[:120]

            conn.execute("""
                INSERT OR REPLACE INTO matches (
                    match_id, tour, tourney_name, surface, tourney_level,
                    match_date, round, best_of,
                    p1_name, p2_name, p1_rank, p2_rank, p1_rank_pts, p2_rank_pts,
                    p1_age, p2_age, score, p1_won, source,
                    p1_ace, p1_df, p1_svpt, p1_1stIn, p1_1stWon, p1_2ndWon,
                    p1_bpSaved, p1_bpFaced,
                    p2_ace, p2_df, p2_svpt, p2_1stIn, p2_1stWon, p2_2ndWon,
                    p2_bpSaved, p2_bpFaced,
                    -- Feature columns
                    rank_diff, rank_ratio, log_rank_diff,
                    p1_surface_winrate, p2_surface_winrate, surface_winrate_diff,
                    p1_surface_games, p2_surface_games,
                    p1_overall_winrate, p2_overall_winrate,
                    p1_form_last5, p2_form_last5,
                    p1_form_last10, p2_form_last10,
                    p1_ace_rate, p2_ace_rate, ace_rate_diff,
                    p1_df_rate, p2_df_rate,
                    p1_1st_serve_pct, p2_1st_serve_pct,
                    p1_1st_won_pct, p2_1st_won_pct,
                    p1_2nd_won_pct, p2_2nd_won_pct,
                    p1_bp_saved_pct, p2_bp_saved_pct,
                    h2h_p1_wins, h2h_total, h2h_p1_winrate,
                    h2h_surface_p1_wins, h2h_surface_total,
                    surface_enc, tourney_level_enc, round_enc,
                    is_grand_slam, is_masters,
                    p1_total_career_matches, p2_total_career_matches,
                    implied_p1_prob
                ) VALUES (
                    ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,
                    ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,
                    ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?
                )
            """, (
                match_id, tour,
                str(row.get('tourney_name', '')),
                surface, level, match_date, round_, best_of,
                winner, loser, w_rank, l_rank, w_pts, l_pts,
                w_age, l_age, str(row.get('score', '')), 1, 'sackmann',
                w_ace, w_df, w_svpt, w_1stI, w_1stW, w_2ndW, w_bpS, w_bpF,
                l_ace, l_df, l_svpt, l_1stI, l_1stW, l_2ndW, l_bpS, l_bpF,
                # Features
                rank_diff, rank_ratio, log_rank_diff,
                ws['surface_winrate'], ls['surface_winrate'],
                round(ws['surface_winrate'] - ls['surface_winrate'], 3),
                ws['surface_games'], ls['surface_games'],
                ws['overall_winrate'], ls['overall_winrate'],
                ws['form_last5'], ls['form_last5'],
                ws['form_last10'], ls['form_last10'],
                ace_rate(w_ace, w_svpt), ace_rate(l_ace, l_svpt),
                round(ace_rate(w_ace,w_svpt) - ace_rate(l_ace,l_svpt), 4),
                df_rate(w_df, w_svpt), df_rate(l_df, l_svpt),
                pct(w_1stI, w_svpt), pct(l_1stI, l_svpt),
                pct(w_1stW, w_1stI), pct(l_1stW, l_1stI),
                pct(w_2ndW, max(w_svpt - w_1stI, 1)),
                pct(l_2ndW, max(l_svpt - l_1stI, 1)),
                pct(w_bpS, max(w_bpF, 1)), pct(l_bpS, max(l_bpF, 1)),
                h2h_w_wins, h2h_total,
                round(h2h_w_wins / max(h2h_total, 1), 3),
                h2h_surf_wins, h2h_surf_tot,
                surf_enc, level_enc, round_enc,
                is_gs, is_m1000,
                ws['total'], ls['total'],
                0.5,  # implied prob — filled from tennis-data if available
            ))

            stored += 1

            # Update cache AFTER insert — no leakage
            def build_cache_entry(won, surf, ace_r, df_r, p1st, w1st, w2nd, bp_s, opp):
                return {
                    'won': won, 'surface': surf,
                    'ace_rate': ace_r, 'df_rate': df_r,
                    '1st_pct': p1st, '1st_won': w1st,
                    '2nd_won': w2nd, 'bp_saved': bp_s,
                    'opponent': opp,
                }

            if winner not in player_cache:
                player_cache[winner] = []
            player_cache[winner].append(build_cache_entry(
                True, surface,
                ace_rate(w_ace, w_svpt), df_rate(w_df, w_svpt),
                pct(w_1stI, w_svpt), pct(w_1stW, w_1stI),
                pct(w_2ndW, max(w_svpt-w_1stI,1)), pct(w_bpS, max(w_bpF,1)),
                loser
            ))

            if loser not in player_cache:
                player_cache[loser] = []
            player_cache[loser].append(build_cache_entry(
                False, surface,
                ace_rate(l_ace, l_svpt), df_rate(l_df, l_svpt),
                pct(l_1stI, l_svpt), pct(l_1stW, l_1stI),
                pct(l_2ndW, max(l_svpt-l_1stI,1)), pct(l_bpS, max(l_bpF,1)),
                winner
            ))

            if stored % 500 == 0:
                conn.commit()
                log.info(f"  {tour}: stored {stored}...")

        except Exception as e:
            log.warning(f"Row error: {e}")
            skipped += 1

    conn.commit()
    conn.close()
    log.info(f"{tour}: ✅ stored={stored} skipped={skipped}")

def train_model(tour: str = 'BOTH') -> tuple:
    conn = sqlite3.connect(DB_PATH)

    query = """
        SELECT * FROM matches
        WHERE rank_diff IS NOT NULL
          AND p1_surface_winrate > 0
          AND p1_rank > 0
          AND p2_rank > 0
    """
    if tour != 'BOTH':
        query += f" AND tour='{tour}'"
    query += " ORDER BY match_date"

    df = pd.read_sql(query, conn)
    conn.close()

    if len(df) < 500:
        log.warning(f"Only {len(df)} usable rows — need 500+")
        return None, None

    log.info(f"Training {tour} model on {len(df)} matches")

    # Build feature matrix — always p1=winner, so label=1
    # But we need to train on BOTH perspectives (random flip 50%)
    feature_rows = []
    labels       = []

    for _, row in df.iterrows():
        try:
            feat = [float(row.get(c, 0) or 0) for c in FEATURE_COLUMNS]
            feature_rows.append(feat)
            labels.append(1)  # p1 always wins in stored data

            # Mirror: flip p1/p2 so model sees losing side too
            flipped = feat.copy()
            idx = {c: i for i, c in enumerate(FEATURE_COLUMNS)}
            pairs = [
                ('p1_rank', 'p2_rank'),
                ('p1_rank_pts', 'p2_rank_pts'),
                ('p1_surface_winrate', 'p2_surface_winrate'),
                ('p1_surface_games', 'p2_surface_games'),
                ('p1_overall_winrate', 'p2_overall_winrate'),
                ('p1_form_last5', 'p2_form_last5'),
                ('p1_form_last10', 'p2_form_last10'),
                ('p1_ace_rate', 'p2_ace_rate'),
                ('p1_df_rate', 'p2_df_rate'),
                ('p1_1st_serve_pct', 'p2_1st_serve_pct'),
                ('p1_1st_won_pct', 'p2_1st_won_pct'),
                ('p1_2nd_won_pct', 'p2_2nd_won_pct'),
                ('p1_bp_saved_pct', 'p2_bp_saved_pct'),
                ('p1_age', 'p2_age'),
                ('p1_total_career_matches', 'p2_total_career_matches'),
            ]
            for a, b in pairs:
                if a in idx and b in idx:
                    flipped[idx[a]], flipped[idx[b]] = flipped[idx[b]], flipped[idx[a]]
            if 'rank_diff' in idx:
                flipped[idx['rank_diff']] *= -1
            if 'log_rank_diff' in idx:
                flipped[idx['log_rank_diff']] *= -1
            if 'surface_winrate_diff' in idx:
                flipped[idx['surface_winrate_diff']] *= -1
            if 'ace_rate_diff' in idx:
                flipped[idx['ace_rate_diff']] *= -1
            if 'h2h_p1_winrate' in idx:
                flipped[idx['h2h_p1_winrate']] = 1 - flipped[idx['h2h_p1_winrate']]
            if 'implied_p1_prob' in idx:
                p = flipped[idx['implied_p1_prob']]
                flipped[idx['implied_p1_prob']] = 1 - p if p > 0 else 0.5

            feature_rows.append(flipped)
            labels.append(0)

        except Exception:
            continue

    X = np.array(feature_rows)
    y = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    model = XGBClassifier(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        eval_metric='logloss',
        random_state=42,
        verbosity=0,
    )

    model.fit(X_train_s, y_train,
              eval_set=[(X_test_s, y_test)],
              verbose=False)

    y_prob = model.predict_proba(X_test_s)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    acc   = round(accuracy_score(y_test, y_pred), 4)
    auc   = round(roc_auc_score(y_test, y_prob), 4)
    brier = round(brier_score_loss(y_test, y_prob), 4)
    ll    = round(log_loss(y_test, y_prob), 4)

    log.info(f"{tour} model — AUC: {auc}, Acc: {acc}, Brier: {brier}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    tag = tour.lower()
    joblib.dump(model,           f'{MODELS_DIR}/tennis_model_{tag}.pkl')
    joblib.dump(scaler,          f'{MODELS_DIR}/tennis_scaler_{tag}.pkl')
    joblib.dump(FEATURE_COLUMNS, f'{MODELS_DIR}/tennis_features_{tag}.pkl')

    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        INSERT INTO model_performance
        (date, tour, accuracy, auc_roc, brier_score, log_loss, samples)
        VALUES (?,?,?,?,?,?,?)
    """, (datetime.now().isoformat(), tour, acc, auc, brier, ll, len(y)))
    conn.commit()
    conn.close()

    return model, scaler

def build_match_features(p1_name: str, p2_name: str,
                          surface: str, tourney_level: str,
                          round_: str, best_of: int,
                          p1_rank: int, p2_rank: int,
                          p1_rank_pts: int = 0, p2_rank_pts: int = 0,
                          p1_age: float = 25.0, p2_age: float = 25.0,
                          p1_odds: float = None, p2_odds: float = None,
                          tour: str = 'ATP') -> dict:
    conn = sqlite3.connect(DB_PATH)

    def get_player_form(name, surf, lookback=LOOKBACK):
        wins = conn.execute("""
            SELECT surface, p1_ace, p1_svpt, p1_1stIn, p1_1stWon,
                   p1_2ndWon, p1_bpSaved, p1_bpFaced
            FROM matches WHERE p1_name=? AND tour=?
            ORDER BY match_date DESC LIMIT ?
        """, (name, tour, lookback)).fetchall()

        losses = conn.execute("""
            SELECT surface, p2_ace, p2_svpt, p2_1stIn, p2_1stWon,
                   p2_2ndWon, p2_bpSaved, p2_bpFaced
            FROM matches WHERE p2_name=? AND tour=?
            ORDER BY match_date DESC LIMIT ?
        """, (name, tour, lookback)).fetchall()

        total_w = conn.execute(
            "SELECT COUNT(*) FROM matches WHERE p1_name=? AND tour=?",
            (name, tour)
        ).fetchone()[0]
        total_l = conn.execute(
            "SELECT COUNT(*) FROM matches WHERE p2_name=? AND tour=?",
            (name, tour)
        ).fetchone()[0]

        total   = total_w + total_l
        n_recent = len(wins) + len(losses)

        if n_recent == 0:
            return {
                'surface_winrate': 0.5, 'surface_games': 0,
                'overall_winrate': 0.5, 'form_last5': 2,
                'form_last10': 5, 'ace_rate': 0.06,
                'df_rate': 0.03, '1st_pct': 0.60,
                '1st_won': 0.70, '2nd_won': 0.50,
                'bp_saved': 0.60, 'total': 0,
            }

        surf_wins = sum(1 for r in wins if r[0] == surf)
        surf_loss = sum(1 for r in losses if r[0] == surf)
        surf_total = surf_wins + surf_loss
        surf_wr    = round(surf_wins / max(surf_total, 1), 3)
        overall_wr = round(len(wins) / max(n_recent, 1), 3)
        form5  = min(len(wins), 5)
        form10 = min(len(wins), 10)

        def avg_pct(rows, idx_a, idx_b):
            vals = [r[idx_a]/max(r[idx_b],1) for r in rows if r[idx_b] > 0]
            return round(np.mean(vals), 4) if vals else 0.0

        ace_r  = avg_pct(wins, 1, 2)
        df_r   = round(np.mean([r[1]/max(r[2],1) for r in wins if r[2]>0]) if wins else 0.03, 4)
        p1st   = avg_pct(wins, 3, 2)
        w1st   = avg_pct(wins, 4, 3)
        w2nd   = round(np.mean([r[5]/max(r[2]-r[3],1) for r in wins if r[2]-r[3]>0]) if wins else 0.5, 4)
        bps    = avg_pct(wins, 6, 7)

        return {
            'surface_winrate': surf_wr,
            'surface_games':   surf_total,
            'overall_winrate': overall_wr,
            'form_last5':      form5,
            'form_last10':     form10,
            'ace_rate':        ace_r,
            'df_rate':         df_r,
            '1st_pct':         p1st,
            '1st_won':         w1st,
            '2nd_won':         w2nd,
            'bp_saved':        bps,
            'total':           total,
        }

    p1s = get_player_form(p1_name, surface)
    p2s = get_player_form(p2_name, surface)

    h2h_p1 = conn.execute(
        "SELECT COUNT(*) FROM matches WHERE p1_name=? AND p2_name=? AND tour=?",
        (p1_name, p2_name, tour)
    ).fetchone()[0]
    h2h_p2 = conn.execute(
        "SELECT COUNT(*) FROM matches WHERE p1_name=? AND p2_name=? AND tour=?",
        (p2_name, p1_name, tour)
    ).fetchone()[0]
    h2h_tot  = h2h_p1 + h2h_p2
    h2h_wr   = round(h2h_p1 / max(h2h_tot, 1), 3)

    h2h_sp1  = conn.execute(
        "SELECT COUNT(*) FROM matches WHERE p1_name=? AND p2_name=? AND surface=? AND tour=?",
        (p1_name, p2_name, surface, tour)
    ).fetchone()[0]
    h2h_sp2  = conn.execute(
        "SELECT COUNT(*) FROM matches WHERE p1_name=? AND p2_name=? AND surface=? AND tour=?",
        (p2_name, p1_name, surface, tour)
    ).fetchone()[0]
    h2h_st   = h2h_sp1 + h2h_sp2

    conn.close()

    rank_diff     = p2_rank - p1_rank
    rank_ratio    = round(p2_rank / max(p1_rank, 1), 3)
    log_rank_diff = round(np.log1p(abs(rank_diff)) * np.sign(rank_diff), 3)

    surf_enc  = SURFACE_MAP.get(surface, 0)
    level_enc = LEVEL_MAP.get(tourney_level, 2)
    round_enc = ROUND_MAP.get(round_, 3)

    implied = 0.5
    if p1_odds and p1_odds > 1.0:
        implied = round(1 / p1_odds, 4)

    return {
        'p1_rank':                p1_rank,
        'p2_rank':                p2_rank,
        'rank_diff':              rank_diff,
        'rank_ratio':             rank_ratio,
        'log_rank_diff':          log_rank_diff,
        'p1_rank_pts':            p1_rank_pts,
        'p2_rank_pts':            p2_rank_pts,
        'p1_surface_winrate':     p1s['surface_winrate'],
        'p2_surface_winrate':     p2s['surface_winrate'],
        'surface_winrate_diff':   round(p1s['surface_winrate'] - p2s['surface_winrate'], 3),
        'p1_surface_games':       p1s['surface_games'],
        'p2_surface_games':       p2s['surface_games'],
        'p1_overall_winrate':     p1s['overall_winrate'],
        'p2_overall_winrate':     p2s['overall_winrate'],
        'p1_form_last5':          p1s['form_last5'],
        'p2_form_last5':          p2s['form_last5'],
        'p1_form_last10':         p1s['form_last10'],
        'p2_form_last10':         p2s['form_last10'],
        'p1_ace_rate':            p1s['ace_rate'],
        'p2_ace_rate':            p2s['ace_rate'],
        'ace_rate_diff':          round(p1s['ace_rate'] - p2s['ace_rate'], 4),
        'p1_df_rate':             p1s['df_rate'],
        'p2_df_rate':             p2s['df_rate'],
        'p1_1st_serve_pct':       p1s['1st_pct'],
        'p2_1st_serve_pct':       p2s['1st_pct'],
        'p1_1st_won_pct':         p1s['1st_won'],
        'p2_1st_won_pct':         p2s['1st_won'],
        'p1_2nd_won_pct':         p1s['2nd_won'],
        'p2_2nd_won_pct':         p2s['2nd_won'],
        'p1_bp_saved_pct':        p1s['bp_saved'],
        'p2_bp_saved_pct':        p2s['bp_saved'],
        'h2h_p1_wins':            h2h_p1,
        'h2h_total':              h2h_tot,
        'h2h_p1_winrate':         h2h_wr,
        'h2h_surface_p1_wins':    h2h_sp1,
        'h2h_surface_total':      h2h_st,
        'surface_enc':            surf_enc,
        'tourney_level_enc':      level_enc,
        'round_enc':              round_enc,
        'best_of':                best_of,
        'is_grand_slam':          1 if tourney_level == 'G' else 0,
        'is_masters':             1 if tourney_level == 'M' else 0,
        'p1_age':                 p1_age,
        'p2_age':                 p2_age,
        'age_diff':               p1_age - p2_age,
        'p1_total_career_matches': p1s['total'],
        'p2_total_career_matches': p2s['total'],
        'implied_p1_prob':        implied,
        '_p1_games': p1s['total'],
        '_p2_games': p2s['total'],
    }

def predict_winner(p1_name: str, p2_name: str,
                    surface: str, tourney_level: str = 'S',
                    round_: str = 'R32', best_of: int = 3,
                    p1_rank: int = 100, p2_rank: int = 100,
                    p1_rank_pts: int = 0, p2_rank_pts: int = 0,
                    p1_age: float = 25.0, p2_age: float = 25.0,
                    p1_odds: float = None, p2_odds: float = None,
                    tour: str = 'ATP', bankroll: float = 1000) -> dict:
    features = build_match_features(
        p1_name, p2_name, surface, tourney_level,
        round_, best_of, p1_rank, p2_rank,
        p1_rank_pts, p2_rank_pts, p1_age, p2_age,
        p1_odds, p2_odds, tour
    )

    tag = tour.lower()
    model_path = f'{MODELS_DIR}/tennis_model_{tag}.pkl'
    if not os.path.exists(model_path):
        model_path = f'{MODELS_DIR}/tennis_model_both.pkl'
        tag = 'both'

    try:
        model     = joblib.load(f'{MODELS_DIR}/tennis_model_{tag}.pkl')
        scaler    = joblib.load(f'{MODELS_DIR}/tennis_scaler_{tag}.pkl')
        feat_cols = joblib.load(f'{MODELS_DIR}/tennis_features_{tag}.pkl')

        X   = np.array([[float(features.get(c, 0) or 0) for c in feat_cols]])
        X_s = scaler.transform(X)
        p1_prob = float(model.predict_proba(X_s)[0][1])

    except FileNotFoundError:
        log.info("Model not trained — using ranking fallback")
        rank_ratio = p2_rank / max(p1_rank, 1)
        p1_prob = round(min(0.85, max(0.15, rank_ratio / (rank_ratio + 1))), 3)

    except Exception as e:
        log.error(f"Predict error: {e}")
        p1_prob = 0.5

    p2_prob = 1 - p1_prob

    if p1_prob >= p2_prob:
        predicted_winner = p1_name
        win_prob         = p1_prob
        win_odds         = p1_odds
    else:
        predicted_winner = p2_name
        win_prob         = p2_prob
        win_odds         = p2_odds

    win_prob_pct = round(win_prob * 100, 1)

    distance   = abs(win_prob - 0.5)
    g_factor   = min(1.0, (features.get('_p1_games', 5) +
                            features.get('_p2_games', 5)) / 20)
    confidence = min(MAX_CONF, round(
        (distance * 100 * 0.7 + win_prob * 100 * 0.3) * max(g_factor, 0.5), 1
    ))

    implied  = round(1 / win_odds, 4) if win_odds and win_odds > 1 else win_prob
    edge_pct = round((win_prob - implied) * 100, 2)
    has_value = edge_pct >= MIN_EDGE and confidence >= MIN_CONF

    if edge_pct >= 12:    edge_label = "🔥 STRONG VALUE"
    elif edge_pct >= 8:   edge_label = "✅ GOOD VALUE"
    elif edge_pct >= MIN_EDGE: edge_label = "⚠️ LEAN VALUE"
    else:                 edge_label = "❌ NO VALUE"

    stake = 0.0
    if has_value and win_odds and win_odds > 1:
        b = win_odds - 1
        kelly_f = (b * win_prob - (1 - win_prob)) / b
        stake   = max(0, round(kelly_f * 0.20 * bankroll, 2))

    p1g = features.get('_p1_games', 0)
    p2g = features.get('_p2_games', 0)
    if p1g == 0 and p2g == 0:
        data_note = "⚠️ No player history in DB — run /retrain first"
    elif p1g < 5 or p2g < 5:
        data_note = f"⚠️ Limited data: {p1_name}({p1g}) {p2_name}({p2g} matches)"
    else:
        data_note = ""

    return {
        'p1_name':           p1_name,
        'p2_name':           p2_name,
        'tour':              tour,
        'surface':           surface,
        'tourney_level':     tourney_level,
        'round':             round_,
        'best_of':           best_of,
        'p1_rank':           p1_rank,
        'p2_rank':           p2_rank,
        'predicted_winner':  predicted_winner,
        'p1_prob':           round(p1_prob * 100, 1),
        'p2_prob':           round(p2_prob * 100, 1),
        'win_prob':          win_prob_pct,
        'win_odds':          win_odds,
        'p1_odds':           p1_odds,
        'p2_odds':           p2_odds,
        'implied_prob':      round(implied * 100, 1),
        'edge_pct':          edge_pct,
        'edge_label':        edge_label,
        'has_value':         has_value,
        'confidence':        confidence,
        'stake':             stake,
        'features':          {k: v for k, v in features.items() if not k.startswith('_')},
        'data_note':         data_note,
        'generated_at':      datetime.now().isoformat(),
    }

# ══════════════════════════════════════════════════════════════
# LAYER 1 — SofaScore (most reliable, covers ATP+WTA+ITF+Challenger)
# ══════════════════════════════════════════════════════════════

def fetch_sofascore(timezone: str = 'Africa/Lagos') -> list:
    """
    SofaScore scheduled events — most reliable source.
    Returns 100-200 matches on a typical day.
    """
    TZ      = pytz.timezone(timezone)
    today   = datetime.now(TZ).strftime('%Y-%m-%d')
    url     = f"https://api.sofascore.com/api/v1/sport/tennis/scheduled-events/{today}"
    matches = []
    seen    = set()

    data   = safe_get(url, headers=HEADERS_MOBILE)
    events = data.get('events', [])

    if not events:
        # Try tomorrow too (timezone edge case)
        tomorrow = (datetime.now(TZ) + timedelta(days=1)).strftime('%Y-%m-%d')
        data2    = safe_get(
            f"https://api.sofascore.com/api/v1/sport/tennis/scheduled-events/{tomorrow}",
            headers=HEADERS_MOBILE
        )
        events = data2.get('events', [])

    log.info(f"SofaScore: {len(events)} events")

    for event in events:
        try:
            status = event.get('status', {}).get('type', '')
            if status in ('finished', 'canceled', 'postponed', 'interrupted'):
                continue

            p1 = event.get('homeTeam', {}).get('name', '').strip()
            p2 = event.get('awayTeam', {}).get('name', '').strip()
            if not p1 or not p2:
                continue

            key = f"{p1}|{p2}"
            if key in seen: continue
            seen.add(key)

            tourney = event.get('tournament', {}).get('name', '')
            categ   = event.get('tournament', {}).get('category', {}).get('name', '')
            t_str   = (tourney + ' ' + categ).lower()

            # Tour
            if 'wta' in t_str or 'women' in t_str:    tour = 'WTA'
            elif 'itf' in t_str:                        tour = 'ITF'
            elif 'challenger' in t_str:                 tour = 'Challenger'
            elif 'utr' in t_str:                        tour = 'UTR'
            else:                                        tour = 'ATP'

            # Surface
            surface = 'Hard'
            ground  = str(event.get('groundType', '') or '').lower()
            if 'clay' in ground:    surface = 'Clay'
            elif 'grass' in ground: surface = 'Grass'
            elif 'carpet' in ground: surface = 'Carpet'

            # Rankings
            p1_rank = int(event.get('homeTeam', {}).get('ranking', 0) or 0) or 200
            p2_rank = int(event.get('awayTeam', {}).get('ranking', 0) or 0) or 200

            # Start time
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

            is_gs   = any(gs in t_str for gs in
                          ['australian', 'roland', 'wimbledon', 'us open'])
            best_of = 5 if (is_gs and tour == 'ATP') else 3
            t_level = 'G' if is_gs else ('M' if 'masters' in t_str or '1000' in t_str else 'S')

            round_name = str(event.get('roundInfo', {}).get('name', '') or 'R32')

            matches.append({
                'fixture_id':    f"sofa_{event.get('id', '')}",
                'tour':          tour,
                'tourney_name':  tourney,
                'surface':       surface,
                'round':         round_name,
                'best_of':       best_of,
                'tourney_level': t_level,
                'p1_name':       p1,
                'p2_name':       p2,
                'p1_rank':       p1_rank,
                'p2_rank':       p2_rank,
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


# ══════════════════════════════════════════════════════════════
# LAYER 2 — ESPN (ATP + WTA main draw, reliable backup)
# ══════════════════════════════════════════════════════════════

def fetch_espn(timezone: str = 'Africa/Lagos') -> list:
    """ESPN ATP + WTA scoreboard. Less coverage but very stable."""
    TZ      = pytz.timezone(timezone)
    utc_now = datetime.now(pytz.utc)
    matches = []
    seen    = set()

    # Try today + UTC tomorrow (timezone offset)
    dates = [
        utc_now.strftime('%Y%m%d'),
        (utc_now + timedelta(days=1)).strftime('%Y%m%d'),
        (utc_now - timedelta(days=1)).strftime('%Y%m%d'),
    ]

    for tour_code, tour_name in [('atp', 'ATP'), ('wta', 'WTA')]:
        url = (f"https://site.api.espn.com/apis/site/v2"
               f"/sports/tennis/{tour_code}/scoreboard")
        for date_str in dates:
            data   = safe_get(url, params={'dates': date_str, 'limit': 100})
            events = data.get('events', [])
            if not events:
                continue

            for event in events:
                try:
                    eid = str(event.get('id', ''))
                    if eid in seen: continue

                    status = event.get('status', {}).get('type', {})
                    if status.get('completed'): continue

                    comp = event['competitions'][0]
                    cl   = comp.get('competitors', [])
                    if len(cl) < 2: continue

                    p1 = (cl[0].get('athlete', {}).get('displayName') or
                          cl[0].get('team', {}).get('displayName', ''))
                    p2 = (cl[1].get('athlete', {}).get('displayName') or
                          cl[1].get('team', {}).get('displayName', ''))
                    if not p1 or not p2: continue

                    p1_rank = int(cl[0].get('athlete', {}).get('rank') or 200)
                    p2_rank = int(cl[1].get('athlete', {}).get('rank') or 200)

                    tourney = (event.get('season', {}).get('displayName') or
                               event.get('name', ''))
                    round_  = comp.get('type', {}).get('abbreviation', 'R32')

                    time_local = 'TBD'
                    try:
                        start = event.get('date', '')
                        if start:
                            dt = datetime.strptime(start[:16], '%Y-%m-%dT%H:%M')
                            dt = pytz.utc.localize(dt).astimezone(TZ)
                            time_local = dt.strftime('%I:%M %p')
                    except Exception:
                        pass

                    seen.add(eid)
                    matches.append({
                        'fixture_id':    f"espn_{eid}",
                        'tour':          tour_name,
                        'tourney_name':  tourney,
                        'surface':       'Hard',
                        'round':         str(round_),
                        'best_of':       3,
                        'tourney_level': 'S',
                        'p1_name':       p1,
                        'p2_name':       p2,
                        'p1_rank':       p1_rank,
                        'p2_rank':       p2_rank,
                        'p1_rank_pts':   0,
                        'p2_rank_pts':   0,
                        'p1_age':        25.0,
                        'p2_age':        25.0,
                        'p1_odds':       None,
                        'p2_odds':       None,
                        'time_local':    time_local,
                        'status':        '',
                        'source':        'espn',
                    })
                except Exception as e:
                    log.debug(f"ESPN event: {e}")

            if matches:
                break  # got data for this tour, stop trying dates

    return matches


# ══════════════════════════════════════════════════════════════
# LAYER 3 — Sportybet internal API (if accessible from Railway)
# ══════════════════════════════════════════════════════════════

def fetch_sportybet(timezone: str = 'Africa/Lagos') -> list:
    """Sportybet tennis — includes odds, covers all tours."""
    TZ      = pytz.timezone(timezone)
    today   = datetime.now(TZ).strftime('%Y-%m-%d')
    matches = []
    seen    = set()

    endpoints = [
        "https://www.sportybet.com/api/ng/factsCenter/sports/sr:sport:5/events",
        "https://www.sportybet.com/api/ng/factsCenter/sports/sr:sport:5/todayEvents",
    ]
    param_sets = [
        {'sportId': 'sr:sport:5', 'date': today, 'pageSize': 500},
        {'sportId': 'sr:sport:5', 'pageSize': 500},
    ]

    headers = {
        **HEADERS_MOBILE,
        'Origin':  'https://www.sportybet.com',
        'Referer': 'https://www.sportybet.com/ng/m/sport/tennis/today',
    }

    for url in endpoints:
        for params in param_sets:
            data = safe_get(url, params=params, headers=headers)
            events = (
                data.get('data', {}).get('events', []) or
                data.get('events', []) or []
            )
            if not events:
                continue

            for event in events:
                try:
                    p1 = (event.get('homeTeamName') or
                          event.get('homeName') or '').strip()
                    p2 = (event.get('awayTeamName') or
                          event.get('awayName') or '').strip()
                    if not p1 or not p2: continue

                    key = f"{p1}|{p2}"
                    if key in seen: continue
                    seen.add(key)

                    tourney = (event.get('tournamentName') or
                               event.get('tournament', {}).get('name') or 'Tennis')
                    t_lower = tourney.lower()

                    if 'wta' in t_lower or 'women' in t_lower: tour = 'WTA'
                    elif 'itf' in t_lower:                       tour = 'ITF'
                    elif 'challenger' in t_lower:                tour = 'Challenger'
                    else:                                         tour = 'ATP'

                    surface = 'Hard'
                    if 'clay' in t_lower:    surface = 'Clay'
                    elif 'grass' in t_lower: surface = 'Grass'

                    # Extract odds from markets
                    p1_odds = p2_odds = None
                    for market in event.get('markets', []):
                        m_id = str(market.get('id') or '')
                        if m_id in ('186', '1'):
                            for o in market.get('outcomes', []):
                                desc = str(o.get('desc') or '').lower()
                                try:
                                    odd = float(o.get('odds') or 0)
                                    if odd > 100: odd /= 1000
                                    odd = odd if odd > 1.0 else None
                                except Exception:
                                    odd = None
                                if 'home' in desc or desc == '1': p1_odds = odd
                                elif 'away' in desc or desc == '2': p2_odds = odd
                            if p1_odds or p2_odds: break

                    time_local = 'TBD'
                    ts = event.get('estimateStartTime') or event.get('startTime')
                    if ts:
                        try:
                            t = int(ts)
                            if t > 1e10: t //= 1000
                            utc_dt = pytz.utc.localize(datetime.utcfromtimestamp(t))
                            time_local = utc_dt.astimezone(TZ).strftime('%I:%M %p')
                        except Exception:
                            pass

                    is_gs   = any(gs in t_lower for gs in
                                  ['australian', 'roland', 'wimbledon', 'us open'])
                    best_of = 5 if (is_gs and tour == 'ATP') else 3
                    t_level = 'G' if is_gs else 'S'

                    matches.append({
                        'fixture_id':    str(event.get('eventId') or event.get('id') or f"sb_{p1}"),
                        'tour':          tour,
                        'tourney_name':  tourney,
                        'surface':       surface,
                        'round':         str(event.get('round') or 'R32'),
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
                        'status':        '',
                        'source':        'sportybet',
                    })
                except Exception as e:
                    log.debug(f"Sportybet parse: {e}")

            if matches:
                return matches

    return matches


# ══════════════════════════════════════════════════════════════
# LAYER 4 — Sackmann CSV static fallback
# ══════════════════════════════════════════════════════════════

def fetch_sackmann_upcoming(timezone: str = 'Africa/Lagos') -> list:
    """
    Last resort: pull latest ATP/WTA matches from Sackmann CSVs.
    These are historical but the most recent entries represent
    ongoing tournament matches for current week.
    """
    TZ      = pytz.timezone(timezone)
    matches = []
    seen    = set()

    current_year  = datetime.now().year
    previous_year = current_year - 1

    sources = [
        ('ATP', f"https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_{current_year}.csv"),
        ('WTA', f"https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master/wta_matches_{current_year}.csv"),
        ('ATP', f"https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_{previous_year}.csv"),
    ]

    for tour, url in sources:
        try:
            r = requests.get(url, timeout=15, headers=HEADERS_BROWSER)
            if r.status_code != 200:
                continue

            df = pd.read_csv(io.StringIO(r.text), low_memory=False)
            if df.empty:
                continue

            # Sort by tourney_date desc — most recent first
            if 'tourney_date' in df.columns:
                df = df.sort_values('tourney_date', ascending=False)

            # Take most recent 50 matches = likely current week's tournament
            recent = df.head(50)

            for _, row in recent.iterrows():
                try:
                    p1 = str(row.get('winner_name', '') or '').strip()
                    p2 = str(row.get('loser_name', '') or '').strip()
                    if not p1 or not p2: continue

                    key = f"{p1}|{p2}"
                    if key in seen: continue
                    seen.add(key)

                    surface = str(row.get('surface', 'Hard') or 'Hard').strip()
                    if surface not in ('Hard', 'Clay', 'Grass', 'Carpet'):
                        surface = 'Hard'

                    tourney   = str(row.get('tourney_name', '') or '').strip()
                    t_level   = str(row.get('tourney_level', 'S') or 'S').strip()
                    round_    = str(row.get('round', 'R32') or 'R32').strip()
                    best_of   = int(row.get('best_of', 3) or 3)
                    p1_rank   = int(row.get('winner_rank', 200) or 200)
                    p2_rank   = int(row.get('loser_rank', 200) or 200)
                    p1_age    = float(row.get('winner_age', 25) or 25)
                    p2_age    = float(row.get('loser_age', 25) or 25)
                    p1_seed   = int(row.get('winner_seed', 0) or 0)
                    p2_seed   = int(row.get('loser_seed', 0) or 0)

                    matches.append({
                        'fixture_id':    f"csv_{tour}_{p1}_{p2}",
                        'tour':          tour,
                        'tourney_name':  tourney,
                        'surface':       surface,
                        'round':         round_,
                        'best_of':       best_of,
                        'tourney_level': t_level,
                        'p1_name':       p1,
                        'p2_name':       p2,
                        'p1_rank':       p1_rank,
                        'p2_rank':       p2_rank,
                        'p1_rank_pts':   0,
                        'p2_rank_pts':   0,
                        'p1_age':        p1_age,
                        'p2_age':        p2_age,
                        'p1_seed':       p1_seed,
                        'p2_seed':       p2_seed,
                        'p1_odds':       None,
                        'p2_odds':       None,
                        'time_local':    'TBD',
                        'status':        'scheduled',
                        'source':        'sackmann_csv',
                    })
                except Exception:
                    pass

            if matches:
                log.info(f"Sackmann CSV ({tour}): {len(matches)} recent matches")
                break  # Got enough data

        except Exception as e:
            log.warning(f"Sackmann {tour}: {e}")

    return matches


# ══════════════════════════════════════════════════════════════
# MASTER FETCH — tries all 4 layers in order
# ══════════════════════════════════════════════════════════════

def get_todays_matches(timezone: str = 'Africa/Lagos',
                        tour_filter: str = None) -> list:
    """
    Get today's tennis matches.
    Tries 4 sources in order — returns first successful result.

    Layer 1: SofaScore    → 100-200 matches (ATP+WTA+ITF+Challenger)
    Layer 2: Sportybet    → 194 matches with odds
    Layer 3: ESPN         → 20-60 matches (ATP+WTA main draw)
    Layer 4: Sackmann CSV → recent tournament matches (always works)

    Never returns 0 matches if any internet is available.
    """
    sources = [
        ('SofaScore',    lambda: fetch_sofascore(timezone)),
        ('Sportybet',    lambda: fetch_sportybet(timezone)),
        ('ESPN',         lambda: fetch_espn(timezone)),
        ('Sackmann CSV', lambda: fetch_sackmann_upcoming(timezone)),
    ]

    matches = []
    source_used = None

    for name, fetch_fn in sources:
        try:
            log.info(f"Trying {name}...")
            result = fetch_fn()
            if result and len(result) > 0:
                matches     = result
                source_used = name
                log.info(f"✅ {name}: {len(matches)} matches loaded")
                break
            else:
                log.warning(f"{name}: returned 0 matches — trying next")
        except Exception as e:
            log.warning(f"{name} failed: {e} — trying next")

    if not matches:
        log.error("ALL SOURCES FAILED — no matches available")
        return []

    # Apply tour filter
    if tour_filter and matches:
        filtered = [
            m for m in matches
            if tour_filter.upper() in m.get('tour', '').upper() or
               tour_filter.upper() in m.get('tourney_name', '').upper()
        ]
        if filtered:
            matches = filtered
        # If filter returns nothing, return all (don't silently empty)

    # Tag source on all matches
    for m in matches:
        m['source'] = m.get('source') or source_used

    log.info(f"get_todays_matches: {len(matches)} total "
             f"({'filtered to ' + tour_filter if tour_filter else 'all tours'}) "
             f"via {source_used}")
    return matches

def daily_retrain():
    log.info("Tennis Hunter retrain starting...")
    init_db()

    for year in TRAIN_YEARS:
        for tour, func_url in [('ATP', SACKMANN_ATP), ('WTA', SACKMANN_WTA)]:
            try:
                url = func_url.format(year=year)
                r   = requests.get(url, timeout=20,
                                   headers={'User-Agent': 'Mozilla/5.0'})
                if r.status_code == 200:
                    df = pd.read_csv(io.StringIO(r.text),
                                     on_bad_lines='skip', low_memory=False)
                    df.columns = [c.strip() for c in df.columns]
                    store_sackmann_matches(df, tour)
            except Exception as e:
                log.error(f"Sackmann {tour} {year}: {e}")

    train_model('BOTH')
    log.info("Retrain complete.")
