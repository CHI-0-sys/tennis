import sqlite3
from datetime import datetime
import numpy as np

DB_PATH = "tennis_hunter.db"

def log_pick(tour, match, tourney_name, surface, p1_name, p2_name,
              predicted_winner, win_prob, book_implied, edge_pct,
              confidence, winner_odds, stake):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        INSERT INTO prediction_log
        (created_at, match_date, tour, match, tourney_name, surface,
         p1_name, p2_name, predicted_winner, win_prob, book_implied,
         edge_pct, confidence, winner_odds, stake)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        datetime.now().isoformat(), datetime.now().strftime('%Y-%m-%d'),
        tour, match, tourney_name, surface,
        p1_name, p2_name, predicted_winner, win_prob, book_implied,
        edge_pct, confidence, winner_odds, stake
    ))
    conn.commit()
    conn.close()


def settle_pick(pick_id: int, actual_winner: str, bankroll: float = 1000) -> dict:
    conn = sqlite3.connect(DB_PATH)
    cols  = [d[0] for d in conn.execute("SELECT * FROM prediction_log LIMIT 0").description]
    pick  = conn.execute(
        "SELECT * FROM prediction_log WHERE id=?", (pick_id,)
    ).fetchone()
    if not pick:
        conn.close()
        return {}

    p     = dict(zip(cols, pick))
    won   = actual_winner.strip().lower() in p['predicted_winner'].lower()
    result = 'WIN' if won else 'LOSS'
    pnl    = (
        round(p['stake'] * (p['winner_odds'] - 1), 2) if won
        else -round(p['stake'], 2)
    )

    conn.execute("""
        UPDATE prediction_log
        SET result=?, actual_winner=?, profit_loss=?, settled_at=?
        WHERE id=?
    """, (result, actual_winner, pnl, datetime.now().isoformat(), pick_id))
    conn.commit()
    conn.close()
    return {'result': result, 'profit_loss': pnl}


def get_stats(days=None, tour=None, surface=None, last_n=None) -> dict:
    conn   = sqlite3.connect(DB_PATH)
    query  = "SELECT * FROM prediction_log WHERE result != 'PENDING'"
    params = []
    if tour:
        query += " AND tour=?"
        params.append(tour)
    if surface:
        query += " AND surface=?"
        params.append(surface)
    if days:
        query += " AND created_at >= datetime('now', ?)"
        params.append(f'-{days} days')
    query += " ORDER BY created_at DESC"
    if last_n:
        query += f" LIMIT {last_n}"

    rows = conn.execute(query, params).fetchall()
    cols = [d[0] for d in conn.execute("SELECT * FROM prediction_log LIMIT 0").description]
    conn.close()

    if not rows:
        return {'total': 0, 'wins': 0, 'losses': 0, 'win_rate': 0,
                'roi': 0, 'total_profit': 0, 'streak': 0}

    picks   = [dict(zip(cols, r)) for r in rows]
    wins    = sum(1 for p in picks if p['result'] == 'WIN')
    total   = len(picks)
    staked  = sum(p['stake'] for p in picks)
    profit  = sum(p['profit_loss'] for p in picks)
    streak  = 0
    if picks:
        for p in picks:
            if p['result'] == picks[0]['result']:
                streak += 1
            else:
                break

    return {
        'total':        total,
        'wins':         wins,
        'losses':       total - wins,
        'win_rate':     round(wins / total * 100, 1),
        'roi':          round(profit / staked * 100, 1) if staked else 0,
        'total_profit': round(profit, 2),
        'streak':       streak,
        'streak_type':  picks[0]['result'] if picks else '',
    }


def get_pending_picks() -> list:
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT * FROM prediction_log WHERE result='PENDING' ORDER BY created_at DESC"
    ).fetchall()
    cols = [d[0] for d in conn.execute("SELECT * FROM prediction_log LIMIT 0").description]
    conn.close()
    return [dict(zip(cols, r)) for r in rows]
