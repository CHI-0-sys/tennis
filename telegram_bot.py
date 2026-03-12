import os, asyncio, logging, sqlite3, json, pytz
from datetime import datetime, time
from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

import tennis_engine as engine
import tracker

load_dotenv()
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
BANKROLL       = float(os.environ.get("BANKROLL", "1000"))
TIMEZONE       = os.environ.get("TIMEZONE", "Africa/Lagos")
CHAT_ID        = os.environ.get("CHAT_ID", "")
TZ             = pytz.timezone(TIMEZONE)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                    level=logging.INFO)
log = logging.getLogger(__name__)

SURFACE_EMOJI = {'Hard': '🔵', 'Clay': '🟤', 'Grass': '🟢', 'Carpet': '⬜'}
TOUR_EMOJI    = {'ATP': '🎾', 'WTA': '🎾'}

def now_local():
    return datetime.now(TZ)

def utc_from_local(hour, minute=0):
    local_dt = TZ.localize(
        datetime.now().replace(hour=hour, minute=minute, second=0, microsecond=0)
    )
    return time(local_dt.astimezone(pytz.utc).hour,
                local_dt.astimezone(pytz.utc).minute,
                tzinfo=pytz.utc)

def confidence_emoji(c):
    if c >= 80:   return '🔥'
    elif c >= 70: return '✅'
    elif c >= 60: return '⚠️'
    else:         return '❌'

def fmt_prediction(p: dict) -> str:
    """Format a full tennis prediction card."""
    try:
        surf_e    = SURFACE_EMOJI.get(p.get('surface', 'Hard'), '🎾')
        feat      = p.get('features', {})
        h2h_tot   = int(feat.get('h2h_total', 0))
        h2h_wr    = feat.get('h2h_p1_winrate', 0.5)
        p1_sw     = feat.get('p1_surface_winrate', 0)
        p2_sw     = feat.get('p2_surface_winrate', 0)
        p1_form5  = int(feat.get('p1_form_last5', 0))
        p2_form5  = int(feat.get('p2_form_last5', 0))
        p1_games  = int(feat.get('p1_total_career_matches', 0))
        p2_games  = int(feat.get('p2_total_career_matches', 0))

        h2h_str = (
            f"H2H: {p['p1_name'].split()[-1]} leads "
            f"{int(h2h_wr*h2h_tot)}-{h2h_tot - int(h2h_wr*h2h_tot)}"
            if h2h_tot > 0 else "H2H: No meetings in DB"
        )

        stake_str = (
            f"💵 Suggested stake : *${p.get('stake', 0)}* of ${BANKROLL}"
            if p.get('stake', 0) > 0 else
            "💵 No stake (edge below threshold)"
        )

        data_note = p.get('data_note', '')
        note_line = f"\n⚠️ _{data_note}_" if data_note else ""

        p1_last = p.get('p1_name', '').split()[-1]
        p2_last = p.get('p2_name', '').split()[-1]

        return (
            f"🎾 *{p.get('p1_name')} vs {p.get('p2_name')}*\n"
            f"{p.get('tour','')} — {p.get('tourney_name','')} "
            f"{surf_e} {p.get('surface','')} — {p.get('round','')}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"🏆 *PREDICTED WINNER: {p.get('predicted_winner','?')}*\n"
            f"📊 Win probability  : *{p.get('win_prob',0)}%*\n"
            f"📖 Book implied     : *{p.get('implied_prob',0)}%*\n"
            f"📐 Edge             : *{p.get('edge_pct',0):+.1f}%* {p.get('edge_label','')}\n"
            f"\n"
            f"🎯 Confidence : *{p.get('confidence',0)}%* {confidence_emoji(p.get('confidence',0))}\n"
            f"💰 Winner odds: *{p.get('win_odds','N/A')}*\n"
            f"{stake_str}\n"
            f"\n"
            f"📈 *KEY FACTORS*\n"
            f"  Ranking         : #{p.get('p1_rank','?')} vs #{p.get('p2_rank','?')}\n"
            f"  {surf_e} Surface W% : {p1_last}: {p1_sw*100:.0f}% | {p2_last}: {p2_sw*100:.0f}%\n"
            f"  Form (last 5)   : {p1_last}: {p1_form5}/5 | {p2_last}: {p2_form5}/5\n"
            f"  Career matches  : {p1_last}: {p1_games} | {p2_last}: {p2_games}\n"
            f"  {h2h_str}\n"
            f"{note_line}\n"
            f"⚠️ Verify odds at Sportybet before betting"
        )
    except Exception as e:
        log.error(f"fmt_prediction error: {e}")
        return (
            f"🎾 *{p.get('p1_name','?')} vs {p.get('p2_name','?')}*\n"
            f"Winner: {p.get('predicted_winner','?')} "
            f"({p.get('win_prob','?')}%) | "
            f"Edge: {p.get('edge_pct','?')}%"
        )


def fmt_daily_summary(value_picks: list, total: int) -> str:
    if not value_picks:
        return (
            f"🎾 *TENNIS HUNTER — {now_local().strftime('%b %d, %Y')}*\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"📊 {total} matches analyzed\n"
            f"❌ No value picks today\n\n"
            f"All edges below {engine.MIN_EDGE}% threshold.\n"
            f"Best to sit out today 💤"
        )

    lines = [
        f"🎾 *TENNIS HUNTER — {now_local().strftime('%b %d, %Y')}*",
        f"━━━━━━━━━━━━━━━━━━━━",
        f"📊 {total} matches analyzed | 🎯 {len(value_picks)} value picks",
        f"",
        f"💎 *VALUE PICKS TODAY:*",
        f"",
    ]
    for i, p in enumerate(value_picks[:6], 1):
        surf_e = SURFACE_EMOJI.get(p.get('surface','Hard'), '🎾')
        e_sym  = "🔥" if p['edge_pct'] >= 10 else "✅"
        lines += [
            f"{i}. *{p['predicted_winner']}* to win",
            f"   vs {p['p1_name'] if p['predicted_winner']==p['p2_name'] else p['p2_name']}",
            f"   {p.get('tour','')} {surf_e} {p.get('surface','')} | "
            f"Prob: {p['win_prob']}% | Edge: {p['edge_pct']:+.1f}% {e_sym}",
            f"   Odds: {p.get('win_odds','?')} | Stake: ${p.get('stake',0)}",
            f"",
        ]

    if value_picks:
        best = value_picks[0]
        lines += [
            f"━━━━━━━━━━━━━━━━━━━━",
            f"🏆 *BEST PICK: {best['predicted_winner']}*",
            f"   Edge: {best['edge_pct']:+.1f}% | Stake: ${best.get('stake',0)}",
            f"━━━━━━━━━━━━━━━━━━━━",
            f"⚠️ Always verify odds at Sportybet before betting",
        ]

    return "\n".join(lines)


# ── Command handlers ────────────────────────────────────────────────────


async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        f"🎾 *TENNIS HUNTER*\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"Match Winner Prediction Bot\n"
        f"🌍 Timezone : *{TIMEZONE}*\n"
        f"💰 Bankroll : *${BANKROLL}*\n\n"
        f"📅 *Daily Schedule*\n"
        f"  7:00 AM → Model retrains\n"
        f"  11:00 AM → Today's picks sent\n"
        f"  9:00 AM → Yesterday's picks settled\n\n"
        f"📋 *Commands*\n"
        f"  /today — Today's value picks\n"
        f"  /schedule — Full daily schedule\n"
        f"  /match Alcaraz vs Djokovic | clay\n"
        f"  /odds Alcaraz vs Djokovic | clay | 1.85 | 2.10\n"
        f"  /atp, /wta, /itf, /challenger — Tour filters\n"
        f"  /player Alcaraz — player stats\n"
        f"  /h2h Alcaraz vs Djokovic\n"
        f"  /record — Win rate + ROI\n"
        f"  /pending — Unsettled picks\n"
        f"  /settle [id] [winner name]\n"
        f"  /retrain — Update model\n"
        f"  /status — DB + model health\n"
        f"  /mychatid\n\n"
        f"🎾 Covers ATP, WTA, Challenger, ITF, UTR\n"
        f"📦 50,000+ matches training data\n"
        f"💡 Run /retrain first to load all data",
        parse_mode='Markdown'
    )


async def cmd_today(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    msg = await update.message.reply_text(
        "🎾 Fetching today's ATP + WTA matches..."
    )
    try:
        matches = engine.get_todays_matches(TIMEZONE)
        if not matches:
            await msg.edit_text(
                "📅 No matches scheduled today.\n"
                "Use /match to predict a specific matchup."
            )
            return

        atp_count = sum(1 for m in matches if m['tour'] == 'ATP')
        wta_count = sum(1 for m in matches if m['tour'] == 'WTA')

        await msg.edit_text(
            f"🎾 *{len(matches)} matches today*\n"
            f"ATP: {atp_count} | WTA: {wta_count}\n"
            f"⏳ Running predictions...",
            parse_mode='Markdown'
        )

        predictions  = []
        value_picks  = []

        for m in matches:
            try:
                pred = engine.predict_winner(
                    p1_name=m['p1_name'], p2_name=m['p2_name'],
                    surface=m['surface'],
                    tourney_level=m.get('tourney_level', 'S'),
                    round_=m.get('round', 'R32'),
                    best_of=m.get('best_of', 3),
                    p1_rank=m.get('p1_rank', 100),
                    p2_rank=m.get('p2_rank', 100),
                    p1_odds=m.get('p1_odds'),
                    p2_odds=m.get('p2_odds'),
                    tour=m.get('tour', 'ATP'),
                    bankroll=BANKROLL
                )
                pred['fixture_id']   = m['fixture_id']
                pred['tourney_name'] = m.get('tourney_name', '')
                pred['time_local']   = m.get('time_local', 'TBD')
                predictions.append(pred)
                if pred['has_value']:
                    value_picks.append(pred)
            except Exception as e:
                log.error(f"Predict error: {e}")

        # Sort by edge
        value_picks.sort(key=lambda x: x['edge_pct'], reverse=True)

        await update.message.reply_text(
            fmt_daily_summary(value_picks, len(predictions)),
            parse_mode='Markdown'
        )

        # Detailed cards
        for p in value_picks[:5]:
            await asyncio.sleep(0.8)
            kb = [[InlineKeyboardButton(
                "📋 Log this pick",
                callback_data=f"log_{p['fixture_id']}"
            )]]
            await update.message.reply_text(
                fmt_prediction(p), parse_mode='Markdown',
                reply_markup=InlineKeyboardMarkup(kb)
            )

        if not value_picks and predictions:
            top3 = sorted(predictions, key=lambda x: x['win_prob'], reverse=True)[:3]
            await update.message.reply_text(
                "📊 *Top picks by probability (no value edge):*",
                parse_mode='Markdown'
            )
            for p in top3:
                await asyncio.sleep(0.8)
                await update.message.reply_text(
                    fmt_prediction(p), parse_mode='Markdown'
                )

        ctx.bot_data['today_predictions'] = {
            p['fixture_id']: p for p in predictions
        }

    except Exception as e:
        log.error(f"cmd_today: {e}", exc_info=True)
        await msg.edit_text(f"❌ Error: {str(e)[:200]}\nTry /retrain first.")


async def cmd_atp(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await _run_tour(update, ctx, 'ATP')

async def cmd_wta(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await _run_tour(update, ctx, 'WTA')

async def cmd_itf(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await _run_tour(update, ctx, 'ITF')

async def cmd_challenger(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await _run_tour(update, ctx, 'Challenger')


async def _run_tour(update: Update, ctx: ContextTypes.DEFAULT_TYPE,
                     tour: str):
    msg = await update.message.reply_text(
        f"🎾 Fetching {tour} matches..."
    )
    try:
        import sportybet_tennis as sb
        matches = sb.get_tennis_matches(TIMEZONE, tour_filter=tour)
        if not matches:
            await msg.edit_text(f"📅 No {tour} matches found today.")
            return

        await msg.edit_text(
            f"🎾 *{len(matches)} {tour} matches*\n"
            f"⏳ Predicting...",
            parse_mode='Markdown'
        )

        predictions = []
        value_picks = []
        for m in matches:
            try:
                pred = engine.predict_winner(
                    p1_name=m['p1_name'], p2_name=m['p2_name'],
                    surface=m['surface'],
                    tourney_level=m.get('tourney_level', 'S'),
                    round_=m.get('round', 'R32'),
                    best_of=m.get('best_of', 3),
                    p1_rank=m.get('p1_rank', 200),
                    p2_rank=m.get('p2_rank', 200),
                    p1_odds=m.get('p1_odds'),
                    p2_odds=m.get('p2_odds'),
                    tour=m.get('tour', 'ATP'),
                    bankroll=BANKROLL
                )
                pred['fixture_id']   = m['fixture_id']
                pred['tourney_name'] = m.get('tourney_name', '')
                pred['time_local']   = m.get('time_local', 'TBD')
                predictions.append(pred)
                if pred.get('has_value'):
                    value_picks.append(pred)
            except Exception as e:
                log.error(f"Predict {m.get('p1_name')}: {e}")

        value_picks.sort(key=lambda x: x.get('edge_pct', 0), reverse=True)
        await update.message.reply_text(
            fmt_daily_summary(value_picks, len(predictions)),
            parse_mode='Markdown'
        )
        for p in value_picks[:5]:
            await asyncio.sleep(0.8)
            kb = [[InlineKeyboardButton(
                "📋 Log", callback_data=f"log_{p['fixture_id']}"
            )]]
            await update.message.reply_text(
                fmt_prediction(p), parse_mode='Markdown',
                reply_markup=InlineKeyboardMarkup(kb)
            )
        ctx.bot_data['today_predictions'] = {
            p['fixture_id']: p for p in predictions
        }
    except Exception as e:
        log.error(f"_run_tour: {e}", exc_info=True)
        await msg.edit_text(f"❌ Error: {str(e)[:200]}")


async def cmd_schedule(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Full schedule grouped by tournament. /schedule or /schedule itf"""
    tour_filter = ctx.args[0].upper() if ctx.args else None
    msg = await update.message.reply_text("🎾 Loading schedule...")
    try:
        import sportybet_tennis as sb
        matches = sb.get_tennis_matches(TIMEZONE, tour_filter=tour_filter)
        if not matches:
            await msg.edit_text("📅 No matches found today.")
            return

        counts = {}
        for m in matches:
            t = m.get('tour', '?')
            counts[t] = counts.get(t, 0) + 1

        by_tourney = {}
        for m in matches:
            t = m.get('tourney_name', 'Unknown')
            by_tourney.setdefault(t, []).append(m)

        BADGE = {'ATP':'🔵','WTA':'🔴','ITF':'⚪','Challenger':'🟡','UTR':'🟢'}
        count_str = ' | '.join(f"{t}:{n}" for t, n in sorted(counts.items()))

        lines = [
            f"🎾 *TENNIS SCHEDULE — TODAY*",
            f"━━━━━━━━━━━━━━━━━━━━",
            f"📊 {len(matches)} matches | {count_str}",
            f"Source: {'Sportybet' if matches[0].get('source')=='sportybet' else 'SofaScore'}",
            f"",
        ]
        shown = 0
        for tourney, ms in sorted(by_tourney.items())[:15]:
            if shown >= 50: break
            lines.append(f"🏆 *{tourney}* ({len(ms)})")
            for m in ms[:5]:
                badge = BADGE.get(m.get('tour',''), '🎾')
                lines.append(
                    f"  {badge} {m['p1_name']} vs {m['p2_name']}\n"
                    f"     {m.get('surface','?')} | "
                    f"{m.get('round','?')} | {m.get('time_local','TBD')}"
                )
                shown += 1
            lines.append("")

        if len(matches) > shown:
            lines.append(f"_...{len(matches)-shown} more not shown_")
        lines += ["━━━━━━━━━━━━━━━━━━━━",
                  "/today /atp /wta /itf /challenger"]

        await msg.edit_text("\n".join(lines), parse_mode='Markdown')
    except Exception as e:
        await msg.edit_text(f"❌ Error: {e}")


async def cmd_match(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    text = ' '.join(ctx.args) if ctx.args else ''
    if ' vs ' not in text.lower():
        await update.message.reply_text(
            "Usage: `/match Alcaraz vs Djokovic | clay`\n"
            "Surfaces: hard, clay, grass",
            parse_mode='Markdown'
        )
        return
    try:
        parts   = text.split('|')
        match_p = parts[0].strip()
        surface = parts[1].strip().title() if len(parts) > 1 else 'Hard'
        if surface.lower() not in ('hard','clay','grass','carpet'):
            surface = 'Hard'

        players = match_p.lower().split(' vs ')
        p1 = players[0].strip().title()
        p2 = players[1].strip().title()

        msg  = await update.message.reply_text(
            f"⏳ Analyzing {p1} vs {p2} on {surface}..."
        )
        pred = engine.predict_winner(
            p1_name=p1, p2_name=p2,
            surface=surface.title(),
            tour='ATP', bankroll=BANKROLL
        )
        pred['fixture_id'] = f"manual_{p1}_{p2}"
        pred['tourney_name'] = 'Manual'
        pred['time_local'] = 'TBD'
        await msg.edit_text(fmt_prediction(pred), parse_mode='Markdown')

    except Exception as e:
        await update.message.reply_text(f"❌ Error: {e}")


async def cmd_odds(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    text = ' '.join(ctx.args) if ctx.args else ''
    parts = text.split('|')
    if len(parts) < 4 or ' vs ' not in parts[0].lower():
        await update.message.reply_text(
            "Usage: `/odds Alcaraz vs Djokovic | clay | 1.60 | 2.30`\n"
            "Last two numbers: p1 odds | p2 odds",
            parse_mode='Markdown'
        )
        return
    try:
        players = parts[0].strip().lower().split(' vs ')
        p1      = players[0].strip().title()
        p2      = players[1].strip().title()
        surface = parts[1].strip().title() or 'Hard'
        p1_odds = float(parts[2].strip())
        p2_odds = float(parts[3].strip())

        msg  = await update.message.reply_text(
            f"⏳ Analyzing with your odds..."
        )
        pred = engine.predict_winner(
            p1_name=p1, p2_name=p2, surface=surface,
            p1_odds=p1_odds, p2_odds=p2_odds,
            tour='ATP', bankroll=BANKROLL
        )
        pred['fixture_id']   = f"odds_{p1}_{p2}"
        pred['tourney_name'] = 'Custom Odds'
        pred['time_local']   = 'TBD'
        await msg.edit_text(fmt_prediction(pred), parse_mode='Markdown')

    except Exception as e:
        await update.message.reply_text(f"❌ Error: {e}")


async def cmd_player(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    name = ' '.join(ctx.args).title() if ctx.args else ''
    if not name:
        await update.message.reply_text("Usage: `/player Alcaraz`", parse_mode='Markdown')
        return
    try:
        conn = sqlite3.connect(engine.DB_PATH)
        total_w = conn.execute(
            "SELECT COUNT(*) FROM matches WHERE p1_name LIKE ?",
            (f'%{name}%',)
        ).fetchone()[0]
        total_l = conn.execute(
            "SELECT COUNT(*) FROM matches WHERE p2_name LIKE ?",
            (f'%{name}%',)
        ).fetchone()[0]

        surf_stats = {}
        for surf in ('Hard', 'Clay', 'Grass'):
            w = conn.execute(
                "SELECT COUNT(*) FROM matches WHERE p1_name LIKE ? AND surface=?",
                (f'%{name}%', surf)
            ).fetchone()[0]
            l = conn.execute(
                "SELECT COUNT(*) FROM matches WHERE p2_name LIKE ? AND surface=?",
                (f'%{name}%', surf)
            ).fetchone()[0]
            total_s = w + l
            surf_stats[surf] = (w, total_s, round(w/max(total_s,1)*100,1))

        last_rank = conn.execute(
            "SELECT p1_rank FROM matches WHERE p1_name LIKE ? ORDER BY match_date DESC LIMIT 1",
            (f'%{name}%',)
        ).fetchone()
        conn.close()

        total = total_w + total_l
        wr    = round(total_w / max(total, 1) * 100, 1)

        surf_lines = "\n".join([
            f"  {SURFACE_EMOJI.get(s,'🎾')} {s}: {v[0]}W/{v[1]}T ({v[2]}%)"
            for s, v in surf_stats.items() if v[1] > 0
        ]) or "  No surface data"

        rank_str = f"#{last_rank[0]}" if last_rank else "Unknown"

        await update.message.reply_text(
            f"🎾 *{name}*\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"📊 Career (in DB): {total_w}W / {total_l}L ({wr}%)\n"
            f"📈 Last known rank: {rank_str}\n"
            f"\n"
            f"🏟️ *Surface breakdown:*\n{surf_lines}\n"
            f"\n"
            f"Data from Sackmann ATP/WTA dataset",
            parse_mode='Markdown'
        )
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {e}")


async def cmd_h2h(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    text = ' '.join(ctx.args) if ctx.args else ''
    if ' vs ' not in text.lower():
        await update.message.reply_text(
            "Usage: `/h2h Alcaraz vs Djokovic`", parse_mode='Markdown'
        )
        return
    players = text.lower().split(' vs ')
    p1 = players[0].strip().title()
    p2 = players[1].strip().title()

    try:
        conn = sqlite3.connect(engine.DB_PATH)
        p1_wins = conn.execute(
            "SELECT COUNT(*), GROUP_CONCAT(tourney_name||' '||round||' '||score, ' | ') "
            "FROM matches WHERE p1_name LIKE ? AND p2_name LIKE ? LIMIT 10",
            (f'%{p1}%', f'%{p2}%')
        ).fetchone()
        p2_wins = conn.execute(
            "SELECT COUNT(*) FROM matches WHERE p1_name LIKE ? AND p2_name LIKE ?",
            (f'%{p2}%', f'%{p1}%')
        ).fetchone()[0]
        conn.close()

        p1w   = p1_wins[0] if p1_wins else 0
        total = p1w + p2_wins

        if total == 0:
            await update.message.reply_text(
                f"❌ No H2H data found for {p1} vs {p2} in DB.\n"
                f"Run /retrain to load more data."
            )
            return

        recent = p1_wins[1] or ''

        await update.message.reply_text(
            f"🎾 *H2H: {p1} vs {p2}*\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"  {p1}: {p1w} wins\n"
            f"  {p2}: {p2_wins} wins\n"
            f"  Total meetings: {total}\n\n"
            f"📋 Recent {p1} wins:\n{recent[:400] if recent else 'None found'}",
            parse_mode='Markdown'
        )
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {e}")


async def cmd_record(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    try:
        arg     = ctx.args[0] if ctx.args else None
        tour    = arg.upper() if arg in ('atp','wta','ATP','WTA') else None
        last_n  = int(arg) if arg and arg.isdigit() else None
        stats   = tracker.get_stats(tour=tour, last_n=last_n)
        title   = f"📊 TENNIS HUNTER RECORD"
        if tour:   title += f" — {tour}"
        if last_n: title += f" (Last {last_n})"

        streak_str = (
            f"{stats['streak']}x {stats['streak_type']}"
            if stats.get('streak', 0) > 1 else "-"
        )
        roi_e = "📈" if stats.get('roi', 0) >= 0 else "📉"

        if stats['total'] == 0:
            await update.message.reply_text("No settled picks yet.")
            return

        await update.message.reply_text(
            f"{title}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"📊 Total picks  : {stats['total']}\n"
            f"✅ Wins         : {stats['wins']}\n"
            f"❌ Losses       : {stats['losses']}\n"
            f"🎯 Win rate     : {stats['win_rate']}%\n"
            f"{roi_e} ROI          : {stats['roi']}%\n"
            f"💰 Total P&L    : ${stats['total_profit']:+.2f}\n"
            f"🔥 Streak       : {streak_str}\n"
            f"━━━━━━━━━━━━━━━━━━━━",
            parse_mode='Markdown'
        )
    except Exception as e:
        await update.message.reply_text(f"❌ Record error: {e}")


async def cmd_pending(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    picks = tracker.get_pending_picks()
    if not picks:
        await update.message.reply_text("✅ No pending picks.")
        return
    lines = [f"⏳ *PENDING PICKS ({len(picks)})*\n"]
    for p in picks[:10]:
        lines.append(
            f"ID *{p['id']}*: {p['match']}\n"
            f"  Pick: {p['predicted_winner']} | "
            f"Prob: {p['win_prob']}% | "
            f"Odds: {p['winner_odds']} | "
            f"Stake: ${p['stake']}\n"
        )
    lines.append("Use `/settle [id] [winner name]` to settle")
    await update.message.reply_text("\n".join(lines), parse_mode='Markdown')


async def cmd_settle(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not ctx.args or len(ctx.args) < 2:
        await update.message.reply_text(
            "Usage: `/settle 5 Alcaraz`\n"
            "Type the actual winner's last name",
            parse_mode='Markdown'
        )
        return
    try:
        pick_id = int(ctx.args[0])
        winner  = ' '.join(ctx.args[1:]).title()
        result  = tracker.settle_pick(pick_id, winner, BANKROLL)
        if not result:
            await update.message.reply_text(f"❌ Pick {pick_id} not found.")
            return
        emoji = "✅" if result['result'] == 'WIN' else "❌"
        await update.message.reply_text(
            f"{emoji} Pick {pick_id}: *{result['result']}*\n"
            f"Winner: {winner}\n"
            f"P&L: ${result['profit_loss']:+.2f}",
            parse_mode='Markdown'
        )
    except Exception as e:
        await update.message.reply_text(f"❌ Settle error: {e}")


async def cmd_retrain(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    msg = await update.message.reply_text(
        "🎾 *Tennis Hunter Retrain*\n"
        "⏳ Step 1/3: Fixing DB schema...",
        parse_mode='Markdown'
    )
    try:
        engine.init_db()

        await msg.edit_text(
            "🎾 *Tennis Hunter Retrain*\n"
            "✅ Step 1/3: DB ready\n"
            f"⏳ Step 2/3: Downloading Sackmann data "
            f"({engine.TRAIN_YEARS[0]}–{engine.TRAIN_YEARS[-1]})...",
            parse_mode='Markdown'
        )

        stored_atp = stored_wta = 0
        for year in engine.TRAIN_YEARS:
            for tour_name, url_template in [
                ('ATP', engine.SACKMANN_ATP),
                ('WTA', engine.SACKMANN_WTA)
            ]:
                try:
                    import io, requests as req
                    url = url_template.format(year=year)
                    r   = req.get(url, timeout=15,
                                  headers={'User-Agent': 'Mozilla/5.0'})
                    if r.status_code == 200:
                        import pandas as _pd
                        df = _pd.read_csv(io.StringIO(r.text),
                                          on_bad_lines='skip', low_memory=False)
                        df.columns = [c.strip() for c in df.columns]
                        engine.store_sackmann_matches(df, tour_name)
                        if tour_name == 'ATP':
                            stored_atp += len(df)
                        else:
                            stored_wta += len(df)
                except Exception as e:
                    log.warning(f"{tour_name} {year}: {e}")

        conn = sqlite3.connect(engine.DB_PATH)
        total = conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
        usable = conn.execute(
            "SELECT COUNT(*) FROM matches WHERE rank_diff IS NOT NULL AND p1_rank > 0"
        ).fetchone()[0]
        conn.close()

        await msg.edit_text(
            f"🎾 *Tennis Hunter Retrain*\n"
            f"✅ Step 1/3: DB ready\n"
            f"✅ Step 2/3: {total:,} matches stored\n"
            f"   ATP: ~{stored_atp:,} | WTA: ~{stored_wta:,}\n"
            f"   Usable rows: {usable:,}\n"
            f"⏳ Step 3/3: Training XGBoost model...",
            parse_mode='Markdown'
        )

        model, _ = engine.train_model('BOTH')

        conn = sqlite3.connect(engine.DB_PATH)
        mp = conn.execute(
            "SELECT auc_roc, accuracy, brier_score, samples "
            "FROM model_performance ORDER BY date DESC LIMIT 1"
        ).fetchone()
        conn.close()

        if model and mp:
            model_line = (
                f"✅ Model trained\n"
                f"   AUC: {mp[0]:.3f} | "
                f"Acc: {mp[1]:.1%} | "
                f"Samples: {mp[3]:,}"
            )
        elif model:
            model_line = "✅ Model trained"
        else:
            model_line = f"⚠️ Need 500+ matches — have {usable}"

        await msg.edit_text(
            f"{'✅' if model else '⚠️'} *Retrain Complete!*\n\n"
            f"📦 Total matches   : {total:,}\n"
            f"✅ Usable rows     : {usable:,}\n"
            f"🤖 {model_line}\n\n"
            f"Run /today to get predictions.\n"
            f"Expected accuracy: 68–73% on ATP matches.",
            parse_mode='Markdown'
        )

    except Exception as e:
        log.error(f"Retrain error: {e}", exc_info=True)
        await msg.edit_text(f"❌ Retrain error:\n`{str(e)[:300]}`",
                            parse_mode='Markdown')


async def cmd_status(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    try:
        conn    = sqlite3.connect(engine.DB_PATH)
        total   = conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
        usable  = conn.execute(
            "SELECT COUNT(*) FROM matches WHERE rank_diff IS NOT NULL AND p1_rank > 0"
        ).fetchone()[0]
        atp_n   = conn.execute(
            "SELECT COUNT(*) FROM matches WHERE tour='ATP'"
        ).fetchone()[0]
        wta_n   = conn.execute(
            "SELECT COUNT(*) FROM matches WHERE tour='WTA'"
        ).fetchone()[0]
        latest  = conn.execute("SELECT MAX(match_date) FROM matches").fetchone()[0]
        pending = conn.execute(
            "SELECT COUNT(*) FROM prediction_log WHERE result='PENDING'"
        ).fetchone()[0]
        mp      = conn.execute(
            "SELECT auc_roc, accuracy, samples, date "
            "FROM model_performance ORDER BY date DESC LIMIT 1"
        ).fetchone()
        conn.close()

        import os
        model_ok = os.path.exists(f'{engine.MODELS_DIR}/tennis_model_both.pkl')
        model_str = (
            f"✅ Trained | AUC: {mp[0]:.3f} | "
            f"Acc: {mp[1]:.1%} | {mp[2]:,} samples"
            if mp and model_ok else
            "❌ Not trained — run /retrain"
        )

        await update.message.reply_text(
            f"📊 *TENNIS HUNTER STATUS*\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"📦 Total matches   : {total:,}\n"
            f"✅ Usable rows     : {usable:,}\n"
            f"🎾 ATP             : {atp_n:,}\n"
            f"🎾 WTA             : {wta_n:,}\n"
            f"📅 Latest data     : {latest or 'None'}\n"
            f"⏳ Pending picks   : {pending}\n"
            f"🤖 Model           : {model_str}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"🌍 Timezone  : {TIMEZONE}\n"
            f"💰 Bankroll  : ${BANKROLL}\n"
            f"🕐 Local     : {datetime.now(TZ).strftime('%I:%M %p %Z')}",
            parse_mode='Markdown'
        )
    except Exception as e:
        await update.message.reply_text(f"Status error: {e}")


async def cmd_mychatid(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    cid = update.effective_chat.id
    await update.message.reply_text(
        f"Your Chat ID: `{cid}`\nAdd to .env: `CHAT_ID={cid}`",
        parse_mode='Markdown'
    )


async def cmd_bankroll(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    global BANKROLL
    if not ctx.args:
        await update.message.reply_text(
            f"Current bankroll: *${BANKROLL}*\nUsage: `/bankroll 2000`",
            parse_mode='Markdown'
        )
        return
    try:
        BANKROLL = float(ctx.args[0])
        await update.message.reply_text(
            f"✅ Bankroll updated to *${BANKROLL}*", parse_mode='Markdown'
        )
    except Exception:
        await update.message.reply_text("Usage: `/bankroll 2000`",
                                        parse_mode='Markdown')


async def handle_callback(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data  = query.data

    if data.startswith('log_'):
        fixture_id = data[4:]
        preds      = ctx.bot_data.get('today_predictions', {})
        pred       = preds.get(fixture_id)
        if not pred:
            await query.message.reply_text("❌ Pick expired — run /today again")
            return
        try:
            tracker.log_pick(
                tour=pred.get('tour','ATP'),
                match=f"{pred['p1_name']} vs {pred['p2_name']}",
                tourney_name=pred.get('tourney_name',''),
                surface=pred.get('surface','Hard'),
                p1_name=pred['p1_name'],
                p2_name=pred['p2_name'],
                predicted_winner=pred['predicted_winner'],
                win_prob=pred['win_prob'],
                book_implied=pred.get('implied_prob', 0),
                edge_pct=pred['edge_pct'],
                confidence=pred['confidence'],
                winner_odds=pred.get('win_odds') or 1.0,
                stake=pred.get('stake', 0),
            )
            await query.message.reply_text(
                f"✅ Pick logged!\n"
                f"*{pred['predicted_winner']}* to win\n"
                f"Use `/pending` to see all picks\n"
                f"After match: `/settle [id] {pred['predicted_winner'].split()[-1]}`",
                parse_mode='Markdown'
            )
        except Exception as e:
            await query.message.reply_text(f"❌ Log error: {e}")


# Scheduled jobs
async def job_retrain(context: ContextTypes.DEFAULT_TYPE):
    engine.daily_retrain()
    if CHAT_ID:
        await context.bot.send_message(CHAT_ID, "✅ Tennis Hunter model retrained")


async def job_daily_picks(context: ContextTypes.DEFAULT_TYPE):
    if not CHAT_ID:
        return
    try:
        matches     = engine.get_todays_matches(TIMEZONE)
        predictions = []
        for m in matches:
            try:
                pred = engine.predict_winner(
                    p1_name=m['p1_name'], p2_name=m['p2_name'],
                    surface=m['surface'],
                    tourney_level=m.get('tourney_level','S'),
                    round_=m.get('round','R32'),
                    best_of=m.get('best_of',3),
                    p1_rank=m.get('p1_rank',100),
                    p2_rank=m.get('p2_rank',100),
                    tour=m.get('tour','ATP'),
                    bankroll=BANKROLL
                )
                pred['fixture_id']   = m['fixture_id']
                pred['tourney_name'] = m.get('tourney_name','')
                predictions.append(pred)
            except Exception:
                pass

        value_picks = sorted(
            [p for p in predictions if p.get('has_value')],
            key=lambda x: x['edge_pct'], reverse=True
        )
        await context.bot.send_message(
            CHAT_ID,
            fmt_daily_summary(value_picks, len(predictions)),
            parse_mode='Markdown'
        )
        for p in value_picks[:3]:
            await asyncio.sleep(1)
            await context.bot.send_message(
                CHAT_ID, fmt_prediction(p), parse_mode='Markdown'
            )
    except Exception as e:
        log.error(f"Daily picks job error: {e}")


async def job_remind_settle(context: ContextTypes.DEFAULT_TYPE):
    if not CHAT_ID:
        return
    picks = tracker.get_pending_picks()
    if picks:
        await context.bot.send_message(
            CHAT_ID,
            f"⏳ {len(picks)} picks need settling.\n"
            f"Use `/settle [id] [winner name]`"
        )


def main():
    if not TELEGRAM_TOKEN:
        print("❌ TELEGRAM_TOKEN not set in .env")
        return

    engine.init_db()

    app = Application.builder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start",    cmd_start))
    app.add_handler(CommandHandler("today",    cmd_today))
    app.add_handler(CommandHandler("atp",      cmd_atp))
    app.add_handler(CommandHandler("wta",      cmd_wta))
    app.add_handler(CommandHandler("itf",        cmd_itf))
    app.add_handler(CommandHandler("challenger", cmd_challenger))
    app.add_handler(CommandHandler("schedule",   cmd_schedule))
    app.add_handler(CommandHandler("match",    cmd_match))
    app.add_handler(CommandHandler("odds",     cmd_odds))
    app.add_handler(CommandHandler("player",   cmd_player))
    app.add_handler(CommandHandler("h2h",      cmd_h2h))
    app.add_handler(CommandHandler("record",   cmd_record))
    app.add_handler(CommandHandler("pending",  cmd_pending))
    app.add_handler(CommandHandler("settle",   cmd_settle))
    app.add_handler(CommandHandler("retrain",  cmd_retrain))
    app.add_handler(CommandHandler("status",   cmd_status))
    app.add_handler(CommandHandler("mychatid", cmd_mychatid))
    app.add_handler(CommandHandler("bankroll", cmd_bankroll))
    app.add_handler(CallbackQueryHandler(handle_callback))

    jq = app.job_queue
    if jq:
        jq.run_daily(job_retrain,      utc_from_local(7,  0))
        jq.run_daily(job_daily_picks,  utc_from_local(11, 0))
        jq.run_daily(job_remind_settle, utc_from_local(9, 0))

    log.info("🎾 TENNIS HUNTER started. All commands registered.")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
