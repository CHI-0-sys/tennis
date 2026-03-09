# TENNIS HUNTER BOT
ATP + WTA match winner prediction bot with value edge detection and Telegram interface.

## FEATURES
- **Zero API Keys**: Uses Jeff Sackmann GitHub data + ESPN free API.
- **ML Engine**: XGBoost classifier trained on 50,000+ matches (1991–2025).
- **Value Betting**: Calculates edge vs bookie implied probability.
- **Kelly Criterion**: automated bet sizing based on confidence and edge.
- **Tracking**: ROI, win rate, and profit/loss logging.

## INSTALLATION

### 1. Prerequisites (Mac)
```bash
brew install libomp
pyenv install 3.12.6
pyenv shell 3.12.6
```

### 2. Setup
```bash
mkdir tennis_hunter && cd tennis_hunter
pip install -r requirements.txt
cp .env.example .env
```

### 3. Configuration
Edit `.env` and add:
- `TELEGRAM_TOKEN`: From @BotFather
- `CHAT_ID`: Run `/mychatid` in your bot to get this
- `BANKROLL`: Your starting balance (default 1000)

## FIRST RUN
1. Start the bot:
   ```bash
   python telegram_bot.py
   ```
2. In Telegram, run `/retrain`. This will:
   - Initialize the SQLite database.
   - Download ~50,000 ATP/WTA matches (2015-2025).
   - Train the XGBoost model (takes 3-5 mins).
3. Run `/status` to confirm the model is trained.
4. Run `/today` to get recommendations.

## COMMANDS
- `/today` — Today's top value picks
- `/match Player A vs Player B` — Predict a custom matchup
- `/record` — View your ROI and win rate
- `/pending` — List unsettled picks
- `/settle [id] [winner]` — Settle a pick manually
- `/retrain` — Refresh data and retrain model
- `/status` — Check DB and model health

## DATA SOURCES
- **Jeff Sackmann (GitHub)**: Historical ATP/WTA match data.
- **tennis-data.co.uk**: Historical odds data.
- **ESPN API**: Real-time fixtures and results.

---
**Disclaimer**: This bot is for educational/analytical purposes. Betting involves risk. Never bet more than you can afford to lose.
# tennis
