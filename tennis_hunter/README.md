# TENNIS HUNTER BOT

ATP + WTA match winner prediction bot using Jeff Sackmann data and XGBoost.

## Features
- **Zero API Keys**: Uses public data from GitHub and ESPN.
- **Deep Training**: Trained on 50,000+ matches (2015–2025).
- **Live Fixtures**: Fetches daily matches from ESPN.
- **ML Predictions**: Uses XGBoost for win probability and edge calculation.
- **Bankroll Management**: Kelly Criterion stakeholder calculation.
- **Pick Tracking**: Log, settle, and track ROI/win rate.

## One-Line Install

```bash
# Mac
brew install libomp && pyenv shell 3.12.6 && pip install requests pandas numpy scipy scikit-learn xgboost joblib pytz "python-telegram-bot[job-queue]" python-dotenv && echo "✅ Tennis Hunter ready"

# Linux / Railway
pyenv shell 3.12.6 && pip install requests pandas numpy scipy scikit-learn xgboost joblib pytz "python-telegram-bot[job-queue]" python-dotenv && echo "✅ Tennis Hunter ready"
```

## Setup
1. Create a Telegram bot via [@BotFather](https://t.me/BotFather) and get your token.
2. Copy `.env.example` to `.env` and add your `TELEGRAM_TOKEN`.
3. Run the bot: `python telegram_bot.py`.
4. In Telegram:
   - `/mychatid` to get your ID, then add it to `.env` as `CHAT_ID`.
   - `/retrain` to download data and train the model (first run only).
   - `/today` to see predictions.

## File Structure
- `tennis_engine.py`: Data fetching, feature engineering, and ML model.
- `tracker.py`: Performance tracking and database management.
- `telegram_bot.py`: Telegram interface and scheduled jobs.
- `requirements.txt`, `Procfile`, `runtime.txt`: Deployment configurations.
