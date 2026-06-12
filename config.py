# ============================================================
# DeltaSignalBot — Configuration
# Fill in TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID before running
# ============================================================

# --- Logging ---
LOG_LEVEL        = "INFO"          # DEBUG | INFO | WARNING | ERROR
LOG_TO_FILE      = True
LOG_DIR          = "logs"
LOG_FILE         = "logs/bot.log"
LOG_ERROR_FILE   = "logs/errors.log"
LOG_MAX_BYTES    = 5 * 1024 * 1024   # 5 MB per file
LOG_BACKUP_COUNT = 5                  # keep last 5 rotated files

# --- Telegram ---
TELEGRAM_BOT_TOKEN = "8800960300:AAExwJ0q7WkWipz2RkvKw9-4Y28ePM5TAUc"   # From @BotFather
TELEGRAM_CHAT_IDS  = ["1948356544", "7151542571"]   # Add more chat IDs here

# --- Account ---
ACCOUNT_BALANCE    = 100  # USD — used for lot-size calculation

# --- Symbols ---
SYMBOLS = ["BTCUSD", "ETHUSD", "XAUTUSD"]

# --- Signal filters ---
MIN_RISK_REWARD  = 2.0   # Minimum R:R to send a signal
MAX_RISK_PERCENT = 1.0   # Maximum % of balance to risk per trade

# --- Alert stages to send ---
# Set to False to suppress WARNING and PREPARE — only full SIGNAL with SL/TP
SEND_WARNING_ALERTS = False
SEND_PREPARE_ALERTS = False

# --- Delta Exchange ---
BASE_URL      = "https://api.india.delta.exchange"
WS_URL        = "wss://socket.india.delta.exchange"
REQUEST_TIMEOUT = 15     # seconds

# --- Timeframes (resolution in minutes as used by the API) ---
TF_4H  = 240
TF_1H  = 60
TF_15M = 15

CANDLES_4H  = 100
CANDLES_1H  = 50
CANDLES_15M = 30

# --- S/R detection ---
SWING_LOOKBACK     = 5      # candles each side for swing pivot
CLUSTER_THRESHOLD  = 0.003  # 0.3 % — merge levels within this range
MAX_SR_LEVELS      = 3      # keep top-N support and resistance levels

# --- ATR multipliers for alert stages ---
ATR_WARNING_MULT = 2.0
ATR_PREPARE_MULT = 0.5

# --- Scheduler ---
CHECK_INTERVAL_MINUTES = 5   # run every N minutes
RESET_TRACKER_HOURS    = 4   # clear duplicate tracker every N hours

# --- Exchange API keys (leave blank to use public data only) ---
DELTA_API_KEY    = ""
DELTA_API_SECRET = ""
BINANCE_API_KEY  = ""
BINANCE_API_SECRET = ""
BYBIT_API_KEY    = ""
BYBIT_API_SECRET = ""
ACTIVE_EXCHANGE  = "delta"   # delta | binance | bybit

# --- TradingView webhook ---
TRADINGVIEW_SECRET = ""

# --- Trading mode ---
PAPER_TRADING_MODE    = True   # True = paper, False = live (be careful!)
AUTO_TRADE            = False  # True = auto-place limit orders on Delta Exchange when signal fires
                               # Requires PAPER_TRADING_MODE = False and DELTA_API_KEY filled in
MAX_CONTRACTS_PER_TRADE = 5   # Hard cap on live order size (1 contract = $1 notional on Delta)
                               # 5 contracts = $5 max exposure. Increase only when confident.
MAX_DAILY_LOSS_PERCENT = 3.0
MAX_DRAWDOWN_PERCENT   = 10.0
USE_AI_ENGINE          = False  # flip True when ready

# --- Database ---
DATABASE_URL = "sqlite:///./data/tradesignal.db"

# --- Notifications (optional) ---
DISCORD_WEBHOOK_URL = ""
EMAIL_SMTP_HOST     = ""
EMAIL_SMTP_PORT     = 587
EMAIL_USERNAME      = ""
EMAIL_PASSWORD      = ""
EMAIL_RECIPIENT     = ""

# --- News sentiment ---
# Alpha Vantage free key → https://www.alphavantage.co/support/#api-key
# Leave blank to use CryptoPanic only (BTC/ETH); gold news will be skipped.
ALPHA_VANTAGE_KEY      = "J7SYW39BC8EUC9V5"
NEWS_SENTIMENT_HOURS   = 6    # scan headlines from last N hours
NEWS_CACHE_MINUTES     = 30   # cache results to stay within free-tier limits
