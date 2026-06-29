# ============================================================
# DeltaSignalBot — Configuration
# Secrets are loaded from environment variables (Railway dashboard
# or local .env file). Never hardcode keys here.
# ============================================================

import os
from dotenv import load_dotenv
load_dotenv()  # loads .env locally; on Railway env vars are already set

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
ACCOUNT_BALANCE    = 3    # USD — actual balance on Delta Exchange

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
# Set USE_TESTNET = True to trade on the testnet (fake money, real API behaviour).
# Register a testnet account at: testnet.india.delta.exchange
# Get testnet API keys and set TESTNET_APIKEY / TESTNET_SECRET in your .env
USE_TESTNET = False

if USE_TESTNET:
    BASE_URL = "https://cdn-ind-testnet.deltaex.org"
    WS_URL   = "wss://testnet-socket.india.delta.exchange"
    DELTA_API_KEY    = os.environ.get("TESTNET_APIKEY", "")
    DELTA_API_SECRET = os.environ.get("TESTNET_SECRET", "")
else:
    BASE_URL = "https://api.india.delta.exchange"
    WS_URL   = "wss://socket.india.delta.exchange"

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

# --- Exchange API keys (set APIKEY and SECRET in Railway Variables or .env) ---
# For testnet keys, set TESTNET_APIKEY and TESTNET_SECRET instead.
# These are overridden below if USE_TESTNET = True.
DELTA_API_KEY    = os.environ.get("APIKEY", "")
DELTA_API_SECRET = os.environ.get("SECRET", "")

BINANCE_API_KEY  = ""
BINANCE_API_SECRET = ""
BYBIT_API_KEY    = ""
BYBIT_API_SECRET = ""
ACTIVE_EXCHANGE  = "delta"   # delta | binance | bybit

# --- TradingView webhook ---
TRADINGVIEW_SECRET = ""

# --- Trading mode ---
PAPER_TRADING_MODE    = False  # True = paper, False = live
AUTO_TRADE            = True   # Auto-place limit orders on Delta Exchange when signal fires
MAX_CONTRACTS_PER_TRADE = 1   # Hard cap — 1 contract = $1 notional. Safe for $3 balance.
MAX_DAILY_LOSS_PERCENT = 3.0
MAX_DRAWDOWN_PERCENT   = 10.0
USE_AI_ENGINE          = False  # flip True when ready

# --- Data directory (Railway Volume mounted at /data, or local ./data) ---
import os as _os
DATA_DIR = _os.environ.get("DATA_DIR", "data")
_os.makedirs(DATA_DIR, exist_ok=True)

# --- Database ---
DATABASE_URL = f"sqlite:///{DATA_DIR}/tradesignal.db"

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
