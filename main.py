# main.py
# This is the main FastAPI application file.

# --- NEW IMPORTS for Scheduler ---
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from datetime import datetime, timezone
import asyncio
import re

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from typing import List, Dict, Any
import logging
import os
import requests

# Import for rate limiting
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# 1. Import our bot engine
from bot_engine import MTFProphetBot 

# --- 2. Configure logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 3. Create the FastAPI app instance ---
app = FastAPI(
    title="Prophet Bot API",
    description="API for the Multi-TimeFrame Swing Prediction Bot",
    version="1.0.0"
)

# --- Rate Limiting Setup ---
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# --- Security Headers Middleware ---
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['X-XSS-Protection'] = '1; mode=block'
        response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
        return response

app.add_middleware(SecurityHeadersMiddleware)

# --- 4. Configure CORS (Hardened) ---
# Get allowed origins from environment variable, default to localhost for development
allowed_origins = os.getenv('ALLOWED_ORIGINS', 'http://localhost:8000,http://127.0.0.1:8000').split(',')

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,  # Restricted origins
    allow_credentials=True,
    allow_methods=["GET"],  # Only allow GET methods
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# --- Configuration from environment variables ---
API_TIMEOUT = int(os.getenv('API_TIMEOUT', '10'))
MAX_SYMBOL_LENGTH = int(os.getenv('MAX_SYMBOL_LENGTH', '20'))
BINANCE_API_BASE_URL = os.getenv('BINANCE_API_BASE_URL', 'https://api.binance.com')

# --- 5. In-memory storage for predictions and scheduler state ---
latest_predictions_store: Dict[str, Dict[str, Any]] = {}
# Store the next run time for the frontend countdown
next_run_time_store: Dict[str, Any] = {"next_run_time_iso": None}

# --- NEW: 6. Create the Scheduler Instance ---
# Use AsyncIOScheduler to work well with FastAPI's async nature
scheduler = AsyncIOScheduler()

# --- NEW: 7. Define the Scheduled Task Function ---
async def run_scheduled_predictions():
    """
    This function will be called by the scheduler.
    It runs predictions for a predefined list of symbols.
    For demo, we'll use BTCUSDT and ETHUSDT.
    You can modify this list or make it dynamic.
    """
    symbols_to_predict = ["BTCUSDT", "ETHUSDT"] # Example list
    logger.info(f"[SCHEDULER] Starting scheduled predictions for: {symbols_to_predict}")
    
    for symbol in symbols_to_predict:
        try:
            logger.info(f"[SCHEDULER] Predicting for {symbol}...")
            bot = MTFProphetBot(symbol=symbol.upper(), intervals=['1d', '1h'], limit=100)
            bot.run_mtf_analysis()
            predictions = bot.predict_upcoming_swings(num_predictions=3, time_horizon_days_min=3, time_horizon_days_max=7)
            predictions['timestamp_utc'] = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%fZ')
            # Store the prediction
            latest_predictions_store[symbol.upper()] = predictions
            logger.info(f"[SCHEDULER] Prediction for {symbol} completed and stored.")
        except Exception as e:
            logger.error(f"[SCHEDULER] Error predicting for {symbol}: {e}", exc_info=True)
    
    logger.info("[SCHEDULER] All scheduled predictions finished.")

# --- NEW: 8. API Endpoint to get Next Run Time ---
@app.get("/api/next_run_time")
@limiter.limit("30/minute")  # Rate limit for this endpoint
async def get_next_run_time(request: Request):
    """
    Returns the ISO format timestamp of the next scheduled run.
    Used by the frontend for the countdown timer.
    """
    next_run_iso = next_run_time_store.get("next_run_time_iso")
    if next_run_iso:
        return {"next_run_time_iso": next_run_iso}
    else:
        # If scheduler hasn't started or job not found, return a default
        return {"next_run_time_iso": None, "message": "Scheduler not active or time not set yet."}

# --- API Endpoints (Existing, with security improvements) ---

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Prophet Bot API!", "status": "Operational"}

@app.get("/api/symbols", response_model=List[str])
@limiter.limit("15/minute")  # Rate limit for symbol fetching
async def get_symbols(request: Request):
    logger.info("Fetching symbols from Binance...")
    url = f"{BINANCE_API_BASE_URL}/api/v3/exchangeInfo"
    headers = {
        'User-Agent': 'MTFProphetBot/1.0'
    }
    
    try:
        response = requests.get(
            url, 
            timeout=(5, API_TIMEOUT), 
            headers=headers,
            verify=True
        )
        response.raise_for_status()
        data = response.json()
        symbols = [
            symbol_info['symbol']
            for symbol_info in data['symbols']
            if symbol_info['status'] == 'TRADING' and
               symbol_info['isSpotTradingAllowed'] == True
        ]
        logger.info(f"Successfully fetched {len(symbols)} trading symbols.")
        return symbols
    except requests.exceptions.Timeout:
        logger.error("Request to Binance API timed out.")
        raise HTTPException(status_code=504, detail="Request to Binance API timed out.")
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching symbols from Binance: {e}")
        raise HTTPException(status_code=503, detail="Failed to fetch symbols from Binance API")
    except KeyError as e:
        logger.error(f"Unexpected response structure from Binance API (KeyError): {e}")
        raise HTTPException(status_code=500, detail="Error processing Binance API response: Missing expected key")
    except Exception as e:
        logger.error(f"Unexpected error fetching/processing symbols from Binance: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Service temporarily unavailable.")

@app.get("/api/predict/{symbol}", response_model=Dict[str, Any])
@limiter.limit("10/minute")  # Rate limit for predictions
async def predict_symbol(request: Request, symbol: str):
    symbol = symbol.upper().strip()
    logger.info(f"Received prediction request for symbol: {symbol}")
    
    # Enhanced input validation
    if not symbol or not isinstance(symbol, str):
        raise HTTPException(status_code=400, detail="Invalid symbol provided.")
    
    # Length validation
    if len(symbol) > MAX_SYMBOL_LENGTH:
        raise HTTPException(status_code=400, detail=f"Symbol too long. Maximum {MAX_SYMBOL_LENGTH} characters.")
    
    if len(symbol) < 3:
        raise HTTPException(status_code=400, detail="Symbol too short. Minimum 3 characters.")
    
    # Format validation (alphanumeric, typically uppercase)
    if not re.match(r'^[A-Z0-9]{3,20}$', symbol):
        raise HTTPException(status_code=400, detail="Invalid symbol format. Must be 3-20 alphanumeric characters.")
    
    try:
        bot = MTFProphetBot(symbol=symbol, intervals=['1d', '1h'], limit=100)
        logger.info(f"Starting MTF analysis for {symbol}...")
        bot.run_mtf_analysis()
        logger.info(f"Generating predictions for {symbol}...")
        predictions = bot.predict_upcoming_swings(num_predictions=3, time_horizon_days_min=3, time_horizon_days_max=7)
        predictions['timestamp_utc'] = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%fZ')
        latest_predictions_store[symbol] = predictions
        logger.info(f"Predictions generated successfully for {symbol}.")
        return {
            "symbol": symbol,
            "predictions": predictions,
            "status": "success"
        }
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        logger.error(f"Prediction failed for {symbol}: {str(e)}", exc_info=True)  # Log full error
        raise HTTPException(status_code=500, detail="Prediction service temporarily unavailable.")  # Generic message

@app.get("/api/latest/{symbol}", response_model=Dict[str, Any])
@limiter.limit("20/minute")  # Rate limit for latest predictions
async def get_latest_prediction(request: Request, symbol: str):
    symbol = symbol.upper().strip()
    
    # Enhanced input validation for latest prediction endpoint
    if not symbol or not isinstance(symbol, str):
        raise HTTPException(status_code=400, detail="Invalid symbol provided.")
    
    # Length validation
    if len(symbol) > MAX_SYMBOL_LENGTH:
        raise HTTPException(status_code=400, detail=f"Symbol too long. Maximum {MAX_SYMBOL_LENGTH} characters.")
    
    if len(symbol) < 3:
        raise HTTPException(status_code=400, detail="Symbol too short. Minimum 3 characters.")
    
    # Format validation
    if not re.match(r'^[A-Z0-9]{3,20}$', symbol):
        raise HTTPException(status_code=400, detail="Invalid symbol format. Must be 3-20 alphanumeric characters.")
    
    if symbol in latest_predictions_store:
        return {
            "symbol": symbol,
            "predictions": latest_predictions_store[symbol],
            "status": "success (from cache)"
        }
    else:
        raise HTTPException(status_code=404, detail=f"No recent prediction found for {symbol}. Please trigger a new prediction via /api/predict/{symbol}.")

# --- NEW: 9. Startup and Shutdown Events for Scheduler ---
@app.on_event("startup")
async def startup_event():
    """
    This function runs when the FastAPI app starts up.
    We initialize and start the scheduler here.
    """
    logger.info("Initializing scheduler...")
    
    # Add the job to the scheduler
    # This sets it to run daily at 00:00 UTC
    scheduler.add_job(
        func=run_scheduled_predictions, # The function to call
        trigger=CronTrigger(hour=0, minute=0, second=0, timezone=timezone.utc), # Run at 00:00:00 UTC daily
        id='daily_prediction_job' # Unique ID for the job
    )
    
    # Start the scheduler
    scheduler.start()
    logger.info("Scheduler started.")
    
    # --- Store the initial next run time ---
    # Find the job and get its next run time
    job = scheduler.get_job('daily_prediction_job')
    if job and job.next_run_time:
        next_run_time_store["next_run_time_iso"] = job.next_run_time.isoformat()
        logger.info(f"Initial next run time set to: {job.next_run_time}")

@app.on_event("shutdown")
async def shutdown_event():
    """
    This function runs when the FastAPI app shuts down.
    We shut down the scheduler here.
    """
    logger.info("Shutting down scheduler...")
    scheduler.shutdown()
    logger.info("Scheduler shut down.")

# --- 10. Main execution block (Updated for Render) ---
if __name__ == "__main__":
    import uvicorn
    import os # Make sure 'os' is imported at the top if not already

    # IMPORTANT: Render sets the PORT environment variable.
    # Use it, otherwise default to 8000 for local development.
    port = int(os.environ.get("PORT", 8000))
    
    # Run the app with uvicorn
    # host="0.0.0.0" is required for Render to expose the app
    # reload=False is recommended for production/deployment
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)