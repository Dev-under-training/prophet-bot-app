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
import traceback
import os
import requests
import xml.etree.ElementTree as ET # Added import for XML parsing

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
# --- FRED API Configuration ---
# TODO: For production, use os.getenv("FRED_API_KEY") and set the environment variable
FRED_API_KEY = "a67de105d456e8a7853a5fd60533ba53" # Use your actual key from testing
FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"


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

# --- NEW: Macro Data Endpoints ---
@app.get("/api/macro/{indicator}")
@limiter.limit("60/minute") # Higher limit for macro data, as it's less frequent
async def get_macro_data(request: Request, indicator: str, limit: int = 24):
    """
    Fetches macroeconomic data from FRED API via server-side proxy to avoid CORS.
    Supported indicators: 'cpi', 'ppi'
    """
    logger.info(f"Fetching macro data for indicator: {indicator}")
    
    # Map simple names to FRED series IDs
    series_map = {
        "cpi": "CPIAUCSL",  # Consumer Price Index for All Urban Consumers: All Items in U.S. City Average
        "ppi": "PPIACO"     # Producer Price Index for All Commodities
    }
    
    series_id = series_map.get(indicator.lower())
    if not series_id:
        logger.warning(f"Unsupported macro indicator requested: {indicator}")
        raise HTTPException(status_code=400, detail=f"Unsupported indicator: {indicator}. Supported: cpi, ppi")
    
    # Construct FRED API URL
    # Note: file_type=json is important for parsing in Python
    fred_url = f"{FRED_BASE_URL}?series_id={series_id}&api_key={FRED_API_KEY}&file_type=json&sort_order=desc&limit={limit}"
    
    try:
        logger.info(f"Making request to FRED API: {fred_url}")
        # Use requests to fetch data from FRED (server-side, no CORS issues)
        response = requests.get(fred_url, timeout=(5, API_TIMEOUT), verify=True) # Reuse API_TIMEOUT from config
        response.raise_for_status() # Raise an exception for bad status codes (4xx, 5xx)
        
        # Parse the JSON response from FRED
        fred_data = response.json()
        
        # Basic validation
        if "observations" not in fred_data:
            logger.error(f"Unexpected response structure from FRED API for {indicator}")
            raise HTTPException(status_code=502, detail="Bad gateway: Unexpected data format from FRED API")
            
        # Sort observations by date ascending for frontend charting
        observations = sorted(fred_data["observations"], key=lambda obs: obs["date"])
        
        logger.info(f"Successfully fetched and processed {len(observations)} data points for {indicator}")
        return {
            "indicator": indicator.upper(),
            "series_id": series_id,
            "data": observations
        }
        
    except requests.exceptions.Timeout:
        logger.error(f"Request to FRED API timed out for {indicator}")
        raise HTTPException(status_code=504, detail="Request to FRED API timed out.")
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching data from FRED API for {indicator}: {e}")
        # Provide a more user-friendly error message for common issues
        if "Invalid API key" in str(e):
            raise HTTPException(status_code=502, detail="Bad gateway: Invalid FRED API key configured on server.")
        raise HTTPException(status_code=502, detail=f"Failed to fetch data from FRED API: {str(e)}")
    except ValueError as e: # JSON decode error
        logger.error(f"Error decoding JSON response from FRED API for {indicator}: {e}")
        raise HTTPException(status_code=502, detail="Bad gateway: Invalid JSON response from FRED API")
    except Exception as e:
        logger.error(f"Unexpected error fetching/processing FRED data for {indicator}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")
# --- END NEW Macro Data Endpoints ---

# --- NEW: FED News Endpoint ---
FED_RSS_URL = "https://libertystreeteconomics.newyorkfed.org/feed/"

@app.get("/api/news/fed", response_model=List[Dict[str, Any]])
@limiter.limit("30/minute") # Rate limit for news fetching
async def get_fed_news(request: Request, limit: int = 10):
    """
    Fetches and parses the latest news/items from the Federal Reserve's RSS feed.
    Returns a list of news items with title, link, description, publication date, and category.
    """
    logger.info(f"Fetching FED news from RSS feed (limit: {limit})")
    
    try:
        # Fetch the RSS feed content
        logger.info(f"Making request to FED RSS feed: {FED_RSS_URL}")
        response = requests.get(FED_RSS_URL, timeout=(5, API_TIMEOUT), verify=True)
        response.raise_for_status() # Raise an exception for bad status codes (4xx, 5xx)
        
        # Get the raw XML text
        rss_content = response.text
        logger.debug(f"Received RSS content (length: {len(rss_content)} chars)")
        
        # --- NEW: Log a snippet of the raw response for debugging ---
        # This helps us see if we got HTML error page or malformed XML
        snippet_length = min(500, len(rss_content))
        logger.debug(f"Snippet of raw RSS response (first {snippet_length} chars): {rss_content[:snippet_length]}")
        # --- END NEW ---
        
        # Check if content seems to be HTML (error page) instead of XML
        if rss_content.strip().startswith("<!DOCTYPE html") or "<html" in rss_content[:200].lower():
             logger.error("Received HTML response instead of XML. This might indicate an error page or blocking.")
             raise HTTPException(status_code=502, detail="Bad gateway: FED server returned an HTML page, possibly an error or block.")
        
        # Check for empty content
        if not rss_content.strip():
            logger.error("Received empty response from FED RSS feed.")
            raise HTTPException(status_code=502, detail="Bad gateway: FED server returned an empty response.")
            
        # Parse the XML content
        logger.info("Attempting to parse RSS content as XML...")
        root = ET.fromstring(rss_content)
        logger.info("XML parsing successful.")
        
        # Define the namespace used in the RSS feed (if any become relevant later)
        # namespaces = { ... } 
        
        # Find the <channel> element (root is usually <rss>)
        channel = root.find('channel') # Use default namespace
        if channel is None:
            # Log more details about the root element found
            logger.error(f"Could not find <channel> element in RSS feed. Root tag found: '{root.tag}'")
            logger.debug(f"Full root element structure: {ET.tostring(root, encoding='unicode')[:500]}...")
            raise HTTPException(status_code=502, detail="Bad gateway: Invalid RSS structure from FED - <channel> not found.")
            
        # Find all <item> elements within the channel
        items = channel.findall('item') # Use default namespace
        logger.info(f"Found {len(items)} items in RSS feed")
        
        news_items = []
        for item in items[:limit]: # Only process up to 'limit' items
            try:
                # Extract data from each <item>
                title_elem = item.find('title')
                title = title_elem.text if title_elem is not None else "No Title"
                
                link_elem = item.find('link')
                link = link_elem.text if link_elem is not None else "#"
                
                # Description might be in <description> or <content:encoded>
                description_elem = item.find('description')
                if description_elem is None:
                    # Try common namespace for encoded content
                    description_elem = item.find('{http://purl.org/rss/1.0/modules/content/}encoded')
                
                description = description_elem.text if description_elem is not None else ""
                
                pub_date_elem = item.find('pubDate')
                pub_date_str = pub_date_elem.text if pub_date_elem is not None else ""
                
                # Category can be multiple
                category_elems = item.findall('category')
                categories = [cat.text for cat in category_elems if cat.text] if category_elems else []
                category_str = ", ".join(categories) if categories else "General"
                
                guid_elem = item.find('guid')
                guid = guid_elem.text if guid_elem is not None else link # Fallback to link
                
                news_item = {
                    "title": title.strip() if title else "No Title",
                    "link": link.strip() if link else "#",
                    "description": description.strip() if description else "",
                    "pub_date": pub_date_str.strip() if pub_date_str else "",
                    "categories": categories,
                    "category_str": category_str,
                    "guid": guid.strip() if guid else link
                }
                news_items.append(news_item)
                
            except Exception as item_error:
                logger.warning(f"Error processing individual RSS item: {item_error}")
                logger.debug(f"Traceback for item error: {traceback.format_exc()}")
                # Continue processing other items even if one fails
                continue
                
        logger.info(f"Successfully processed {len(news_items)} FED news items")
        return news_items
        
    except requests.exceptions.Timeout:
        logger.error("Request to FED RSS feed timed out")
        raise HTTPException(status_code=504, detail="Request to FED RSS feed timed out.")
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching FED RSS feed: {e}")
        raise HTTPException(status_code=502, detail=f"Failed to fetch FED RSS feed: {str(e)}")
    except ET.ParseError as e:
        logger.error(f"Error parsing FED RSS XML: {e}")
        # Log the traceback for more details on where parsing failed
        logger.debug(f"XML Parse traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=502, detail="Bad gateway: Invalid XML response from FED RSS feed. Parsing failed.")
    except HTTPException: # Re-raise HTTPExceptions we throw
        raise
    except Exception as e:
        logger.error(f"Unexpected error fetching/processing FED news: {e}", exc_info=True) # exc_info logs the full traceback
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")
# --- END NEW: FED News Endpoint ---

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
