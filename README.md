# P.R.O.P.H.E.T - Market Swing Predictor

🔮 **P.R.O.P.H.E.T** stands for **Probabilistic Realtime Optimizer for Price Historical Evaluation & Trading**. It is an advanced algorithmic tool designed to forecast potential future market turning points, known as swing highs and swing lows, for various cryptocurrency trading pairs using Multi-TimeFrame (MTF) analysis.

This application is built with Python, leveraging the FastAPI web framework for the backend API and standard HTML/CSS/JavaScript for the frontend.

## Table of Contents

- [Features](#features)
- [How It Works](#how-it-works)
- [Technology Stack](#technology-stack)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Running the Application](#running-the-application)
- [Deployment](#deployment)
- [API Endpoints](#api-endpoints)
- [Security Considerations](#security-considerations)
- [Disclaimer](#disclaimer)
- [License](#license)

## Features

- **Multi-TimeFrame Analysis**: Analyzes price data on both 1-Day (1D) and 1-Hour (1H) timeframes.
- **Swing Detection**: Identifies and classifies significant price swings (Higher Highs, Lower Highs, Higher Lows, Lower Lows).
- **Technical Indicators**: Calculates key indicators like RSI and MACD.
- **Fibonacci Projections**: Uses Fibonacci retracements and extensions for price level prediction.
- **Scheduled Predictions**: Automatically runs predictions daily.
- **Web Interface**: User-friendly frontend to interact with the predictor.
- **Rate Limiting**: Protects the API from abuse.
- **Security Enhancements**: Includes input validation, CORS hardening, and security headers.

## How It Works

1.  **Data Fetching**: Retrieves real-time OHLCV data for a specified trading pair from Binance.
2.  **Swing Identification**: Detects significant price swings on 1D and 1H charts.
3.  **Indicator Calculation**: Computes technical indicators to assess momentum.
4.  **Fibonacci Analysis**: Applies Fibonacci tools to recent significant price moves to project potential future support and resistance levels.
5.  **Confluence Scoring**: Ranks potential future swing levels based on the strength of confluence (where multiple technical factors align).
6.  **Prediction Output**: Generates ranked lists of the 3 most probable swing high levels and 3 most probable swing low levels for the next 3-7 days.

## Technology Stack

- **Backend**: [Python 3.11](https://www.python.org/), [FastAPI](https://fastapi.tiangolo.com/), [APScheduler](https://apscheduler.readthedocs.io/)
- **Data Analysis**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/), [TA-Lib](https://ta-lib.org/)
- **Frontend**: HTML5, CSS3, Vanilla JavaScript
- **Web Server**: [Uvicorn](https://www.uvicorn.org/)
- **Rate Limiting**: [Slowapi](https://github.com/laurentS/slowapi)
- **Deployment (Suggested)**: [Render](https://render.com/)

## Prerequisites

- Python 3.11 installed on your system.
- `pip` (Python package installer).
- A C compiler (for installing `TA-Lib` Python wrapper, which depends on the underlying TA-Lib C library).
    - **Windows**: Visual Studio Build Tools or Visual Studio Community.
    - **macOS**: Xcode Command Line Tools (`xcode-select --install`).
    - **Linux**: `build-essential` (Debian/Ubuntu) or `gcc` (Red Hat/CentOS/Fedora).

## Installation

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/Dev-under-training/prophet-bot-app.git
    cd prophet-bot-app
    ```

2.  **Create a Virtual Environment** (Recommended):
    ```bash
    python -m venv venv
    # Activate the virtual environment:
    # On Windows:
    venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install TA-Lib C Library**:
    Follow the instructions on the [TA-Lib website](https://ta-lib.org/hdr_dw.html) or use a package manager:
    - **Windows**: Pre-compiled binaries are often easiest. Search for "TA-Lib Windows precompiled".
    - **macOS (Homebrew)**: `brew install ta-lib`
    - **Linux (Ubuntu/Debian)**:
        ```bash
        sudo apt-get update
        sudo apt-get install build-essential wget
        wget https://sourceforge.net/projects/ta-lib/files/ta-lib/0.4.0/ta-lib-0.4.0-src.tar.gz
        tar -xzf ta-lib-0.4.0-src.tar.gz
        cd ta-lib/
        ./configure --prefix=/usr
        make
        sudo make install
        # On some systems, you might need to update the library cache:
        sudo ldconfig
        cd ..
        ```

4.  **Install Python Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    This installs FastAPI, Uvicorn, Requests, Pandas, NumPy, TA-Lib (Python wrapper), APScheduler, and Slowapi.

## Running the Application

1.  Ensure your virtual environment is activated (if you created one).
2.  Run the main application file:
    ```bash
    python main.py
    ```
3.  The application will start, typically on `http://0.0.0.0:8000`.
4.  Access the web interface by navigating to `http://localhost:8000/static/index.html` in your web browser.

## Deployment

This application is designed to be easily deployable on platforms like Render.

**For Render:**

1.  Push your code (including the updated `requirements.txt`) to your GitHub repository.
2.  Create a new **Web Service** on Render.
3.  Connect it to your GitHub repository.
4.  Configure the build command: `pip install -r requirements.txt`
5.  Configure the start command: `python main.py`
6.  Set environment variables in the Render dashboard if needed (e.g., `ALLOWED_ORIGINS`).
7.  Render automatically uses the `PORT` environment variable, which the application handles.

## API Endpoints

- `GET /`: Welcome message.
- `GET /api/symbols`: Fetches a list of available trading symbols from Binance.
- `GET /api/predict/{symbol}`: Triggers a new prediction analysis for the specified `symbol` (e.g., `BTCUSDT`).
- `GET /api/latest/{symbol}`: Retrieves the latest cached prediction for the specified `symbol`. If not found, it triggers a new prediction.
- `GET /api/next_run_time`: Gets the ISO timestamp for the next scheduled daily prediction run.

## Security Considerations

This application includes several security enhancements:
- **Input Validation**: Strict validation and sanitization of user inputs (e.g., trading symbols).
- **Rate Limiting**: Prevents abuse of API endpoints using `slowapi`.
- **CORS Hardening**: Restricts allowed origins and methods.
- **Security Headers**: Adds headers like `X-Content-Type-Options`, `X-Frame-Options`, etc.
- **Timeouts**: Configures timeouts for external API requests.

## Disclaimer

**P.R.O.P.H.E.T is for informational and educational purposes only.** It is not financial advice. Cryptocurrency markets are extremely volatile and influenced by numerous unpredictable factors. Past performance is not indicative of future results.

**Risk Warning:** Trading cryptocurrencies involves substantial risk of loss. Never trade money you cannot afford to lose. The predictions generated by P.R.O.P.H.E.T are based on technical analysis and historical data patterns. They are probabilistic estimates and should not be relied upon as guaranteed outcomes. Always conduct your own research and consider seeking advice from qualified financial professionals before making any investment decisions.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details (you should create a `LICENSE` file if you intend to specify a license).