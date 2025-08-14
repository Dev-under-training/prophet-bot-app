# bot_engine.py
# This file contains the core MTFProphetBot class logic.

import requests
import pandas as pd
import numpy as np
import talib
from datetime import datetime

# --- 1. Define the complete MTFProphetBot class ---
class MTFProphetBot:
    """
    A Multi-TimeFrame prediction bot for identifying potential swing points
    based on confluence across the 1-hour and 1-day timeframes.
    It predicts the next 3 swing highs and 3 swing lows for the next 3-7 days.
    """

    def __init__(self, symbol, intervals=['1d'], limit=500):
        """Initializes the MTF Prophet Bot."""
        self.symbol = symbol
        self.intervals = intervals
        self.limit = limit
        self.timeframes = {interval: {} for interval in intervals}

    # --- Core Data Fetching ---
    def fetch_binance_data_mtf(self):
        """Fetches OHLCV data from Binance API for all specified intervals."""
        url = "https://api.binance.com/api/v3/klines"
        for interval in self.intervals:
            params = {'symbol': self.symbol, 'interval': interval, 'limit': self.limit}
            try:
                response = requests.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                df = pd.DataFrame(data, columns=[
                    'timestamp', 'open', 'high', 'low', 'close',
                    'volume', 'close_time', 'quote_asset_volume',
                    'number_of_trades', 'taker_buy_base_asset_volume',
                    'taker_buy_quote_asset_volume', 'ignore'
                ])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = pd.to_numeric(df[col])
                df.set_index('timestamp', inplace=True)
                df.sort_index(inplace=True)
                self.timeframes[interval]['df'] = df[['open', 'high', 'low', 'close', 'volume']]
            except Exception as e:
                self.timeframes[interval]['df'] = None

    # --- Robust Swing Detection ---
    def identify_swings_on_df_robust(self, df, window=5):
        """Identifies and classifies swings (HH, HL, LH, LL) robustly."""
        if df is None or len(df) < window * 3: return df
        df = df.copy()
        df['swing_high'] = np.nan
        df['swing_low'] = np.nan
        df['swing_type'] = None
        # 1. Identify raw swing points
        for i in range(window, len(df) - window):
            if df['high'].iloc[i] == df['high'].iloc[i - window:i + window + 1].max():
                df.loc[df.index[i], 'swing_high'] = df['high'].iloc[i]
            if df['low'].iloc[i] == df['low'].iloc[i - window:i + window + 1].min():
                df.loc[df.index[i], 'swing_low'] = df['low'].iloc[i]
        # 2. Classify swings
        swing_points_df = df.dropna(subset=['swing_high', 'swing_low'], how='all')
        swing_points_df = swing_points_df[(~swing_points_df['swing_high'].isna()) | (~swing_points_df['swing_low'].isna())]
        swing_points_df.sort_index(inplace=True)
        if len(swing_points_df) < 1: return df
        classified_swings = []
        for idx, row in swing_points_df.iterrows():
            current_price = row['swing_high'] if not np.isnan(row['swing_high']) else row['swing_low']
            is_high = not np.isnan(row['swing_high'])
            swing_type = "Error_Unassigned"
            if len(classified_swings) == 0:
                swing_type = 'First High' if is_high else 'First Low'
            else:
                last_relevant_swing = None
                for prev_swing in reversed(classified_swings):
                    if prev_swing[2] is None: continue
                    prev_is_high = 'High' in str(prev_swing[2])
                    if is_high and not prev_is_high: # High vs Low
                        last_relevant_swing = prev_swing; break
                    elif not is_high and prev_is_high: # Low vs High
                        last_relevant_swing = prev_swing; break
                if last_relevant_swing is None and len(classified_swings) > 0:
                    last_relevant_swing = classified_swings[-1]
                if last_relevant_swing and last_relevant_swing[2] is not None:
                    try:
                        if is_high:
                            swing_type = 'HH' if current_price > last_relevant_swing[1] else 'LH'
                        else: # is_low
                            swing_type = 'LL' if current_price < last_relevant_swing[1] else 'HL'
                    except:
                        swing_type = f"Error_Classify"
                else:
                    swing_type = f"Unclassified {'High' if is_high else 'Low'}"
            if swing_type == "Error_Unassigned":
                swing_type = f"Error_Unassigned_{idx}"
            df.loc[idx, 'swing_type'] = swing_type
            classified_swings.append((idx, current_price, swing_type))
        return df

    # --- Indicator Calculation ---
    def calculate_indicators_on_df(self, df):
        """Calculates technical indicators."""
        if df is None or len(df) < 30: return df
        df = df.copy()
        df['rsi'] = talib.RSI(df['close'], timeperiod=14)
        df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(df['close'])
        return df

    # --- Fibonacci Calculation ---
    def calculate_fibonacci_on_df(self, df, lookback=100):
        """Calculates Fibonacci retracement and extension levels."""
        if df is None: return df
        df = df.copy()
        recent_data = df.tail(lookback)
        swings = recent_data.dropna(subset=['swing_high', 'swing_low'], how='all')
        swings = swings[(~swings['swing_high'].isna()) | (~swings['swing_low'].isna())]
        swings.sort_index(inplace=True)
        if len(swings) >= 2:
            # Use last two swings to define the move
            start_swing = swings.iloc[-2]
            end_swing = swings.iloc[-1]
            start_price = start_swing['swing_high'] if not np.isnan(start_swing['swing_high']) else start_swing['swing_low']
            end_price = end_swing['swing_high'] if not np.isnan(end_swing['swing_high']) else end_swing['swing_low']
            if start_price is not None and end_price is not None:
                move_size = end_price - start_price
                # Calculate common levels
                fib_ratios = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.272, 1.618]
                for ratio in fib_ratios:
                    # Extensions for impulse, Retracements for corrective waves
                    level_price = end_price - move_size * ratio if move_size > 0 else end_price + abs(move_size) * ratio
                    df[f'fib_{ratio * 100:.1f}'.replace('.0', '')] = level_price
        return df

    # --- Main Analysis Runner ---
    def run_mtf_analysis(self):
        """Runs the complete MTF analysis workflow."""
        self.fetch_binance_data_mtf()
        for interval in self.intervals:
            df = self.timeframes[interval].get('df')
            if df is not None and not df.empty:
                # Use robust swing detection
                df = self.identify_swings_on_df_robust(df, window=5)
                df = self.calculate_indicators_on_df(df)
                df = self.calculate_fibonacci_on_df(df, lookback=100)
                self.timeframes[interval]['df'] = df
                # Store Fibonacci levels
                fib_cols = [col for col in df.columns if col.startswith('fib_')]
                if fib_cols:
                    self.timeframes[interval]['fib_levels'] = df[fib_cols].iloc[-1].dropna().to_dict()
                else:
                    self.timeframes[interval]['fib_levels'] = {}
                # Store swing data
                swing_points = df.dropna(subset=['swing_type'])
                if not swing_points.empty:
                    self.timeframes[interval]['recent_swings'] = swing_points.tail(10)
                    self.timeframes[interval]['all_swings'] = swing_points
                else:
                    self.timeframes[interval]['recent_swings'] = pd.DataFrame()
                    self.timeframes[interval]['all_swings'] = pd.DataFrame()

    # --- Core Prediction Logic ---
    def predict_upcoming_swings(self, num_predictions=3, time_horizon_days_min=3, time_horizon_days_max=7):
        """
        Predicts the next N swing highs and lows based on 1h and 1d analysis.
        """
        predictions = {'swing_highs': [], 'swing_lows': []}
        
        # Ensure required timeframes are analyzed
        if '1d' not in self.timeframes or '1h' not in self.timeframes:
            return predictions
        
        df_1d = self.timeframes['1d'].get('df')
        df_1h = self.timeframes['1h'].get('df')
        
        if df_1d is None or df_1h is None or df_1d.empty or df_1h.empty:
            return predictions

        # --- 1. Determine 1D Trend Context ---
        swings_1d_df = df_1d.dropna(subset=['swing_type'])
        if swings_1d_df.empty:
            return predictions
            
        recent_swings_1d = swings_1d_df.tail(5) # Last 5 swings
        
        # Simple trend determination
        last_swing_type = recent_swings_1d['swing_type'].iloc[-1]
        last_price_high = recent_swings_1d['swing_high'].iloc[-1]
        last_price_low = recent_swings_1d['swing_low'].iloc[-1]
        last_swing_price = last_price_high if not np.isnan(last_price_high) else last_price_low
        
        if last_swing_type in ['HH', 'HL']:
            primary_trend = 'UPTREND'
            expected_next_swing_type = 'LOW' # Pullback expected
        elif last_swing_type in ['LH', 'LL']:
            primary_trend = 'DOWNTREND'
            expected_next_swing_type = 'HIGH' # Bounce expected
        else: # First High/Low or Undefined
            primary_trend = 'UNDEFINED'
            expected_next_swing_type = 'BOTH' # Predict both

        current_price_1d = df_1d['close'].iloc[-1]

        # --- 2. Identify Key 1D Levels for Projection ---
        fib_ext_levels_1d = {}
        fib_retr_levels_1d = {}
        if len(recent_swings_1d) >= 3:
            # Get the last completed swing move (e.g., Low -> High -> Low)
            s1 = recent_swings_1d.iloc[-3]
            s2 = recent_swings_1d.iloc[-2]
            s3 = recent_swings_1d.iloc[-1]
            
            s1_price = s1['swing_high'] if not np.isnan(s1['swing_high']) else s1['swing_low']
            s2_price = s2['swing_high'] if not np.isnan(s2['swing_high']) else s2['swing_low']
            s3_price = s3['swing_high'] if not np.isnan(s3['swing_high']) else s3['swing_low']
            
            if all(p is not None for p in [s1_price, s2_price, s3_price]):
                move1_size = s2_price - s1_price # Impulse 1 (e.g., 1->2)
                move2_size = s3_price - s2_price # Correction (e.g., 2->3)
                
                # Fibonacci Extension targets from s2 (start of correction) in direction of primary trend
                fib_ext_ratios = [1.0, 1.272, 1.618] # Common extensions
                if primary_trend == 'UPTREND':
                    # Projecting upwards from s2
                    for ratio in fib_ext_ratios:
                        target_price = s2_price + move1_size * ratio
                        fib_ext_levels_1d[f'Ext_{ratio*100:.0f}'] = target_price
                elif primary_trend == 'DOWNTREND':
                    # Projecting downwards from s2
                    for ratio in fib_ext_ratios:
                        target_price = s2_price - move1_size * ratio
                        fib_ext_levels_1d[f'Ext_{ratio*100:.0f}'] = target_price
                
                # Fibonacci Retracement levels for the correction (s2 -> s3)
                fib_retr_ratios = [0.236, 0.382, 0.5, 0.618, 0.786]
                for ratio in fib_retr_ratios:
                    if primary_trend == 'UPTREND': # Correction was down (s2 -> s3), retrace up
                        retr_price = s3_price + (s2_price - s3_price) * ratio
                    else: # Correction was up (s2 -> s3), retrace down
                        retr_price = s3_price - (s3_price - s2_price) * ratio
                    fib_retr_levels_1d[f'Retr_{ratio*100:.1f}'] = retr_price

        # --- 3. Identify Key 1H Levels for Confluence ---
        swings_1h_df = df_1h.dropna(subset=['swing_type'])
        fib_1h_levels = {}
        if not swings_1h_df.empty:
            recent_swings_1h = swings_1h_df.tail(10) # Last 10 swings on 1h
            
            # Get recent 1H swing points for Fibonacci
            if len(recent_swings_1h) >= 2:
                last_1h_swing = recent_swings_1h.iloc[-1]
                prev_1h_swing = recent_swings_1h.iloc[-2]
                
                l1h_price = last_1h_swing['swing_high'] if not np.isnan(last_1h_swing['swing_high']) else last_1h_swing['swing_low']
                p1h_price = prev_1h_swing['swing_high'] if not np.isnan(prev_1h_swing['swing_high']) else prev_1h_swing['swing_low']
                
                if l1h_price is not None and p1h_price is not None:
                    move_1h_size = l1h_price - p1h_price
                    
                    # Fibonacci levels based on last 1h move
                    fib_common_ratios = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.272, 1.618]
                    for ratio in fib_common_ratios:
                        # Retracements of the last move
                        if move_1h_size > 0: # Up move
                            level_price = l1h_price - move_1h_size * ratio
                        else: # Down move
                            level_price = l1h_price - move_1h_size * ratio # Subtracting a negative is adding
                        fib_1h_levels[f'1h_Fib_{ratio*100:.1f}'] = level_price

        # --- 4. Combine and Rank Predictions ---
        candidate_levels = []
        
        # Add 1D Extension targets if trend is strong
        if primary_trend in ['UPTREND', 'DOWNTREND'] and fib_ext_levels_1d:
            for name, level in fib_ext_levels_1d.items():
                candidate_levels.append({
                    'level': level,
                    'type': '1D_Extension',
                    'name': name,
                    'score': 8 # Base high score for 1D projection
                })

        # Add 1D Retracement targets (likely near-term swings)
        if fib_retr_levels_1d:
            for name, level in fib_retr_levels_1d.items():
                base_score = 6
                if '61.8' in name:
                    base_score += 2
                candidate_levels.append({
                    'level': level,
                    'type': '1D_Retracement',
                    'name': name,
                    'score': base_score
                })
                
        # Add 1H Fibonacci levels
        if fib_1h_levels:
            for name, level in fib_1h_levels.items():
                base_score = 4 # Lower base score for 1H
                if '61.8' in name or '50' in name:
                    base_score += 1
                candidate_levels.append({
                    'level': level,
                    'type': '1H_Fib',
                    'name': name,
                    'score': base_score
                })

        # --- 5. Refine based on expected next swing type ---
        refined_candidates = []
        for cand in candidate_levels:
            cand_level = cand['level']
            # Boost score if level is on the expected side
            if expected_next_swing_type == 'LOW' and cand_level < current_price_1d:
                cand['score'] += 2
                refined_candidates.append(cand)
            elif expected_next_swing_type == 'HIGH' and cand_level > current_price_1d:
                cand['score'] += 2
                refined_candidates.append(cand)
            elif expected_next_swing_type == 'BOTH':
                refined_candidates.append(cand)
                
        # --- 6. Sort by score and Select Top N for Highs and Lows ---
        refined_candidates.sort(key=lambda x: x['score'], reverse=True)
        
        # Separate into potential highs and lows
        potential_highs = [c for c in refined_candidates if c['level'] > current_price_1d]
        potential_lows = [c for c in refined_candidates if c['level'] < current_price_1d]
        
        # Take top N from each
        top_highs = potential_highs[:num_predictions]
        top_lows = potential_lows[:num_predictions]
            
        # Format output
        predictions['swing_highs'] = [
            {
                'level': h['level'],
                'source': f"{h['type']} - {h['name']}",
                'score': h['score'],
                'time_horizon_days_min': time_horizon_days_min,
                'time_horizon_days_max': time_horizon_days_max
            } for h in top_highs
        ]
        predictions['swing_lows'] = [
            {
                'level': l['level'],
                'source': f"{l['type']} - {l['name']}",
                'score': l['score'],
                'time_horizon_days_min': time_horizon_days_min,
                'time_horizon_days_max': time_horizon_days_max
            } for l in top_lows
        ]
        
        return predictions

# The file should END here with the class definition.
