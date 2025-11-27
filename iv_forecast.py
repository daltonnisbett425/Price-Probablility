import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from datetime import datetime, timedelta

# --- 1. Plain English Input Helper ---
def parse_timeframe(user_input):
    """
    Translates plain English into yfinance intervals and periods,
    and decides how far into the future to look.
    """
    u = user_input.lower().strip()
    
    # Defaults
    interval = "1d"
    period = "1y"
    days_to_forecast = 180
    
    if any(x in u for x in ['short', 'hour', '1h', 'day trade', 'intraday']):
        print(">> Mode: Short Term (Hourly candles, 3 month history)")
        interval = "1h"
        period = "3mo"
        days_to_forecast = 45 # Look 45 days into future
        
    elif any(x in u for x in ['med', 'day', 'daily', 'swing']):
        print(">> Mode: Medium Term (Daily candles, 1 year history)")
        interval = "1d"
        period = "1y"
        days_to_forecast = 180 # Look 6 months into future

    elif any(x in u for x in ['long', 'week', 'year', 'invest']):
        print(">> Mode: Long Term (Weekly candles, 2 year history)")
        interval = "1wk"
        period = "2y"
        days_to_forecast = 365 # Look 1 year into future
        
    return interval, period, days_to_forecast

# --- 2. Data Fetching ---
def get_data(ticker, interval, period, days_to_forecast):
    print(f"--- Fetching {ticker} data ---")
    stock = yf.Ticker(ticker)
    
    # Get History
    try:
        hist = stock.history(period=period, interval=interval)
        if hist.empty:
            raise ValueError("No history found.")
        current_price = hist['Close'].iloc[-1]
        last_date = hist.index[-1].to_pydatetime().replace(tzinfo=None) # Strip timezone for compatibility
    except Exception as e:
        print(f"Error fetching history: {e}")
        return None, None, None, None

    # Get Option Expirations
    try:
        expirations = stock.options
        valid_dates = []
        for e in expirations:
            d = datetime.strptime(e, "%Y-%m-%d")
            days_out = (d - datetime.now()).days
            # Filter options based on the "Mode" (don't show 2 year options for a day trade)
            if 2 < days_out < days_to_forecast:
                valid_dates.append(e)
    except:
        return None, None, None, None

    return stock, hist, current_price, valid_dates

# --- 3. The Math (Heatmap Logic) ---
def calculate_heatmap(stock, current_price, exp_dates, hist_data):
    RISK_FREE_RATE = 0.046
    grid_data = []
    processed_dates = []
    ranges_50_pct = [] 
    
    # Calculate Y-Axis Range based on History AND Future
    # We look at historical volatility to guess how wide the graph should be
    hist_std = hist_data['Close'].std()
    price_min = min(hist_data['Low'].min(), current_price * 0.7)
    price_max = max(hist_data['High'].max(), current_price * 1.3)
    
    # Create the price grid
    price_range = np.linspace(price_min * 0.8, price_max * 1.2, 300)

    # Add the "Today" column (Starting point of the cone)
    # At t=0, probability is 100% at current price (represented as a sharp spike)
    # We skip plotting t=0 in the heatmap to avoid visual glitches, 
    # but the cones connect from here.
    
    print(f"Calculating probabilities for {len(exp_dates)} expiration dates...")
    
    for date_str in exp_dates:
        try:
            opt = stock.option_chain(date_str)
            calls, puts = opt.calls, opt.puts
            if calls.empty or puts.empty: continue

            exp_date = datetime.strptime(date_str, "%Y-%m-%d")
            T = (exp_date - datetime.now()).days / 365.0
            if T <= 0.001: continue

            # Get ATM Volatility
            calls['dist'] = abs(calls['strike'] - current_price)
            puts['dist'] = abs(puts['strike'] - current_price)
            iv = (calls.sort_values('dist').iloc[0]['impliedVolatility'] + 
                  puts.sort_values('dist').iloc[0]['impliedVolatility']) / 2
            
            if iv < 0.05: iv = 0.2 # Safety floor

            # Log-Normal Distribution
            mu = np.log(current_price) + (RISK_FREE_RATE - 0.5 * iv**2) * T
            sigma_t = iv * np.sqrt(T)
            
            pdf_values = stats.lognorm.pdf(price_range, s=sigma_t, scale=np.exp(mu))
            grid_data.append(pdf_values / np.max(pdf_values)) # Normalize
            processed_dates.append(exp_date)
            
            # 50% Cone Data
            ppf_25 = stats.lognorm.ppf(0.25, s=sigma_t, scale=np.exp(mu))
            ppf_75 = stats.lognorm.ppf(0.75, s=sigma_t, scale=np.exp(mu))
            ranges_50_pct.append({'date': exp_date, 'lower': ppf_25, 'upper': ppf_75})

        except:
            continue

    if not grid_data: return None, None, None, None

    # Transpose for plotting
    heatmap_matrix = np.array(grid_data).T
    return heatmap_matrix, price_range, processed_dates, ranges_50_pct

# --- 4. Main Execution ---
if __name__ == "__main__":
    # --- USER INPUTS ---
    ticker = input("Enter Stock Ticker (e.g., NVDA, SPY): ").upper()
    timeframe_input = input("Enter Time Frame (e.g., 'short term', 'hourly', 'daily', 'long term'): ")
    
    interval, period, forecast_days = parse_timeframe(timeframe_input)
    
    # Run
    stock, hist, current_price, exp_dates = get_data(ticker, interval, period, forecast_days)
    
    if stock and len(exp_dates) > 0:
        heatmap_matrix, price_range, final_dates, ranges_50_pct = calculate_heatmap(stock, current_price, exp_dates, hist)
        
        # --- PLOTTING ---
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # 1. Plot Historical Price
        # Clean timestamps for plotting
        hist_dates = [x.to_pydatetime().replace(tzinfo=None) for x in hist.index]
        ax.plot(hist_dates, hist['Close'], color='white', linewidth=2, label='Historical Price')
        
        # 2. Plot Heatmap
        # Meshgrid needs the dates on X and prices on Y
        X, Y = np.meshgrid(final_dates, price_range)
        c = ax.pcolormesh(X, Y, heatmap_matrix, shading='auto', cmap='inferno', alpha=0.9)
        
        # 3. Plot The 50% Cone (Red Lines)
        # We need to connect the cone to the current price
        cone_dates = [hist_dates[-1]] + [x['date'] for x in ranges_50_pct]
        cone_lower = [current_price] + [x['lower'] for x in ranges_50_pct]
        cone_upper = [current_price] + [x['upper'] for x in ranges_50_pct]
        
        ax.plot(cone_dates, cone_lower, color='cyan', linestyle='--', linewidth=1.5, label='25th Percentile')
        ax.plot(cone_dates, cone_upper, color='cyan', linestyle='--', linewidth=1.5, label='75th Percentile')
        
        # Formatting
        ax.set_title(f"{ticker}: Historical Price + Option Implied Future ({timeframe_input})", fontsize=16, color='white')
        ax.axhline(current_price, color='gray', linestyle=':', alpha=0.5)
        ax.legend()
        
        # Fix X-axis overlap
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        print("Displaying Chart...")
        plt.show()
    else:
        print("Could not retrieve enough data to build the chart.")
