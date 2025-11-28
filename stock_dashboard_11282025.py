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
        days_to_forecast = 45 
        
    elif any(x in u for x in ['med', 'day', 'daily', 'swing']):
        print(">> Mode: Medium Term (Daily candles, 1 year history)")
        interval = "1d"
        period = "1y"
        days_to_forecast = 180 

    elif any(x in u for x in ['long', 'week', 'year', 'invest']):
        print(">> Mode: Long Term (Weekly candles, 2 year history)")
        interval = "1wk"
        period = "2y"
        days_to_forecast = 365 
        
    return interval, period, days_to_forecast

# --- 2. Data Fetching ---
def get_data(ticker, interval, period, days_to_forecast):
    print(f"--- Fetching {ticker} data ---")
    stock = yf.Ticker(ticker)
    
    try:
        hist = stock.history(period=period, interval=interval)
        if hist.empty:
            raise ValueError("No history found.")
        current_price = hist['Close'].iloc[-1]
        
        # Calculate Historical Volatility (Annualized) for comparison
        # Log returns -> std dev -> annualized
        log_returns = np.log(hist['Close'] / hist['Close'].shift(1))
        # annualize factor depends on interval
        ann_factor = 252 if interval == '1d' else (252*7 if interval == '1h' else 52)
        hist_vol = log_returns.std() * np.sqrt(ann_factor)

    except Exception as e:
        print(f"Error fetching history: {e}")
        return None, None, None, None, None

    # Get Option Expirations
    try:
        expirations = stock.options
        valid_dates = []
        for e in expirations:
            d = datetime.strptime(e, "%Y-%m-%d")
            days_out = (d - datetime.now()).days
            if 2 < days_out < days_to_forecast:
                valid_dates.append(e)
    except:
        return None, None, None, None, None

    return stock, hist, current_price, valid_dates, hist_vol

# --- 3. The Math (Heatmap Logic) ---
def calculate_heatmap(stock, current_price, exp_dates, hist_data):
    RISK_FREE_RATE = 0.046
    grid_data = []
    processed_dates = []
    ranges_50_pct = [] 
    iv_tracking = [] # To calculate average IV later
    
    hist_std = hist_data['Close'].std()
    price_min = min(hist_data['Low'].min(), current_price * 0.7)
    price_max = max(hist_data['High'].max(), current_price * 1.3)
    
    price_range = np.linspace(price_min * 0.8, price_max * 1.2, 300)

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
            
            # Average of Call IV and Put IV at the Money
            call_iv = calls.sort_values('dist').iloc[0]['impliedVolatility']
            put_iv = puts.sort_values('dist').iloc[0]['impliedVolatility']
            iv = (call_iv + put_iv) / 2
            
            if iv < 0.05: iv = 0.2 # Safety floor
            
            iv_tracking.append(iv) # Store for stats

            # Log-Normal Distribution
            mu = np.log(current_price) + (RISK_FREE_RATE - 0.5 * iv**2) * T
            sigma_t = iv * np.sqrt(T)
            
            pdf_values = stats.lognorm.pdf(price_range, s=sigma_t, scale=np.exp(mu))
            
            # Normalize pdf for heatmap intensity (0 to 1)
            if np.max(pdf_values) > 0:
                grid_data.append(pdf_values / np.max(pdf_values)) 
            else:
                grid_data.append(pdf_values)

            processed_dates.append(exp_date)
            
            # 50% Cone Data
            ppf_25 = stats.lognorm.ppf(0.25, s=sigma_t, scale=np.exp(mu))
            ppf_75 = stats.lognorm.ppf(0.75, s=sigma_t, scale=np.exp(mu))
            ranges_50_pct.append({'date': exp_date, 'lower': ppf_25, 'upper': ppf_75})

        except Exception as e:
            continue

    if not grid_data: return None, None, None, None, 0

    avg_iv = np.mean(iv_tracking) if iv_tracking else 0
    heatmap_matrix = np.array(grid_data).T
    
    return heatmap_matrix, price_range, processed_dates, ranges_50_pct, avg_iv

# --- 4. Main Execution ---
if __name__ == "__main__":
    ticker = input("Enter Stock Ticker (e.g., NVDA, SPY): ").upper()
    timeframe_input = input("Enter Time Frame (e.g., 'short', 'daily', 'long'): ")
    
    interval, period, forecast_days = parse_timeframe(timeframe_input)
    
    # Run Data Fetch
    stock, hist, current_price, exp_dates, hist_vol = get_data(ticker, interval, period, forecast_days)
    
    if stock and len(exp_dates) > 0:
        heatmap_matrix, price_range, final_dates, ranges_50_pct, avg_iv = calculate_heatmap(
            stock, current_price, exp_dates, hist
        )
        
        # --- CONSOLE REPORT ---
        print("\n" + "="*40)
        print(f"   ANALYSIS REPORT: {ticker}")
        print("="*40)
        print(f"Current Price:      ${current_price:.2f}")
        print(f"Avg Implied Vol:    {avg_iv*100:.2f}%")
        print(f"Historical Vol:     {hist_vol*100:.2f}%")
        
        # IV Rank / Comparison Logic
        vol_gap = (avg_iv - hist_vol) * 100
        if vol_gap > 5:
            status = "EXPENSIVE (IV > HV). Options selling strategies favored."
        elif vol_gap < -5:
            status = "CHEAP (IV < HV). Options buying strategies favored."
        else:
            status = "FAIRLY PRICED (IV ~ HV)."
        print(f"Pricing Status:     {status}")
        
        # Expected Move Calculation (approximate for the full forecast period)
        # Formula: Price * IV * sqrt(days/365)
        days_out = (final_dates[-1] - datetime.now()).days
        expected_move = current_price * avg_iv * np.sqrt(days_out/365)
        print(f"Expected Move (~{days_out}d): +/- ${expected_move:.2f}")
        print("="*40 + "\n")

        # --- PLOTTING ---
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # 1. Plot Historical Price
        hist_dates = [x.to_pydatetime().replace(tzinfo=None) for x in hist.index]
        ax.plot(hist_dates, hist['Close'], color='white', linewidth=2, label='Historical Price')
        
        # 2. Plot Heatmap
        X, Y = np.meshgrid(final_dates, price_range)
        c = ax.pcolormesh(X, Y, heatmap_matrix, shading='auto', cmap='inferno', alpha=0.9)
        
        # Add Colorbar
        cbar = fig.colorbar(c, ax=ax)
        cbar.set_label('Probability Density (Brighter = More Likely)')
        
        # 3. Plot The 50% Cone
        cone_dates = [hist_dates[-1]] + [x['date'] for x in ranges_50_pct]
        cone_lower = [current_price] + [x['lower'] for x in ranges_50_pct]
        cone_upper = [current_price] + [x['upper'] for x in ranges_50_pct]
        
        ax.plot(cone_dates, cone_lower, color='cyan', linestyle='--', linewidth=1.5, label='25th Percentile')
        ax.plot(cone_dates, cone_upper, color='cyan', linestyle='--', linewidth=1.5, label='75th Percentile')
        
        # 4. Data Dashboard (Text Box)
        # We put this inside the plot so it's captured in screenshots
        info_text = (
            f"{ticker} Analysis\n"
            f"Price: ${current_price:.2f}\n"
            f"IV (Avg): {avg_iv*100:.1f}%\n"
            f"HV (Hist): {hist_vol*100:.1f}%"
        )
        
        # Place text box in top left (relative coordinates 0-1)
        ax.text(0.02, 0.95, info_text, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='black', alpha=0.7, edgecolor='gray'))

        # 5. Price Annotation
        ax.annotate(f'${current_price:.2f}', xy=(hist_dates[-1], current_price), 
                    xytext=(hist_dates[-1] + timedelta(days=5), current_price),
                    arrowprops=dict(facecolor='white', shrink=0.05),
                    color='white', fontsize=10, fontweight='bold')

        # Formatting
        ax.set_title(f"{ticker}: Volatility Forecast ({timeframe_input.capitalize()})", fontsize=16, color='white')
        ax.axhline(current_price, color='gray', linestyle=':', alpha=0.5)
        ax.grid(color='gray', linestyle='-', linewidth=0.2, alpha=0.5) # Faint grid
        ax.legend(loc='lower left')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        print("Displaying Chart...")
        plt.show()
    else:
        print("Could not retrieve enough data to build the chart.")
