import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from datetime import datetime, timedelta

# --- Configuration ---
TICKER = "SPY"
RISK_FREE_RATE = 0.045

def get_option_data(ticker_symbol):
    print(f"--- Fetching data for {ticker_symbol} ---")
    stock = yf.Ticker(ticker_symbol)

    # 1. Safer Price Fetching
    try:
        hist = stock.history(period="1d")
        if not hist.empty:
            current_price = hist['Close'].iloc[-1]
        else:
            current_price = stock.fast_info.last_price
    except Exception as e:
        print(f"Error fetching price: {e}")
        return None, None, None

    print(f"Current Price: ${current_price:.2f}")

    # 2. Get Expirations
    try:
        expirations = stock.options
        if not expirations:
            print("No option expirations found!")
            return None, None, None
    except Exception as e:
        print(f"Error fetching expirations: {e}")
        return None, None, None

    # Filter dates (Next 7 to 180 days)
    valid_dates = []
    for e in expirations:
        try:
            d = datetime.strptime(e, "%Y-%m-%d")
            days_out = (d - datetime.now()).days
            if 7 < days_out < 180:
                valid_dates.append(e)
        except:
            continue

    return stock, current_price, valid_dates[:12]

def calculate_probability_density(stock, current_price, exp_dates):
    grid_data = []
    processed_dates = []
    ranges_50_pct = []

    # Define Y-axis range
    price_range = np.linspace(current_price * 0.7, current_price * 1.3, 200)

    print(f"Processing {len(exp_dates)} expiration dates...")

    for date_str in exp_dates:
        try:
            # Fetch chain
            opt = stock.option_chain(date_str)
            calls = opt.calls
            puts = opt.puts

            if calls.empty or puts.empty:
                continue

            # Calculate T (years)
            exp_date = datetime.strptime(date_str, "%Y-%m-%d")
            T = (exp_date - datetime.now()).days / 365.0
            if T <= 0.01: continue

            # Find ATM Implied Volatility
            calls['dist'] = abs(calls['strike'] - current_price)
            puts['dist'] = abs(puts['strike'] - current_price)

            atm_iv_call = calls.sort_values('dist').iloc[0]['impliedVolatility']
            atm_iv_put = puts.sort_values('dist').iloc[0]['impliedVolatility']

            iv = (atm_iv_call + atm_iv_put) / 2
            if iv == 0 or np.isnan(iv): iv = 0.15

            # Black-Scholes / Log-Normal Logic
            mu = np.log(current_price) + (RISK_FREE_RATE - 0.5 * iv**2) * T
            sigma_t = iv * np.sqrt(T)

            # Generate PDF
            pdf_values = stats.lognorm.pdf(price_range, s=sigma_t, scale=np.exp(mu))
            pdf_normalized = pdf_values / np.max(pdf_values)
            
            grid_data.append(pdf_normalized)
            processed_dates.append(exp_date)

            # Calculate 50% Lines
            ppf_25 = stats.lognorm.ppf(0.25, s=sigma_t, scale=np.exp(mu))
            ppf_75 = stats.lognorm.ppf(0.75, s=sigma_t, scale=np.exp(mu))

            ranges_50_pct.append({'date': exp_date, 'lower': ppf_25, 'upper': ppf_75})

        except Exception as e:
            print(f"Failed on {date_str}: {e}")
            continue

    if not grid_data:
        raise ValueError("Could not generate any probability data.")

    # Transpose for plotting
    heatmap_matrix = np.array(grid_data).T
    return heatmap_matrix, price_range, processed_dates, ranges_50_pct

# --- Execution Block ---
if __name__ == "__main__":
    try:
        stock, current_price, exp_dates = get_option_data(TICKER)

        if stock and exp_dates:
            heatmap_matrix, price_range, final_dates, ranges_50_pct = calculate_probability_density(stock, current_price, exp_dates)

            print("Generating Plot...")
            fig, ax = plt.subplots(figsize=(12, 8))
            plt.style.use('dark_background')

            X, Y = np.meshgrid(final_dates, price_range)
            c = ax.pcolormesh(X, Y, heatmap_matrix, shading='auto', cmap='viridis')

            # Plot Lines
            dates_line = [x['date'] for x in ranges_50_pct]
            lower_line = [x['lower'] for x in ranges_50_pct]
            upper_line = [x['upper'] for x in ranges_50_pct]

            ax.plot(dates_line, lower_line, color='red', lw=2, ls='--', label='25th Percentile')
            ax.plot(dates_line, upper_line, color='red', lw=2, ls='--', label='75th Percentile')

            ax.set_title(f"{TICKER} Option-Implied Probability", color='white')
            ax.axhline(current_price, color='white', linestyle=':', alpha=0.5)

            cbar = plt.colorbar(c, ax=ax)
            cbar.set_label("Probability Density")

            plt.legend()
            plt.tight_layout()
            plt.show()
            print("Done.")
        else:
            print("Could not initialize data.")

    except Exception as e:
        print("\nCRITICAL ERROR:")
        print(e)
        import traceback
        traceback.print_exc()
