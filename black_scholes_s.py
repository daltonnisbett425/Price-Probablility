import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as si
from datetime import datetime, timedelta

# ==========================================
# CONFIGURATION (Based on your Screenshot)
# ==========================================
TICKER = 'GOOG'
STRIKE_PRICE = 300.00
EXPIRATION_DATE = '2026-01-02'
RISK_FREE_RATE = 0.0425  # Approx 4.25% (10-year treasury yield)

# ==========================================
# 1. DATA FETCHING & PREP
# ==========================================
def fetch_data(ticker):
    print(f"--- Fetching data for {ticker} ---")
    # Get 1 year of history to calculate volatility
    stock = yf.Ticker(ticker)
    hist = stock.history(period="1y")
    current_price = hist['Close'].iloc[-1]
    
    # Calculate Log Returns
    hist['Log Returns'] = np.log(hist['Close'] / hist['Close'].shift(1))
    
    # Calculate Historical Volatility (Annualized)
    # 252 trading days in a year
    historical_volatility = np.sqrt(252) * hist['Log Returns'].std()
    
    print(f"Current Price: ${current_price:.2f}")
    print(f"Annualized Historical Volatility: {historical_volatility:.2%}")
    
    return current_price, historical_volatility, hist

# ==========================================
# 2. BLACK-SCHOLES CALCULATOR
# ==========================================
def black_scholes_metrics(S, K, T, r, sigma, option_type="call"):
    """
    S: Spot price
    K: Strike price
    T: Time to maturity (years)
    r: Risk-free interest rate
    sigma: Volatility of underlying asset
    """
    
    # d1 and d2 calculations
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - (sigma * np.sqrt(T))
    
    # Cumulative distribution function
    N_d1 = si.norm.cdf(d1, 0.0, 1.0)
    N_d2 = si.norm.cdf(d2, 0.0, 1.0)
    
    # Option Price & Greeks
    if option_type == "call":
        price = (S * N_d1) - (K * np.exp(-r * T) * N_d2)
        delta = N_d1
        # Probability of expiring ITM (Risk-Neutral)
        prob_itm = N_d2 
    else:
        price = (K * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0)) - (S * si.norm.cdf(-d1, 0.0, 1.0))
        delta = -si.norm.cdf(-d1, 0.0, 1.0)
        prob_itm = si.norm.cdf(-d2, 0.0, 1.0)

    # Other Greeks (Same for Call/Put mostly, simplified)
    gamma = si.norm.pdf(d1, 0.0, 1.0) / (S * sigma * np.sqrt(T))
    vega = S * si.norm.pdf(d1, 0.0, 1.0) * np.sqrt(T)
    theta = -(S * si.norm.pdf(d1, 0.0, 1.0) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * N_d2

    return {
        "Theoretical Price": price,
        "Probability ITM (BSM)": prob_itm,
        "Delta": delta,
        "Gamma": gamma,
        "Theta": theta,
        "Vega": vega
    }

# ==========================================
# 3. MONTE CARLO SIMULATION (The "More is Better" Part)
# ==========================================
def monte_carlo_simulation(S, mu, sigma, T, num_simulations=10000, num_steps=252):
    dt = T / num_steps
    simulation_results = np.zeros(num_simulations)
    
    print(f"--- Running {num_simulations} Monte Carlo Simulations ---")
    
    # We simulate price paths
    for i in range(num_simulations):
        price_path = [S]
        for _ in range(int(num_steps * T)): # Adjust steps for actual time remaining
            # Geometric Brownian Motion
            drift = (mu - 0.5 * sigma**2) * dt
            shock = sigma * np.sqrt(dt) * np.random.normal()
            price = price_path[-1] * np.exp(drift + shock)
            price_path.append(price)
        simulation_results[i] = price_path[-1]
        
    return simulation_results

# ==========================================
# MAIN EXECUTION
# ==========================================

# 1. Get Data
current_price, volatility, history = fetch_data(TICKER)

# 2. Calculate Time to Expiry (T) in Years
today = datetime.now()
expiry = datetime.strptime(EXPIRATION_DATE, "%Y-%m-%d")
days_to_expiry = (expiry - today).days
T = days_to_expiry / 365.25

if T <= 0:
    print("Option has already expired!")
    exit()

print(f"Days to Expiration: {days_to_expiry}")

# 3. Run Black Scholes
bsm_stats = black_scholes_metrics(current_price, STRIKE_PRICE, T, RISK_FREE_RATE, volatility)

# 4. Run Monte Carlo (using historical return mean as drift)
# Note: BSM assumes risk-neutral drift (r), but for "real world" odds we often check historical drift.
# However, for safety, we often use r or 0 drift for conservative estimates. 
# Here we use the Risk Free Rate as the drift for standard comparison.
sim_results = monte_carlo_simulation(current_price, RISK_FREE_RATE, volatility, T)

# Calculate Monte Carlo Probabilities
itm_count = np.sum(sim_results > STRIKE_PRICE)
mc_prob_itm = itm_count / len(sim_results)

# ==========================================
# OUTPUT & VISUALIZATION
# ==========================================
print("\n" + "="*40)
print(f"ANALYSIS FOR {TICKER} ${STRIKE_PRICE} CALL ({EXPIRATION_DATE})")
print("="*40)
print(f"Black-Scholes Theoretical Price: ${bsm_stats['Theoretical Price']:.2f}")
print(f"BSM Probability of Profit (ITM): {bsm_stats['Probability ITM (BSM)']:.2%}")
print(f"Monte Carlo Probability (ITM):   {mc_prob_itm:.2%}")
print("-" * 40)
print("THE GREEKS:")
print(f"Delta (Sensitivity to Stock Price): {bsm_stats['Delta']:.4f}")
print(f"Gamma (Rate of change of Delta):    {bsm_stats['Gamma']:.4f}")
print(f"Theta (Time Decay per day):         {bsm_stats['Theta']/365:.4f}")
print(f"Vega  (Sensitivity to Volatility):  {bsm_stats['Vega']/100:.4f}")
print("="*40)

# Plotting
plt.figure(figsize=(12, 6))

# Plot History
plt.plot(history.index, history['Close'], label='Historical Price', color='blue')

# Plot projected "Cone"
last_date = history.index[-1]
future_dates = [last_date + timedelta(days=x) for x in range(days_to_expiry + 1)]

# Project upper and lower bounds (1 and 2 std deviations)
# Expected move = Price * Volatility * sqrt(Time)
sigma_move = current_price * volatility * np.sqrt(np.linspace(0, T, days_to_expiry + 1))
upper_1std = current_price * np.exp(RISK_FREE_RATE * np.linspace(0, T, days_to_expiry+1)) + sigma_move
lower_1std = current_price * np.exp(RISK_FREE_RATE * np.linspace(0, T, days_to_expiry+1)) - sigma_move

plt.plot(future_dates, upper_1std, 'g--', label='+1 Std Dev (Upper)')
plt.plot(future_dates, lower_1std, 'r--', label='-1 Std Dev (Lower)')
plt.axhline(y=STRIKE_PRICE, color='k', linestyle='-', linewidth=2, label=f'Strike (${STRIKE_PRICE})')

plt.title(f"{TICKER} Price History & Probability Cone to {EXPIRATION_DATE}")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Histogram of Monte Carlo Results
plt.figure(figsize=(10, 5))
plt.hist(sim_results, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
plt.axvline(STRIKE_PRICE, color='red', linestyle='dashed', linewidth=2, label=f'Strike ${STRIKE_PRICE}')
plt.title(f"Monte Carlo Distribution of {TICKER} Price on {EXPIRATION_DATE}")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.legend()
plt.show()
