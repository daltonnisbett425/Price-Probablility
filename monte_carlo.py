import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as si
import yfinance as yf
import pandas as pd

def get_market_data(ticker):
    """
    Fetches live data from Yahoo Finance:
    1. Current Spot Price (S0)
    2. Annualized Volatility (sigma) based on last 1 year of returns
    """
    try:
        print(f"  [Network] Fetching live data for {ticker}...")
        stock = yf.Ticker(ticker)
        
        # Get 1 year of history to calculate volatility
        hist = stock.history(period="1y")
        
        if hist.empty:
            raise ValueError("No data found")

        # 1. Get current price (Last available close)
        S0 = hist['Close'].iloc[-1]
        
        # 2. Calculate Annualized Volatility
        # Log returns = ln(Price_t / Price_{t-1})
        hist['Log_Ret'] = np.log(hist['Close'] / hist['Close'].shift(1))
        # Std Dev of log returns * sqrt(252 trading days)
        sigma = hist['Log_Ret'].std() * np.sqrt(252)
        
        return S0, sigma
        
    except Exception as e:
        print(f"  [Error] Could not fetch data: {e}")
        return None, None

def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = (S * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0))
    return call_price

def run_simulation(ticker, S0, K, T, r, sigma, n_simulations=1000, n_steps=252):
    dt = T / n_steps
    Z = np.random.standard_normal((n_steps + 1, n_simulations))
    S = np.zeros_like(Z)
    S[0] = S0
    
    for t in range(1, n_steps + 1):
        S[t] = S[t-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[t])

    payoffs = np.maximum(S[-1] - K, 0)
    option_price_mc = np.exp(-r * T) * np.mean(payoffs)
    running_avg = np.cumsum(payoffs * np.exp(-r * T)) / np.arange(1, n_simulations + 1)

    return S, payoffs, running_avg, option_price_mc

def plot_results(ticker, S, payoffs, running_avg, option_price_mc, bs_price, S0, K, sigma):
    plt.style.use('dark_background')
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    plt.subplots_adjust(hspace=0.4)

    # Plot 1: Price Paths
    axes[0].plot(S[:, :50], lw=1, alpha=0.7)
    axes[0].axhline(K, color='red', linestyle='--', linewidth=1.5, label=f'Strike: ${K:.2f}')
    axes[0].set_title(f'Live Simulation: {ticker} (Start: ${S0:.2f})', fontsize=14, color='white')
    axes[0].set_ylabel('Stock Price ($)')
    axes[0].legend(loc='upper left')
    axes[0].grid(True, alpha=0.2)

    # Plot 2: Distribution
    axes[1].hist(payoffs, bins=50, color='cyan', alpha=0.6, edgecolor='white')
    axes[1].axvline(np.mean(payoffs), color='magenta', linestyle='dashed', linewidth=2, label=f'Mean Payoff')
    axes[1].set_title(f'Projected Payoff Distribution (Vol: {sigma*100:.1f}%)', fontsize=14)
    axes[1].set_ylabel('Frequency')
    axes[1].grid(True, alpha=0.2)

    # Plot 3: Convergence
    axes[2].plot(running_avg, color='lime', lw=1.5)
    axes[2].axhline(option_price_mc, color='red', linestyle='--', label=f'MC Price: ${option_price_mc:.2f}')
    axes[2].axhline(bs_price, color='white', linestyle=':', label=f'BS Theor. Price: ${bs_price:.2f}')
    axes[2].set_title('Monte Carlo Convergence', fontsize=14)
    axes[2].set_xlabel('Simulations')
    axes[2].legend()
    axes[2].grid(True, alpha=0.2)

    print(f"\n[RESULT] Monte Carlo Fair Value: ${option_price_mc:.4f}")
    plt.show()

def main():
    print("--------------------------------------------------")
    print("   LIVE MARKET DATA OPTION PRICER (yfinance)      ")
    print("--------------------------------------------------")
    
    while True:
        user_input = input("\nWhich stock do you want to simulate? (e.g. 'Simulate NVDA'): ").lower()

        if user_input in ['exit', 'quit']:
            break

        # Attempt to extract a ticker from the input
        # Looks for the last word in the sentence usually being the ticker if simple
        potential_ticker = user_input.split()[-1].upper()
        
        # Basic check if it looks like a ticker (2-5 letters)
        if len(potential_ticker) < 2 or len(potential_ticker) > 5 or any(char.isdigit() for char in potential_ticker):
             potential_ticker = "SPY" # Default if detection fails

        print(f"\n> Pulling live data for: {potential_ticker}")
        
        S0, sigma = get_market_data(potential_ticker)
        
        if S0 is not None:
            print(f"  [Data] Current Price: ${S0:.2f}")
            print(f"  [Data] Historical Volatility (1yr): {sigma*100:.2f}%")
            
            try:
                K = float(input(f"  Strike Price (default ${round(S0*1.1, 2)}): ") or round(S0*1.1, 2))
                T = float(input("  Time to Expiration (Years, default 1.0): ") or 1.0)
                r = 0.045 # Fixed risk free rate for simplicity
                
                print("\n> Running Simulation...")
                bs_price = black_scholes_call(S0, K, T, r, sigma)
                S, payoffs, running_avg, mc_price = run_simulation(potential_ticker, S0, K, T, r, sigma)
                plot_results(potential_ticker, S, payoffs, running_avg, mc_price, bs_price, S0, K, sigma)
                
            except ValueError:
                print("  [Error] Invalid numbers entered.")
        else:
            print("  [Error] Ticker not found. Try again.")

if __name__ == "__main__":
    main()
