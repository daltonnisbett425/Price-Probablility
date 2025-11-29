import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import scipy.stats as si
from datetime import datetime, date
from typing import Tuple, Optional

# -----------------------------------------------------------------------------
# Configuration & Style
# -----------------------------------------------------------------------------
plt.style.use('dark_background')
np.random.seed(42)  # For reproducible results in testing (remove for production)

class CryptoOptionPricer:
    def __init__(self):
        self.ticker = "BTC-USD"
        
    def get_spot_price(self, ticker: str = "BTC-USD") -> float:
        """Fetches live spot price from yfinance with fallback."""
        print(f"\n[Network] Fetching live price for {ticker}...")
        try:
            data = yf.Ticker(ticker)
            history = data.history(period="1d")
            if not history.empty:
                price = history['Close'].iloc[-1]
                print(f"  > Current Spot Price: ${price:,.2f}")
                return price
            else:
                raise ValueError("Empty data returned")
        except Exception as e:
            print(f"  [Error] Failed to fetch price: {e}")
            manual = float(input("  > Please enter current spot price manually: "))
            return manual

    def get_time_to_expiry(self, target_date_str: str) -> float:
        """Calculates years to expiry using 365-day crypto calendar."""
        try:
            target_date = datetime.strptime(target_date_str, "%Y-%m-%d").date()
            today = date.today()
            delta = (target_date - today).days
            
            if delta <= 0:
                raise ValueError("Target date must be in the future.")
            
            # Crypto trades 24/7, so we use 365 days, not 252
            T = delta / 365.0
            print(f"  > Time to Expiry: {delta} days ({T:.4f} years)")
            return T
        except ValueError as ve:
            print(f"  [Error] Date format error: {ve}")
            return self.get_time_to_expiry(input("  > Enter Target Date (YYYY-MM-DD): "))

    def merton_jump_diffusion(self, S0: float, T: float, mu: float, sigma: float, 
                            lam: float, mu_j: float, sigma_j: float, 
                            n_sims: int = 10000, n_steps: int = 365) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulates price paths using Merton Jump Diffusion (MJD).
        
        Parameters:
        - mu: User's expected annualized return (Drift)
        - lam: Lambda (average number of jumps per year)
        - mu_j: Mean jump size (e.g., -0.05 for -5%)
        - sigma_j: Standard deviation of jump size
        """
        dt = T / n_steps
        
        # 1. Diffusion Component (Geometric Brownian Motion)
        # Z1 is the random noise for normal volatility
        Z1 = np.random.standard_normal((n_steps, n_sims))
        
        # 2. Jump Component (Poisson Process + Log-Normal Jumps)
        # Poisson determines IF a jump happens (1 or 0 usually, can be >1)
        # We assume Poisson intensity is scaled by dt
        Poisson = np.random.poisson(lam * dt, (n_steps, n_sims))
        
        # Jump sizes are log-normally distributed: N(mu_j, sigma_j)
        # We generate random jump magnitudes for every step/sim, but only apply them where Poisson > 0
        Jump_Magnitude = np.random.normal(mu_j, sigma_j, (n_steps, n_sims))
        
        # 3. Drift Correction (Compensator)
        # We want the asset to grow at rate 'mu' ON AVERAGE.
        # Since jumps add extra drift (usually negative for crypto crashes), we must correct the 
        # diffusion drift so that E[S_T] = S0 * exp(mu * T).
        # Expected value of jump factor E[k] = exp(mu_j + 0.5*sigma_j^2) - 1
        k = np.exp(mu_j + 0.5 * sigma_j**2) - 1
        drift_correction = lam * k
        
        # Adjusted drift for the diffusion part
        drift_term = (mu - 0.5 * sigma**2 - drift_correction) * dt
        
        # 4. Simulation Loop
        # Using log-returns for numerical stability: ln(St) = ln(S0) + sum(log_returns)
        
        # Diffusion part of log return
        log_ret_diffusion = drift_term + sigma * np.sqrt(dt) * Z1
        
        # Jump part of log return: When jump happens, add the jump size
        log_ret_jumps = Poisson * Jump_Magnitude
        
        # Total log returns at each step
        log_returns = log_ret_diffusion + log_ret_jumps
        
        # Accumulate returns to get price paths
        # cumsum axis=0 implies summing over time steps
        log_path = np.cumsum(log_returns, axis=0)
        
        # Add initial price
        S = S0 * np.exp(np.vstack([np.zeros((1, n_sims)), log_path]))
        
        return S

    def calculate_kelly(self, win_prob: float, market_price: float, bankroll_fraction: float = 1.0) -> float:
        """
        Calculates optimal bet size using Kelly Criterion for Binary Options.
        Formula: f* = (p - C) / (1 - C)
        where p = probability of winning, C = cost of share (decimal odds)
        """
        if market_price >= 1.0 or market_price <= 0:
            return 0.0
            
        edge = win_prob - market_price
        
        if edge <= 0:
            return 0.0
            
        kelly_fraction = edge / (1 - market_price)
        return max(0.0, kelly_fraction * bankroll_fraction)

    def plot_results(self, S: np.ndarray, K: float, T: float, price_yes: float, 
                     prob_yes: float, kelly: float):
        """Generates professional Dark Mode plots."""
        
        final_prices = S[-1, :]
        n_sims = len(final_prices)
        
        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(2, 2)
        
        # Plot 1: Monte Carlo Paths (First 100)
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(S[:, :100], lw=1, alpha=0.6)
        ax1.axhline(K, color='cyan', linestyle='--', lw=1.5, label=f'Strike: ${K:,.0f}')
        ax1.set_title(f'Price Simulation (First 100 Paths) - {T:.2f} Years', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Price ($)')
        ax1.grid(True, alpha=0.2)
        ax1.legend()

        # Plot 2: Terminal Price Distribution
        ax2 = fig.add_subplot(gs[0, 1])
        n, bins, patches = ax2.hist(final_prices, bins=75, color='#39FF14', alpha=0.6, edgecolor='black')
        ax2.axvline(K, color='cyan', linestyle='dashed', linewidth=2)
        ax2.text(K, max(n)*0.8, f' Strike\n ${K:,.0f}', color='cyan', ha='right')
        
        # Highlight the area above strike
        ax2.set_title(f'Terminal Distribution (N={n_sims})', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Price ($)')
        ax2.grid(True, alpha=0.2)

        # Plot 3: Convergence of Probability
        ax3 = fig.add_subplot(gs[1, :])
        # Calculate running probability
        bool_above = S[-1, :] > K
        running_prob = np.cumsum(bool_above) / np.arange(1, n_sims + 1)
        
        ax3.plot(running_prob, color='white', lw=1.5)
        ax3.axhline(prob_yes, color='#39FF14', linestyle='--', label=f'Model Prob: {prob_yes*100:.2f}%')
        ax3.axhline(price_yes, color='red', linestyle=':', linewidth=2, label=f'Polymarket Price: {price_yes*100:.1f}¢')
        
        ax3.set_title('Probability Convergence vs Market Price', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Number of Simulations')
        ax3.set_ylabel('Probability')
        ax3.legend()
        ax3.grid(True, alpha=0.2)
        
        plt.tight_layout()
        
        # Print Summary to Console
        print("\n" + "="*50)
        print(f"  RESULTS SUMMARY (Strike: ${K:,.0f})")
        print("="*50)
        print(f"  Model Probability (Fair Value) : {prob_yes*100:.2f}%")
        print(f"  Polymarket Price               : {price_yes*100:.1f}¢")
        print(f"  Edge                           : {(prob_yes - price_yes)*100:.2f}%")
        print("-" * 30)
        print(f"  KELLY BET SIZE                 : {kelly*100:.2f}% of Bankroll")
        print("="*50 + "\n")
        
        plt.show()

    def run(self):
        print("--- POLYMARKET VALUE FINDER (MERTON JUMP DIFFUSION) ---")
        
        # 1. Inputs
        S0 = self.get_spot_price(self.ticker)
        
        target_date = input("Enter Target Date (YYYY-MM-DD): ")
        T = self.get_time_to_expiry(target_date)
        
        try:
            K = float(input("Enter Strike Price ($): "))
            sigma = float(input("Enter Implied Volatility (decimal, e.g. 0.65): "))
            mu = float(input("Enter Expected Drift (decimal, e.g. 0.20 for 20%): "))
            
            print("\n[Jump Parameters - Press Enter for Defaults]")
            lam_in = input("  Lambda (Jumps/Year, default 1.0): ")
            lam = float(lam_in) if lam_in else 1.0
            
            mu_j_in = input("  Mean Jump Size (decimal, default -0.05): ")
            mu_j = float(mu_j_in) if mu_j_in else -0.05
            
            sigma_j_in = input("  Jump Volatility (decimal, default 0.10): ")
            sigma_j = float(sigma_j_in) if sigma_j_in else 0.10
            
            poly_price = float(input("\nCurrent Polymarket 'Yes' Price (decimal, e.g. 0.34): "))
            
        except ValueError:
            print("[Error] Invalid numeric input.")
            return

        # 2. Simulation
        print("\n> Running Merton Jump Diffusion Simulation...")
        S = self.merton_jump_diffusion(S0, T, mu, sigma, lam, mu_j, sigma_j)
        
        # 3. Analysis
        final_prices = S[-1, :]
        prob_yes = np.mean(final_prices > K)
        kelly = self.calculate_kelly(prob_yes, poly_price, bankroll_fraction=0.5) # Half-Kelly for safety
        
        # 4. Plot
        self.plot_results(S, K, T, poly_price, prob_yes, kelly)

if __name__ == "__main__":
    app = CryptoOptionPricer()
    app.run()
