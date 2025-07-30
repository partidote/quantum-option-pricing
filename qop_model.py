"""
Quantum Option Pricing Model
----------------------------

Authors: Aaditeya Tripathi & Anany Pravin
Date: July 30, 2025

Description:
This script implements a quantum-inspired option pricing engine using Discrete-Time Quantum Walks (DTQW)
and Iterative Amplitude Estimation (IAE) to price European call options. It compares results against the
Black-Scholes-Merton model and provides strike/volatility sweeps and real market data integration.

Functional Highlights:
-----------------------
1. **Volatility Estimation**: Uses historical price data to compute annualized volatility.
2. **Quantum Walk Construction**: Builds a walk circuit over log-price space with bias from interest rate and volatility.
3. **Payoff Oracle**: Encodes piecewise-constant call option payoffs.
4. **Amplitude Estimation**: Applies IAE to determine expected option value.
5. **BSM Benchmarking**: Computes the closed-form BSM price for direct comparison.
6. **Strike & Volatility Sweeps**: Includes functions to run pricing across a grid of strikes or volatilities.
7. **Market Data Integration**: Pulls real-time market data via yFinance.

Output:
--------
- Quantum option price (`quantum_price`)
- Classical BSM price (`bsm_price`)
- Visualization tools for price comparison across parameters
- Reusable circuit objects for experimentation

Dependencies: Qiskit, NumPy, SciPy, Matplotlib, yFinance, Pandas
"""

import datetime
import numpy as np
import pandas as pd          
import yfinance as yf
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit.circuit.library import LinearAmplitudeFunction
from qiskit_algorithms import IterativeAmplitudeEstimation, EstimationProblem
from qiskit_algorithms.utils import algorithm_globals
from qiskit.primitives import Sampler
from scipy.stats import norm

# ---------------- HISTORICAL VOLATILITY ----------------
def fetch_historical_prices(ticker: str, start: str = None, end: str = None) -> np.ndarray:
    def try_fetch(**kwargs):
        df = yf.download(ticker, progress=False, auto_adjust=False, **kwargs)
        if df.empty or 'Close' not in df:
            return np.array([])
        closes = df['Close'].dropna().values
        return closes if len(closes) >= 2 else np.array([])

    for kwargs in [{"start": start, "end": end}, {"period": "1y"}, {"period": "2y"}]:
        closes = try_fetch(**kwargs)
        if len(closes) >= 2:
            return closes
    return np.array([])

def annualized_volatility(closes: np.ndarray) -> float:
    if closes.ndim > 1 and closes.shape[1] == 1:
        closes = closes.flatten()
    if len(closes) < 2:
        return float('nan')
    log_rets = np.diff(np.log(closes))
    if len(log_rets) < 2 or not np.isfinite(log_rets).all():
        return float('nan')
    sigma_daily = np.std(log_rets, ddof=1)
    sigma_annual = sigma_daily * np.sqrt(252)
    return float(sigma_annual) if np.isfinite(sigma_annual) else float('nan')

# ---------------- QUANTUM MODULE ----------------
def compute_ideal_num_steps(sigma, T, S0, num_pos_qubits, span_sigma=3):
    grid_points = 2 ** num_pos_qubits
    L = span_sigma * sigma * np.sqrt(T)
    log_range = 2 * L
    ideal_steps = (sigma * 4) * T * (grid_points * 2) / (log_range ** 2)
    return 4*max(1, int(np.round(ideal_steps)))

def compute_optimal_theta(r, sigma, T, num_steps):
    mu_rn = r - 0.5 * sigma ** 2
    p = 1 / (1 + np.exp(-2 * mu_rn / sigma ** 2))
    return 2 * np.arcsin(np.sqrt(p))

def generate_grid_and_payoffs(S0, K, sigma, T, num_pos_qubits, num_steps):
    num_values = 2 ** num_pos_qubits
    delta = sigma * np.sqrt(T / num_steps)
    x_vals = np.arange(num_values) - (num_values // 2)
    log_prices = x_vals * delta
    prices = S0 * np.exp(log_prices)
    payoffs = np.maximum(prices - K, 0)
    max_payoff = np.max(payoffs)
    normalized_payoffs = payoffs / max_payoff if max_payoff != 0 else payoffs
    return prices, payoffs, normalized_payoffs, max_payoff

def build_quantum_walk(pos_qubits, coin_qubit, steps, theta):
    walk = QuantumCircuit(1 + len(pos_qubits))
    walk.ry(theta, coin_qubit)
    walk.h(pos_qubits)
    for _ in range(steps):
        walk.ry(theta, coin_qubit)
        for i in range(len(pos_qubits)):
            walk.mcx([coin_qubit] + pos_qubits[:i], pos_qubits[i])
        walk.x(coin_qubit)
        for i in reversed(range(len(pos_qubits))):
            walk.mcx([coin_qubit] + pos_qubits[:i], pos_qubits[i])
        walk.x(coin_qubit)
    return walk

# ----------------------------------------------------------------------
# EXACT payoff oracle – one (slope, offset) pair per lattice point
# ----------------------------------------------------------------------
def build_piecewise_payoff(normalized_payoffs, num_pos_qubits):
    """
    Return a LinearAmplitudeFunction whose value on each basis state |x>
    equals normalized_payoffs[x].  Works by giving slope = 0 and a unique
    offset for every x (i.e. piece-wise constant segments of length 1).
    """
    num_values = 2 ** num_pos_qubits          # 16 when 4 qubits
    slopes      = [0.0] * num_values          # flat
    offsets     = normalized_payoffs.tolist() # exact value at each node
    breakpoints = list(range(num_values))     # 0,1,2,…,15

    return LinearAmplitudeFunction(
        num_state_qubits = num_pos_qubits,
        slope            = slopes,
        offset           = offsets,
        domain           = (0, num_values - 1),
        image            = (0, 1),
        breakpoints      = breakpoints,
        name             = "ExactPayoff"
    )

def build_pricing_circuit(params):
    num_steps = compute_ideal_num_steps(params['sigma'], params['T'], params['S0'], params['num_pos_qubits'], params['span_sigma'])
    theta = compute_optimal_theta(params['r'], params['sigma'], params['T'], num_steps)
    prices, payoffs, normalized_payoffs, max_payoff = generate_grid_and_payoffs(
        params['S0'], params['K'], params['sigma'], params['T'], params['num_pos_qubits'], num_steps)

    coin_qubit = 0
    pos_qubits = list(range(1, 1 + params['num_pos_qubits']))
    piecewise_func = build_piecewise_payoff(normalized_payoffs, params['num_pos_qubits'])

    ancilla_qubit = 1 + params['num_pos_qubits']
    extra_qubits = piecewise_func.num_qubits - (params['num_pos_qubits'] + 1)
    total_qubits = 1 + params['num_pos_qubits'] + 1 + extra_qubits

    qc = QuantumCircuit(total_qubits)
    walk = build_quantum_walk(pos_qubits, coin_qubit, num_steps, theta)
    qc.compose(walk, qubits=range(walk.num_qubits), inplace=True)

    payoff_qubit_map = pos_qubits + [ancilla_qubit] + list(range(ancilla_qubit + 1, total_qubits))
    payoff_gate = piecewise_func.to_gate(label="Payoff")
    qc.append(payoff_gate, payoff_qubit_map)

    return qc, normalized_payoffs, max_payoff, params['num_pos_qubits'], ancilla_qubit

# ---------------- IAE SIMULATION ----------------
def simulate_with_iae(qc, normalized_payoffs, max_payoff, num_pos_qubits, r, T, ancilla_index):
    algorithm_globals.random_seed = 123
    sampler = Sampler()
    objective_qubits = [ancilla_index]

    problem = EstimationProblem(
        state_preparation=qc,
        objective_qubits=objective_qubits
    )

    iae = IterativeAmplitudeEstimation(
        epsilon_target=0.01,
        alpha=0.05,
        sampler=sampler
    )

    result = iae.estimate(problem)
    mean_amp = result.estimation
    quantum_price = mean_amp * max_payoff * np.exp(-r * T)
    return quantum_price, mean_amp

# ---------------- MAIN DRIVER ----------------
def run_with_real_market_data(
    ticker: str,
    option_strike: float = None,
    expiration_date: str = None,
    hist_start: str = None,
    hist_end: str = None,
    num_pos_qubits: int = 4,
    verbose: bool = True
):
    today = datetime.date.today()
    if hist_end is None:
        hist_end = today.strftime("%Y-%m-%d")
    if hist_start is None:
        one_year_ago = today - datetime.timedelta(days=365)
        hist_start = one_year_ago.strftime("%Y-%m-%d")

    closes = fetch_historical_prices(ticker, start=hist_start, end=hist_end)
    if len(closes) < 2:
        raise ValueError("Not enough historical data to compute volatility.")

    sigma = annualized_volatility(closes)

    if np.isnan(sigma) or sigma <= 0:
        raise ValueError("Invalid historical data to compute volatility.")

    S0 = closes[-1].item() if np.ndim(closes[-1]) == 0 else float(np.squeeze(closes[-1]))
    if option_strike is None:
        option_strike = S0 * 1.02
    if expiration_date is None:
        raise ValueError("You must specify an expiration_date in YYYY-MM-DD format.")

    exp_dt = datetime.datetime.strptime(expiration_date, "%Y-%m-%d").date()
    T_days = (exp_dt - today).days
    if T_days <= 0:
        raise ValueError("Expiration date must be in the future.")
    T = T_days / 365.0
    r = 0.005

    span_sigma = 3.0
    #if T > 2.0:
        #print(f"⏳ T = {T:.2f} is large → reducing span_sigma to prevent tail blow-up.")
        #span_sigma = 2.0  # Adaptive compression of the grid

    params = {
        "S0": S0,
        "K": option_strike,
        "r": r,
        "T": T,
        "sigma": sigma,
        "num_pos_qubits": num_pos_qubits,
        "span_sigma": 3,
    }

    # ------------------------------------------------------------
    # build circuit & price
    # ------------------------------------------------------------
    qc, normalized_payoffs, max_payoff, nq, ancilla_index = build_pricing_circuit(params)
    price, mean_amp = simulate_with_iae(
        qc, normalized_payoffs, max_payoff, nq, r, T, ancilla_index
    )

    # Black-Scholes benchmark
    d1 = (np.log(S0 / option_strike) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    bsm = S0 * norm.cdf(d1) - option_strike * np.exp(-r * T) * norm.cdf(d2)

    # ------------------------------------------------------------
    # optional console output
    # ------------------------------------------------------------
    if verbose:
        print("----- Quantum Pricing Result (IAE) -----")
        print(f"Normalized expected payoff   = {mean_amp:.6f}")
        print(f"Quantum Option Price (disc.) = {price:.4f}")
        print(f"Black-Scholes Price          = {bsm:.4f}\n")
        print(f"Strike Price (K):            = {option_strike:.2f}")

    # ------------------------------------------------------------
    # return structured result
    # ------------------------------------------------------------
    return {
        "S0":            S0,
        "sigma":         sigma,
        "T":             T,
        "quantum_price": price,
        "bsm_price":     bsm,
        "circuit":       qc,
    }


# ----------------------------------------------------------------------
# PLOT: Quantum-walk call price vs Black-Scholes across a strike grid
# ----------------------------------------------------------------------

def plot_quantum_vs_bsm_strike_sweep(
    ticker: str               = "NVDA",
    expiry_days: int          = 30,          # days until option expiry
    strike_range: tuple       = (0.8, 1.2),  # strikes as % of spot
    num_strikes: int          = 41,          # grid resolution
    num_pos_qubits: int       = 4,           # lattice size (16 nodes)
):
    """
    Uses `run_with_real_market_data` once per strike and draws both curves.
    Assumes the exact-payoff oracle (no `num_segments` argument needed).
    """
    # ---------- market snapshot ----------
    closes = fetch_historical_prices(ticker)
    if len(closes) < 2:
        raise ValueError("Not enough data to compute volatility.")
    S0    = float(closes[-1])
    sigma = annualized_volatility(closes)
    T     = expiry_days / 365
    r     = 0.005

    # ---------- strike grid ----------
    K_vals = np.linspace(strike_range[0] * S0,
                         strike_range[1] * S0,
                         num_strikes)

    quantum_prices, bsm_prices = [], []

    expiry_date_str = (datetime.date.today() +
                       datetime.timedelta(days=expiry_days)
                       ).strftime("%Y-%m-%d")

    for K in K_vals:
        res = run_with_real_market_data(
            ticker          = ticker,
            option_strike   = float(K),
            expiration_date = expiry_date_str,
            num_pos_qubits  = num_pos_qubits,
        )
        quantum_prices.append(res["quantum_price"])
        bsm_prices.append(res["bsm_price"])

    # ---------- plot ----------
    plt.figure(figsize=(8, 5))
    plt.plot(K_vals, bsm_prices,    linestyle="--", label="Black-Scholes")
    plt.plot(K_vals, quantum_prices, marker="o",   label="Quantum Walk")
    plt.xlabel("Strike K")
    plt.ylabel("Call Price")
    plt.title(f"{ticker} : Quantum vs BSM (T = {expiry_days} d)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def table_and_plot_strike_sweep(
    ticker: str,
    expiry_days: int      = 250,
    strike_range: tuple   = (0.8, 1.2),
    num_strikes: int      = 41,
    num_pos_qubits: int   = 4,
    num_segments: int     = 12,
):
    """
    Builds a pandas DataFrame of Quantum-walk vs Black-Scholes prices
    across strike values, prints the table, then plots both curves.
    """
    closes = fetch_historical_prices(ticker)
    if len(closes) < 2:
        raise ValueError("Not enough data.")
    S0     = float(closes[-1])
    sigma  = annualized_volatility(closes)

    K_vals = np.linspace(strike_range[0]*S0, strike_range[1]*S0, num_strikes)
    expiry_str = (datetime.date.today() +
                  datetime.timedelta(days=expiry_days)
                 ).strftime("%Y-%m-%d")

    rows = []
    for K in K_vals:
        res = run_with_real_market_data(
            ticker          = ticker,
            option_strike   = float(K),
            expiration_date = expiry_str,
            hist_start      = None,
            hist_end        = None,
            num_pos_qubits  = num_pos_qubits,
            verbose         = False        # <-- suppress per-strike prints
        )
        rows.append({
            "Strike": round(K, 2),
            "QuantumPrice": res["quantum_price"],
            "BSM":          res["bsm_price"],
        })

    df = pd.DataFrame(rows)
    print("\n=== Call-price table ===")
    print(df.to_string(index=False))

    # ---- plot ----
    plt.figure(figsize=(8,5))
    plt.plot(df["Strike"], df["BSM"],         "--", label="Black-Scholes")
    plt.plot(df["Strike"], df["QuantumPrice"], "o-", label="Quantum Walk")
    plt.xlabel("Strike K"); plt.ylabel("Call Price")
    plt.title(f"{ticker} : Quantum vs BSM (T={expiry_days} d)")
    plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()

    return df


def sweep_volatility_vs_price(ticker, expiry_days, sigma_range=(0.1, 1.0), num_sigma=10, span_sigma=3):
    from copy import deepcopy
    closes = fetch_historical_prices(ticker)
    S0 = closes[-1]
    r = 0.005
    T = expiry_days / 365
    K = S0 * 1.05  # Fixed strike

    sigmas = np.linspace(*sigma_range, num_sigma)
    quantum_prices, bsm_prices = [], []

    for sigma in sigmas:
        params = {
            "S0": S0, "K": K, "r": r, "T": T,
            "sigma": sigma, "num_pos_qubits": 4, "span_sigma": span_sigma,
        }
        qc, norm_p, max_p, nq, anc = build_pricing_circuit(params)
        price, _ = simulate_with_iae(qc, norm_p, max_p, nq, r, T, anc)
        quantum_prices.append(price)

        d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        bsm = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        bsm_prices.append(bsm)

    # Plotting
    plt.figure(figsize=(8, 5))
    plt.plot(sigmas, bsm_prices, "--", label="Black-Scholes")
    plt.plot(sigmas, quantum_prices, "o-", label="Quantum Walk")
    plt.xlabel("Volatility (σ)")
    plt.ylabel("Call Price")
    plt.title(f"{ticker}: Quantum vs BSM Across Volatility (span_sigma={span_sigma})")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()



# ---------------- EXAMPLE ----------------
if __name__ == "__main__":
    
    my_ticker = "NVDA"
    my_days = 30

    # Store return value in 'result'
    result = run_with_real_market_data(
        ticker= my_ticker,
        expiration_date=(datetime.date.today()
                         + datetime.timedelta(days=my_days)).strftime("%Y-%m-%d"),
        num_pos_qubits=4,
        verbose=True
    )

    print(f"Stock: {my_ticker} | Current Price (S0): {result['S0']:.2f}")
    print(f"Annualized volatility for {my_ticker} = {result['sigma']:.4f}\n")

    # strike sweep
    table_and_plot_strike_sweep(
        ticker= my_ticker,
        expiry_days=my_days,
        strike_range=(0.8, 1.2),
        num_strikes=41,
        num_pos_qubits=4
    )
    
