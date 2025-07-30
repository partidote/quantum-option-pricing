# 🧠 Quantum Option Pricing using Quantum Walks and Iterative Amplitude Estimation

This repository implements a quantum-inspired model for pricing European call options. It combines **Discrete-Time Quantum Walks (DTQW)** on a log-price lattice with **Iterative Amplitude Estimation (IAE)** from Qiskit. The model is benchmarked against the classical **Black–Scholes–Merton (BSM)** model and analyzed across strike, volatility, and interest rate conditions.

---

## 📌 Overview

The QOP model simulates log-price evolution via a quantum walk circuit, with directionality governed by a coin rotation angle derived from the **risk-neutral drift**. The expected payoff is encoded using Qiskit's `LinearAmplitudeFunction`, and IAE estimates its value with additive precision.

We observe both agreement and divergence with BSM pricing, and highlight changes in option prices with changes in strike prices and volatility.

---

## ⚙️ Key Features

- ✅ Quantum walk simulation of log-price diffusion  
- ✅ Parameterized coin angle from market drift  
- ✅ Exact piecewise-constant payoff encoding  
- ✅ Amplitude estimation for expected value computation  
- ✅ BSM comparison and volatility/strike sweeps  
- ✅ Real market data support via `yfinance`

  ---

- ## 👥 Contributors

- **Aaditeya Tripathi** — California State University, East Bay
- **Anany Pravin** — University of Illinois, Urbana-Champaign

