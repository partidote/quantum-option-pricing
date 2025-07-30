"""
Add this code block to your driver to generate plots of QOP vs BSM prices when span_sigma = 2 and span_sigma = 3
"""

# volatility sweep with default span_sigma = 3
    sweep_volatility_vs_price(
        ticker= my_ticker,
        expiry_days=my_days,
        sigma_range=(0.1, 1.2),
        num_sigma=12
    )

    # volatility sweep with span_sigma = 2
    sweep_volatility_vs_price(
        ticker= my_ticker,
        expiry_days=my_days,
        sigma_range=(0.1, 1.2),
        num_sigma=12,
        span_sigma=2
    )
