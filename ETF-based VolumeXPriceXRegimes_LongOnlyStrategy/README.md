# ETF-Based Volume × Price Regime Long-Only Strategy

## Overview

This project implements a regime-aware long-only ETF strategy based on the interaction between price dynamics and volume signals.

The core idea is to detect structural market regimes using price–volume behavior and adjust exposure accordingly within a long-only framework.

---

## Research Motivation

Volume often contains additional information beyond price trends, particularly in regime transitions.  
This project investigates whether combining:

- price momentum / trend signals
- volume-based confirmation
- regime classification logic

can improve long-only allocation timing for ETFs.

---

## Data

Primary inputs include:

- ETF historical price data
- Volume data
- IVIX (volatility proxy)
- Market index reference data

Files:
- `ETF-VolumeXPrice.ipynb` – main research notebook
- `etf_pool_data.csv / parquet / xls` – ETF universe data
- `IVIX_SH.csv` – volatility index
- `WIND_ALLA_881001.csv` – market benchmark reference

---

## Methodology

### 1. Feature Construction

- Price momentum / trend features
- Volume expansion / contraction signals
- Price-volume interaction metrics

### 2. Regime Detection

- Classification of market states
- Signal smoothing to reduce noise
- Conditional exposure adjustment

### 3. Long-Only Allocation Logic

- Exposure scaling based on regime
- Position filtering under adverse regimes
- Rebalancing at predefined frequency

### 4. Backtesting Framework

- Rolling evaluation
- Strategy return calculation
- Risk-adjusted performance metrics
- Benchmark comparison

---

## Key Strengths Demonstrated

- Regime-aware signal design
- Integration of volume information
- Long-only strategy discipline
- Structured evaluation pipeline

---

## Expected Reviewer Focus

When reviewing this project, focus on:

- how regimes are defined
- how signals translate into allocation
- robustness under different market conditions
- comparison versus static long-only benchmark

---

## Notes

- Designed as a research prototype.
- The structure allows extension to multi-asset or risk-parity frameworks.
- Emphasis is placed on logical consistency rather than curve-fitting.

---

## Disclaimer

For academic and research demonstration purposes only.
Does not constitute investment advice.
