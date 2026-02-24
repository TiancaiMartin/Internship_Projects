# Dynamic Random Forest (Rolling) Strategy

## Overview

This project implements a rolling-window Random Forest framework for dynamic signal generation and strategy allocation.

The goal is to build a machine learning–based signal that adapts over time using updated training data and produces out-of-sample trading decisions under realistic evaluation settings.

---

## Objective

- Construct predictive signals using structured financial features
- Apply rolling-window training to avoid look-ahead bias
- Generate dynamic allocation or trading signals
- Evaluate performance under strict out-of-sample testing

---

## Methodology

### 1. Feature Engineering
- Time-series based features
- Momentum / trend-based variables
- Custom signal transformations

### 2. Rolling Model Training
- Expanding or fixed rolling window
- Retrain model at each rebalance step
- Predict next-period signal

### 3. Signal to Strategy Mapping
- Convert prediction output into position sizing
- Threshold-based or ranking-based selection
- Portfolio construction logic

### 4. Backtesting Framework
- Out-of-sample evaluation only
- Rolling performance tracking
- Risk-adjusted metrics (Sharpe, drawdown, volatility)

---

## Key Strengths Demonstrated

- Machine learning applied to time-series finance
- Proper rolling evaluation methodology
- Feature-driven modeling pipeline
- Full signal → allocation → performance workflow

---

## Files

- `Dynamic_Random_Forest(Rolling).ipynb`  
  Main implementation notebook

---

## Notes

- Designed to simulate realistic predictive deployment.
- Emphasis on avoiding look-ahead bias and data leakage.
- Structured to allow feature replacement or model substitution.

---

## Disclaimer

For academic and research demonstration only.
