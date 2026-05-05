# trade-bot

A high-conviction, regime-aware algorithmic trading system designed to identify and execute **momentum continuation and breakout trades** with a focus on:

- Maximum return
- Convex upside capture
- High Sharpe optimization
- Strict risk management

This system integrates **data ingestion, feature engineering, machine learning, portfolio optimization, and execution** into a unified pipeline powered by the Alpaca API.

---

## 🧠 Core Philosophy

This system is **not** designed to trade frequently.

Instead, it:
- Trades **only the best opportunities**
- Focuses on **explosive upside setups**
- Concentrates capital into **top-ranked symbols**
- Uses **probability + expected value** for decision-making
- Minimizes drawdowns via strict execution controls

---

## 🏗️ System Architecture


Market Data (Alpaca)
↓
Feature Engineering
↓
ML Model (Continuation / Explosive Alpha)
↓
Ranking & Signal Generation
↓
Portfolio Optimization (Sharpe + Return)
↓
Execution Engine (Alpaca)


---

## 📦 Project Structure

### Core Components

- **Scanner**
  - :contentReference[oaicite:0]{index=0}
  - Builds dynamic universe (S&P 500 + NASDAQ)
  - Identifies breakout + momentum candidates

- **Feature Engineering**
  - :contentReference[oaicite:1]{index=1}
  - Generates indicators:
    - RSI, MACD, Bollinger Bands
    - SMA (5/20/50/200)
    - Volatility, ATR, volume metrics
    - Regime features (SPY / QQQ / VIXY)

- **Model**
  - :contentReference[oaicite:2]{index=2}
  - Supervised continuation model
  - Outputs:
    - Trade probabilities
    - Target weights
    - Regime-aware actions

- **Walk-Forward Simulation**
  - :contentReference[oaicite:3]{index=3}
  - Validates strategy using rolling windows
  - Focuses on:
    - Explosive returns
    - Convex risk-reward
    - Capital preservation

- **Portfolio Optimizer**
  - :contentReference[oaicite:4]{index=4}
  - Maximizes:
    - Expected return
    - Sharpe ratio
  - Penalizes:
    - Volatility
    - Reversal risk
    - Weak confidence

- **Execution Engine**
  - :contentReference[oaicite:5]{index=5}
  - Production-grade Alpaca executor
  - Features:
    - Sell-first logic
    - Cash buffer protection
    - Kill switch
    - Pre-trade sanity checks
    - Slippage tracking

- **Streaming Executor (Continuous Mode)**
  - :contentReference[oaicite:6]{index=6}
  - Runs continuously
  - Integrates:
    - Scanner
    - Execution
    - Portfolio updates

- **Orchestrator**
  - :contentReference[oaicite:7]{index=7}
  - End-to-end pipeline runner:
    - Data → Features → Model → Portfolio → Orders

- **Visualization**
  - :contentReference[oaicite:8]{index=8}
  - Generates:
    - Feature importance plots
    - Threshold optimization charts
    - Trade distributions

---

## ⚙️ Key Features

### 📊 Signal Generation
- Breakout detection (`close > previous high`)
- Momentum confirmation:
  - RSI
  - MACD crossovers
  - Stochastic RSI
- Volume & liquidity filtering

---

### 🤖 Machine Learning
- Ensemble models (Gradient Boosting + Random Forest)
- Probability-based trade selection
- Threshold optimization
- Regime-aware routing

---

### 🧠 Regime Awareness
Market states:
- **TREND_RISK_ON**
- **VOLATILE_RISK_ON**
- **CHOP_NEUTRAL**
- **DEFENSIVE_RISK_OFF**

Adjusts:
- Exposure
- Position sizing
- Trade thresholds

---

### 💰 Portfolio Optimization
- Max Sharpe + max return targeting
- Convex position sizing
- Kelly-style adjustments
- Risk penalties for:
  - Volatility
  - Bear conditions
  - Reversal signals

---

### ⚡ Execution Engine
Production safeguards:
- Market-hours guard
- Daily loss kill switch
- Cash buffer enforcement
- Max position sizing
- Spread / liquidity checks
- Slippage tracking

---

## 📁 Outputs

All outputs are stored in:


daily_system_outputs/


Key files:
- `daily_features.csv`
- `regime_predictions.csv`
- `regime_actions.csv`
- `portfolio_targets.csv`
- `rebalance_orders.csv`
- `exit_orders.csv`

---

## 🚀 Usage

### 1. Install Dependencies

```bash
pip install -r requirements.txt
2. Configure Environment

Create a .env file:

APCA_API_KEY_ID=your_key
APCA_API_SECRET_KEY=your_secret
3. Run Full Pipeline (Dry Run)
python run_daily_system_walkforward_emulated.py
4. Execute Trades (Paper)
python run_daily_system_walkforward_emulated.py --execute
5. Live Execution (Use Carefully)
python alpaca_executor.py --live
⚠️ Risk Management

The system enforces:

Max daily loss:
% threshold
absolute dollar threshold
Cash buffer requirements
Max position limits
Exposure caps by regime
Kill switch (auto stop trading)
📈 Strategy Characteristics
Property	Behavior
Trade Frequency	Low
Win Rate Target	~60%
Focus	High-conviction trades
Holding Period	Multi-day
Edge Source	Momentum + continuation
Risk Style	Convex / asymmetric
🧪 Research Notes
Uses strictly causal features
Avoids data leakage
Walk-forward validated
Designed for real execution constraints
🛑 Disclaimer

This system is for research and educational purposes only.

Trading involves substantial risk. Use paper trading before deploying real capital.