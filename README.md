# Market Regime Monitor

[![Update regime data](https://github.com/myreprise/market-regime-monitor/actions/workflows/update.yml/badge.svg)](https://github.com/myreprise/market-regime-monitor/actions/workflows/update.yml)

### ▶ Live dashboard: **https://myreprise.github.io/market-regime-monitor/**

A daily read on the U.S. equity market **regime**, computed two independent ways
and published as a static dashboard on GitHub Pages that refreshes itself every
weekday via GitHub Actions.

- **A transparent rule-based composite** — trend + volatility + chop, from
  classic technical statistics. Fully explainable, no model artifacts.
- **A statistical 4-state Hidden Markov Model** — an unsupervised regime model
  that classifies the market into *Risk-On*, *Risk-On-Retreat*, *Risk-Off*, and
  *Risk-Off-Stable* from price-derived features.

Showing both side by side is the point: when a hand-built rule engine and an
unsupervised statistical model **agree**, that's a stronger signal than either
alone — and when they *disagree*, that's information too.

> **Not investment advice.** This is a portfolio / research project. It describes
> the market's *state*, not what to buy or sell.

---

## The two engines

### 1. Rule-based composite (`regime_monitor/rule_based/`)

A deterministic classifier that combines three diagnostics into one label:

| Block | Statistic | Signal |
|---|---|---|
| **Trend** | EMA(21/50/200) stack + rolling log-price regression slope & R² | up / down / flat |
| **Volatility** | ATR% of price, Bollinger-vs-Keltner squeeze | low / normal / high / extreme |
| **Chop** | ADX (Wilder) + structure | trending / choppy / coil |

These combine into labels like `bull`, `bear`, `range`, `bull_high_vol`. Every
input is a well-known indicator computed in plain pandas — you can read exactly
why it says what it says.

### 2. Statistical 4-state HMM (`regime_monitor/hmm/`)

A Gaussian Hidden Markov Model trained on a standardized, PCA-whitened
feature vector derived entirely from **public price data**:

- SPY EMA(21/50/200) spreads, 126-day return
- VIX 252-day percentile and 10-day slope
- QQQ-vs-SPY rotation, credit spread (HYG−IEF), TLT return, copper-gold (CPER−GLD)
- Breadth: fraction of the 11 SPDR sector ETFs above their own EMA50
- Treasury-yield slope proxies (^IRX, ^TNX)

At inference the model reports a Bayesian posterior over the four regimes using
equal state priors and the learned Gaussian emissions (deliberately bypassing
the HMM start-probability, which otherwise biases single-day reads toward the
regime the training window opened in).

**Data sources:** all features come from `yfinance` — **no API keys, no paid
feeds, no index-constituent files.** "Breadth" here is sector-ETF based.

---

## Quick start

```bash
pip install -r requirements.txt

# Fetch fresh data, compute both engines, write the site JSON
python regime_pipeline.py

# Offline — use cached prices only
python regime_pipeline.py --cache-only

# More history for the timeline ribbon
python regime_pipeline.py --history-years 5
```

Outputs (consumed by the web front-end in Phase 2):

```
docs/data/latest.json     # today's read from both engines + data freshness
docs/data/history.json    # daily regime time series + SPY close (the ribbon)
```

---

## Architecture

```
regime_monitor/
├── data.py                 # yfinance fetch + on-disk parquet cache (offline-capable)
├── indicators/             # pure-pandas EMA, ATR, ADX, Bollinger, Keltner, regression
├── rule_based/             # trend + vol + chop composite detector (+ fast history)
└── hmm/
    ├── features.py         # the 18-feature contract, built from price frames
    └── model.py            # scaler → PCA whitener → per-state Gaussian posterior
models/                     # trained HMM artifacts (scaler, PCA, GaussianHMM, meta)
regime_pipeline.py          # entry point → docs/data/*.json
docs/                       # static GitHub Pages site (dependency-free HTML/CSS/SVG)
.github/workflows/          # scheduled job that refreshes the data
```

The whole thing runs on price data through a single command, which is what makes
it deployable as a free, self-refreshing GitHub Pages site: a scheduled GitHub
Action (`.github/workflows/update.yml`) reruns the pipeline each weekday evening,
commits the fresh `docs/data/*.json`, and GitHub Pages republishes automatically.

---

## Honest notes & limitations

- **The HMM model is a snapshot.** It was trained on ~2022–2026 daily data and is
  shipped as a fixed artifact (see `models/training_meta.json`). It is *not*
  continuously retrained; reads far outside the training window are
  extrapolation.
- **Faithful — not bit-identical — to the original.** This is a clean-room
  extraction of a regime engine from a larger private research codebase. The
  standalone pipeline recomputes features consistently with how the model was
  trained, which can differ slightly from the original production path.
- **Regime, not forecast.** The output describes the current *state* of the
  market. It is explicitly not a return predictor.
- **Confidence can saturate.** The single-observation Gaussian posterior can read
  ~100% on clearly-trending days; treat it as "which regime," with transition
  risk as the nuance.

---

## Tests

```bash
python -m pytest tests/ -q
```
