"""Download SPY/QQQ/TLT/GLD daily data and produce proposal summary stats + figures.

Usage: python scripts/eda.py   (writes data/ and figs/ under the repo root)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
DATA, FIGS = ROOT / "data", ROOT / "figs"
DATA.mkdir(exist_ok=True)
FIGS.mkdir(exist_ok=True)

TICKERS = ["SPY", "QQQ", "TLT", "GLD"]
START, END = "2010-01-01", "2026-01-01"
SPLITS = {"train": ("2010", "2019"), "val": ("2020", "2021"), "test": ("2022", "2025")}
L = 40  # lookback window

raw = yf.download(TICKERS, start=START, end=END, auto_adjust=True, progress=False, threads=False)
close, volume = raw["Close"][TICKERS], raw["Volume"][TICKERS]
pd.concat({"close": close, "volume": volume}, axis=1).to_csv(DATA / "etf_daily.csv")

r = np.log(close).diff().dropna()
missing = close.isna().sum()

# Stress indicator: SPY 20-day realized vol (annualized) above the 80th percentile of the TRAIN period.
vol20 = r["SPY"].rolling(20).std() * np.sqrt(252)
thr = vol20.loc[SPLITS["train"][0]:SPLITS["train"][1]].quantile(0.8)
stress = vol20 > thr

lines = [f"Data: Yahoo Finance via yfinance, adjusted close, {r.index[0].date()} to {r.index[-1].date()}",
         f"Trading days with returns: {len(r)}; missing closes per ticker: {missing.to_dict()}",
         f"Stress threshold (SPY 20d vol, train 80th pct): {thr:.3f} annualized", ""]
lines.append("split  years      days  samples(L=40)  stress_days  stress_frac")
for name, (a, b) in SPLITS.items():
    n = len(r.loc[a:b])
    s = int(stress.loc[a:b].sum())
    lines.append(f"{name:5s}  {a}-{b}  {n:5d}  {n - L + 1:12d}  {s:11d}  {s / n:10.2%}")

desc = pd.DataFrame({
    "mean_daily_%": r.mean() * 100,
    "std_daily_%": r.std() * 100,
    "ann_vol_%": r.std() * np.sqrt(252) * 100,
    "skew": r.skew(),
    "excess_kurt": r.kurt(),
    "min_%": r.min() * 100,
    "max_%": r.max() * 100,
    "lag1_autocorr": [r[c].autocorr(1) for c in TICKERS],
    "lag1_autocorr_|r|": [r[c].abs().autocorr(1) for c in TICKERS],
}).round(3)
lines += ["", "Daily log-return statistics (2010-2025):", desc.to_string()]

calm_corr = r[~stress.reindex(r.index, fill_value=False)].corr().round(2)
stress_corr = r[stress.reindex(r.index, fill_value=False)].corr().round(2)
lines += ["", "Correlation, calm days:", calm_corr.to_string(),
          "", "Correlation, stress days:", stress_corr.to_string()]
(DATA / "summary_stats.txt").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
COLORS = {"SPY": "#1f77b4", "QQQ": "#ff7f0e", "TLT": "#2ca02c", "GLD": "#d62728"}
SPLIT_SHADE = {"val": "#dddddd", "test": "#bbbbbb"}

# Fig 1: normalized prices with split shading
fig, ax = plt.subplots(figsize=(6, 2.6))
for c in TICKERS:
    ax.plot(close.index, close[c] / close[c].dropna().iloc[0], lw=1, label=c, color=COLORS[c])
for name, (a, b) in SPLITS.items():
    if name in SPLIT_SHADE:
        ax.axvspan(pd.Timestamp(f"{a}-01-01"), pd.Timestamp(f"{b}-12-31"), color=SPLIT_SHADE[name], alpha=0.5, lw=0)
        ax.text(pd.Timestamp(f"{a}-03-01"), ax.get_ylim()[1] * 0.9, name, fontsize=8)
ax.set_yscale("log")
ax.set_ylabel("Growth of $1 (log)")
ax.legend(ncol=4, frameon=False, loc="upper left", bbox_to_anchor=(0, 1.15))
fig.tight_layout()
fig.savefig(FIGS / "fig1_prices_splits.png", dpi=200)

# Fig 2: SPY realized vol with stress threshold
fig, ax = plt.subplots(figsize=(6, 2.3))
ax.plot(vol20.index, vol20, lw=0.8, color="#1f77b4")
ax.axhline(thr, ls="--", color="#d62728", lw=1, label=f"stress threshold ({thr:.2f})")
ax.fill_between(vol20.index, 0, vol20.max(), where=stress, color="#d62728", alpha=0.15, lw=0)
ax.set_ylabel("SPY 20d realized vol\n(annualized)")
ax.legend(frameon=False, loc="upper left")
fig.tight_layout()
fig.savefig(FIGS / "fig2_spy_vol_stress.png", dpi=200)

# Fig 3: correlation calm vs stress
fig, axes = plt.subplots(1, 2, figsize=(6, 2.6))
for ax, m, title in zip(axes, [calm_corr, stress_corr], ["Calm days", "Stress days"]):
    im = ax.imshow(m.values, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(4), TICKERS)
    ax.set_yticks(range(4), TICKERS)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{m.values[i, j]:.2f}", ha="center", va="center", fontsize=7)
    ax.set_title(title)
fig.colorbar(im, ax=axes, shrink=0.8)
fig.savefig(FIGS / "fig3_corr_calm_vs_stress.png", dpi=200, bbox_inches="tight")

# Fig 4: volatility clustering -- ACF of r vs |r| for SPY
lags = range(1, 31)
fig, ax = plt.subplots(figsize=(6, 2.3))
ax.bar([l - 0.2 for l in lags], [r["SPY"].autocorr(l) for l in lags], width=0.4, label="returns $r_t$")
ax.bar([l + 0.2 for l in lags], [r["SPY"].abs().autocorr(l) for l in lags], width=0.4, label="|returns| $|r_t|$")
ax.axhline(0, color="k", lw=0.5)
ax.set_xlabel("lag (trading days)")
ax.set_ylabel("SPY autocorrelation")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(FIGS / "fig4_spy_acf.png", dpi=200)
print("figures written to", FIGS)
