# Research and validation notes

Audit date: 2026-08-11.

> This bot is not demonstrated to be profitable, production-ready, or legally
> compliant for a particular account. Keep `LIVE_TRADING=false`. Software
> safeguards reduce some failure modes; they do not prevent market loss,
> slippage, gaps, broker outages, model error, or regulatory change. No
> backtest, metric, or AI system can guarantee future gains.

## Design rationale

### Causal market-data path

The broad scan uses current WebSocket ticks, then one full-quote request for
depth/circuit checks, and historical candles only for the shortlist. The setup
engine drops the forming five-minute candle, warms EMA/RSI/ATR over earlier
sessions, resets VWAP each session, and compares volume with earlier sessions'
same clock-time bar. The signal is rechecked for age, current price, and
breakout distance after a possibly slow AI review.

These are sensible live-data causality controls, but they are not a backtest.
They have not established that ORB/VWAP/EMA/RSI/RVOL has positive expectancy.

### Broker state is authoritative

Kite states that receiving an `order_id` does not establish exchange receipt or
execution; the order history/current status must be checked. The bot therefore
persists intent before mutation, gives orders unique tags, consumes WebSocket
updates, reconciles ambiguous submissions, confirms terminal order state, and
requires a flat broker position before recording a close. This follows the
[Kite order lifecycle](https://kite.trade/docs/connect/v3/orders/) and
[Kite postback guidance](https://kite.trade/docs/connect/v3/postbacks/), which
also describes partial-fill `UPDATE` events and recommends WebSocket postbacks
for individual developers.

Automatic market protection is useful but not a fill guarantee. Kite documents
that it converts MARKET/SL-M orders to protected limit orders and that an order
may still be rejected by exchange price-protection rules.

### AI is bounded and optional

- `off` avoids the OpenAI call and records `OFF`.
- `shadow` records a review but does not block deterministic paper execution.
- `gate` requires approval/thresholds and treats an AI failure as rejection.
- AI never changes size, stops, or hard daily/position limits.

The AI sees only supplied numerical features. It has no privileged future,
news, order-book, or fill information. Its confidence is not a probability of
profit.

## Remaining validation and measurement blockers

### 1. There is no historical replay or walk-forward engine

`performance_report.py` accounts for completed journal trades; it does not
recreate the cross-sectional scan, order path, or rejected opportunities. A
valid replay still needs:

- point-in-time instrument/series eligibility and corporate-action handling;
- only information available by each simulated decision timestamp, with a bar
  finalization delay and an exchange calendar;
- the entire cross-sectional ranking and portfolio-capacity competition;
- bid/ask spread, impact, latency, partial/unfilled/rejected orders, all current
  charges, and conservative ambiguous intrabar sequencing;
- expanding or rolling walk-forward selection, frozen out-of-sample windows,
  and one untouched final holdout.

Downloading today's instrument master is causal for today's live scan, but
reusing it for historical tests would introduce survivorship/eligibility bias.
Archive the daily instrument master and raw decision-time inputs before
building a backtest.

Every tried rule, threshold, prompt, and model counts toward selection. The
[Probability of Backtest Overfitting](https://scholarworks.wmich.edu/math_pubs/42/)
paper explains why an ordinary holdout can be unreliable after investment
strategy search. The
[Deflated Sharpe Ratio paper](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf)
adjusts for multiple selection and non-normal returns. Walk-forward evidence,
PBO/DSR, confidence intervals, and parameter stability are complementary; none
rescues a leaky or unrealistic simulation.

### 2. Journals are insufficient for reproducible profit attribution

The journal does not snapshot a complete run manifest containing the code
revision, all strategy/risk/cost parameters, universe version, data timestamp,
and prompt/schema hash. Aggregating logs can therefore mix incompatible
strategies and cost regimes.

The report's drawdown is realized close-to-close rupee drawdown. It excludes
intratrade/unrealized drawdown, overlapping capital, exposure, deposits, and
withdrawals. Profit factor and expectancy need sample uncertainty, regime and
symbol concentration, a benchmark, and an independently frozen evaluation
period before they support any conclusion.

Paper P&L uses a fixed adverse slippage setting and estimated charges. It does
not model queue priority or continuous paths, and the default five-second LTP
poll can miss excursions between observations. Multiple exit orders can make a
two-turnover fee estimate differ from the contract note. Reconcile estimates
against broker trade records and the current
[Zerodha charges schedule](https://zerodha.com/charges/); contract notes remain
authoritative.

### 3. Current AI cohorts are not clean causal evidence

In shadow mode, an unavailable API or model error becomes the same `REJECT`
decision stored on `CLOSE` as a genuine model rejection. The report also does
not retain `ai_mode`, prompt version, response model, response ID, latency, or
error on each close, so APPROVE/REJECT cohorts can mix shadow and gate runs,
models, prompts, and service failures.

Shadow execution is not an exact gate-policy counterfactual: the AI call delays
entry/revalidation, and executing shadow-rejected trades consumes daily-trade
and open-position capacity that gate mode would have left available. Cohort
means alone therefore cannot prove incremental AI value.

Before using `gate`, record the full candidate input and attribution, separate
`REJECT` from `AI_ERROR`/`UNAVAILABLE`, freeze the portfolio replay policy, and
compare after-cost out-of-sample portfolio returns with the exact deterministic
baseline. Also evaluate behavioral invariants such as schema compliance,
refusal to invent facts, and stable rejection of inconsistent inputs.

Official OpenAI guidance recommends task-specific, production-representative
evals, logging, held-out data, human calibration, and continuous evaluation;
it explicitly warns against “vibe-based” evaluation. See
[OpenAI evaluation best practices](https://developers.openai.com/api/docs/guides/evaluation-best-practices).
OpenAI also notes that model behavior can change between snapshots and
recommends pinned model versions plus application evals; see
[API backwards compatibility](https://developers.openai.com/api/reference/overview#backwards-compatibility).
Treat every model, prompt, schema, or threshold change as a new strategy trial.

### 4. Risk and operational simulations remain incomplete

- Paper daily-loss checking currently uses realized P&L, not portfolio
  mark-to-market P&L; its kill-switch behavior is therefore not a full live-risk
  simulation.
- Maximum positions limits trade count, but there is no sector, correlation,
  beta, gross/net exposure, or aggregate gap-risk control.
- A connected WebSocket tick can be accepted up to the configured 180-second
  age, while the full-quote response refreshes circuit calculations but not all
  ranking fields. Record and enforce tighter feature freshness.
- NIFTY regime selects the latest returned session without explicitly requiring
  today's date. A transient data gap can silently use the previous session.
- During a live session, an external partial position change is not continuously
  checked against the protective stop's pending quantity. An oversized stop can
  create reversal risk; reconciliation on restart is too late.

These are live-money blockers, not optimization opportunities.

## Broker, sandbox, and regulatory boundaries

Zerodha staff currently state that Kite Connect has
[no sandbox environment](https://kite.trade/forum/discussion/15871/how-to-test-applications-before-putting-in-real-money).
Recheck this before every readiness review because availability can change. A
local simulator with injected timeouts, duplicated/out-of-order postbacks,
partial fills, rejections, reconnects, restarts, and orphan orders is required,
but it cannot certify real OMS/exchange behavior. Never treat intentionally
rejected live orders or an underfunded account as a safe sandbox.

The NSE's
[May 2025 retail-algo implementation standards](https://nsearchives.nseindia.com/content/circulars/INVG67858.pdf)
require mapped static IP access for client-generated algos and daily API-session
logout. The later
[NSE retail-algo FAQ](https://nsearchives.nseindia.com/web/sites/default/files/inline-files/FAQ_Retail%20Algo_03112025_NSE.pdf)
describes the static-IP obligation for a self-hosted “Tech Savvy” API client and
algo-order tagging. This project appears to fall in that self-hosted category,
but only Zerodha/NSE can confirm account-specific implementation and current
rules. Matching the configured IP through a public-IP service does not prove
broker whitelisting or compliance.

## Minimum evidence before any live canary

1. Build the point-in-time recorder and cost-aware portfolio replay first.
2. Register all research trials; freeze parameters before each walk-forward
   window and evaluate one untouched final holdout once.
3. Demonstrate stable after-cost results across regimes, symbols, sectors, and
   nearby parameter values, with uncertainty, PBO/DSR, and concentration shown.
4. Show that AI adds repeatable after-cost value over the exact baseline on a
   frozen eval set; otherwise keep `AI_MODE=off`.
5. Pass deterministic fault-injection and restart reconciliation tests, resolve
   the blockers above, and complete sustained forward paper observation.
6. Reconfirm current broker/exchange rules, static-IP mapping, charges, alerting,
   manual kill procedures, and independent broker-state reconciliation.
7. Require explicit human approval and the smallest practical supervised
   canary. Promotion must never be automatic.

Even after all seven steps, losses remain possible and profit is not assured.
