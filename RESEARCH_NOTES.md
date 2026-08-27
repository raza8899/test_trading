# Research and validation notes

Audit date: 2026-08-26.

> This bot is not demonstrated to be profitable, production-ready, or legally
> compliant for a particular account. Keep `LIVE_TRADING=false`. Software
> safeguards reduce some failure modes; they do not prevent market loss,
> slippage, gaps, broker outages, model error, or regulatory change. No
> backtest, metric, or AI system can guarantee future gains.

## Design rationale

### Causal market-data path

The broad scan uses current WebSocket ticks, then one full-quote request for
depth/circuit checks, and historical candles only for the shortlist. One
captured cutoff now governs the historical request, closed-bar filter, and
strict current-session continuity check. The setup engine fails closed on a
missing, duplicate, misaligned, malformed, partial, or stale current-session
bar. It warms EMA/RSI/ATR over earlier sessions, resets VWAP each session, and
compares both breakout-bar volume at the same clock time and the complete
15-minute opening volume with earlier sessions.

The setup must be the first directional candle close beyond the opening range.
Online `shadow`/`gate` review is synchronous. Revalidation uses ask for LONG and
bid for SHORT, preserves the original signal close as the immutable drift
baseline, checks the exchange quote timestamp after broker I/O, and reapplies
spread, circuit, breakout, and VWAP-extension gates. A final post-account-
preflight revalidation and a timestamp-only check after durable intent prevent
stale broker mutation.

These are sensible live-data causality controls, but they are not a backtest.
They have not established that ORB/VWAP/EMA/RSI/RVOL has positive expectancy.

### Part 3 evidence and limits

The strongest directly relevant evidence found was a recent opening-range
working paper: its unfiltered strategy did not establish an edge, while its
particular completed five-minute opening-candle direction rule and same-window
relative-volume selection materially improved the reported U.S. sample. This
bot retains a different 15-minute range and requires a directional breakout
candle, so the paper supports causal range/volume construction rather than this
exact rule or threshold. It has limited external validity for NSE single stocks
and does not provide an untouched final holdout:
[Swiss Finance Institute Research Paper 24-98](https://www.alexandria.unisg.ch/server/api/core/bitstreams/3c2989c4-688d-4d78-8a71-f02690990d51/content).

Kite documents historical candles as timestamped OHLCV records and live quotes
as exchange snapshots with depth and exchange timestamps. Therefore historical
bars are used only after a causal close/grace boundary, while submission uses
the currently executable side of a fresh depth snapshot:
[Kite historical data](https://kite.trade/docs/connect/v3/historical/),
[Kite market quotes](https://kite.trade/docs/connect/v3/market-quotes/), and
[Kite WebSocket fields](https://kite.trade/docs/connect/v3/websocket/).

VWAP alignment and volatility-normalized chase limits remain plausible filters,
not independently proven profit sources. Execution cost is mandatory because
NSE defines impact cost as quantity- and order-book-dependent rather than a
fixed constant: [NSE impact-cost methodology](https://www.nseindia.com/static/products-services/indices-impact-cost).
No reviewed source establishes universal NSE thresholds for RVOL, ATR distance,
VWAP extension, RSI, or opening-range length. Those parameters still require a
registered, cost-aware chronological walk-forward evaluation and one untouched
holdout; tuning them against the small live journal would be overfitting.

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

- `off` avoids execution-loop OpenAI calls and records `OFF`; it is the
  recommended baseline.
- `shadow` does not use the label as a direct veto, but its synchronous latency
  can cross the cutoff or change final revalidation. It is not execution-neutral.
- `gate` requires approval/thresholds and fail-closes on a separate AI error;
  it remains paper-research only.
- AI never changes size, stops, or hard daily/position limits.

Both the online reviewer and separate offline worker remove direct identifiers,
provenance timestamps, and composite scores from model input. Their payloads
are only identifier-stripped, not guaranteed anonymous. Neither path has
privileged future, news, order-book, or fill information. AI confidence is not
a probability of profit.

## Remaining validation and measurement blockers

### 1. The replay engine exists, but no valid historical dataset exists

Part 4 adds a strict opportunity/portfolio replay and chronological
walk-forward evaluator. The live detector and replay share one pure scalar rule
kernel, so recorded feature rows use the same setup thresholds and score. The
engine uses bid/ask references, visible-depth participation, adverse slippage,
all modeled charges, portfolio capacity, the live entry-cutoff/circuit gates,
gap-aware stops, explicitly counted worst-case entry-bar assumptions,
stop-first ambiguous later bars, and an explicit final-holdout boundary. It
rejects missing, modified, duplicate, non-causal, or incomplete inputs and never
imports the broker or AI client.

This is replay infrastructure, not profitability evidence. Its supported OHLC
execution policy is explicitly low fidelity, and the repository has no dataset
meeting [the point-in-time contract](REPLAY_DATA_CONTRACT.md). A full-parity
recording still needs:

- point-in-time instrument/series eligibility and corporate-action handling;
- only information available by each simulated decision timestamp, with a bar
  finalization delay and an exchange calendar;
- the entire cross-sectional ranking and portfolio-capacity competition;
- bid/ask spread, impact, latency, partial/unfilled/rejected orders, all current
  charges, kill-switch liquidation quotes, and conservative ambiguous intrabar
  sequencing;
- raw data lineage behind the derived feature ledger and deterministic
  cross-sectional tie-breaking.

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

V3.8 journals a normalized run manifest and hashes `bot.py`,
`strategy_rules.py`, `trading_core.py`, the pinned requirements file, and
material configuration. After final
revalidation and trade rebuild it attempts to journal the final eligible
candidate with a stable `idea_id`. That stream is cutoff- and capacity-truncated
and is not the complete rejected-opportunity universe. Point-in-time
universe/data versions, installed-runtime provenance, corporate actions, and an
explicit exchange-session-rules version are still missing.

The performance reporter keeps legacy P&L calculable but warns when execution
mode or configuration fingerprints are mixed/missing, and when AI response
model or prompt provenance is mixed/missing. Such output is incompatible or
unverifiable evidence; the warning does not repair the aggregation.

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

V3.2 separates local/API failure as `ERROR`. A successful online review records
its input hash, decision ID, actual response model/ID, latency, basic token
counts, and setup snapshot. Skipped, unavailable, or failed reviews retain a
status/reason but may have blank provider metadata and zero usage. The report's
`ERROR` cohort contains completed trades carrying that label; it is not a count
of every failed review. Journal fields are provenance aids, not immutable proof.
Cross-run causal reporting still needs experiment-isolated joins between
`AI_CANDIDATE`, `AI_REVIEW`, and portfolio outcomes.

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
The configured `gpt-5.6` name is an alias that currently routes to GPT-5.6 Sol,
not a fixed behavioral snapshot. Pin an explicit snapshot when available,
retain the actual response model, and treat alias drift or every model, prompt,
schema, or threshold change as a new trial. See the official
[GPT-5.6 Sol model page](https://developers.openai.com/api/docs/models/gpt-5.6-sol).

### 4. Risk and operational simulations remain incomplete

- Paper daily-loss checking currently uses realized P&L, not portfolio
  mark-to-market P&L; its kill-switch behavior is therefore not a full live-risk
  simulation.
- V3.2 reserves aggregate open-stop risk and gross exposure before entry, but
  sector, correlation, beta, net-directional and aggregate gap-risk controls
  remain absent.
- The configured WebSocket age defaults to 30 seconds and validation rejects
  values over 60. Full/execution quotes now retain and validate the exchange
  timestamp, but the full-quote response still does not refresh every ranking
  field.
- NIFTY and setup data now require a complete, contiguous current session
  through the causal cutoff.
- Live monitoring now checks exact stop identity and pending quantity against
  broker position quantity; account-wide fencing across another machine or
  clone remains unresolved.

These are live-money blockers, not optimization opportunities.

## Broker, sandbox, and regulatory boundaries

Kite documents an official
[no-real-money sandbox](https://kite.trade/docs/connect/v3/sandbox/). It supports
LIMIT orders but omits MARKET, GTT, and margin-calculation behavior, so it cannot
cover the complete production path. Retain a deterministic local simulator with
injected timeouts, duplicated/out-of-order postbacks, partial fills, rejections,
reconnects, restarts, and orphan orders; neither environment guarantees live OMS
or exchange behavior.

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

SEBI introduced a phased Closing Auction Session (CAS) for eligible equity-cash
securities, and NSE Clearing states an effective date of 2026-08-03. The bot's
15:10 forced exit is before CAS, but CAS changes closing-price construction and
end-of-session data semantics. A recorder or replay must explicitly version the
exchange-session regime and keep pre-CAS and post-CAS assumptions separate. See
the [SEBI CAS circular](https://www.sebi.gov.in/legal/circulars/jan-2026/introduction-of-closing-auction-session-cas-in-the-equity-cash-segment-and-certain-modifications-in-the-pre-open-auction-session_99122.html)
and
[NSE Clearing circular NCL/CMPT/74898](https://nsearchives.nseindia.com/content/circulars/CMPT74898.pdf).

## Minimum evidence before any live canary

1. Complete the point-in-time recorder and populate the cost-aware replay with
   contract-valid forward data; the engine alone is not evidence.
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
7. Require a separate legal, broker, operational, and human approval before even
   considering the smallest supervised canary. This audit grants no live-trading
   approval, and promotion must never be automatic.

Even after all seven steps, losses remain possible and profit is not assured.
