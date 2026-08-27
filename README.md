# Zerodha Kite + GPT AI Intraday Bot V3.8.1

This project scans NSE cash equities for intraday opening-range breakouts and
manages paper or Zerodha Kite orders through a hard risk layer. GPT is an
optional reviewer; it is not allowed to size positions or relax risk limits.

> **Live-money status:** live execution is guarded but inherently risky. Keep
> `LIVE_TRADING=false` until broker reconciliation/recovery tests pass in your
> deployment environment. No indicator, backtest, optimizer, or AI model can
> guarantee profit. Passing this test suite validates software invariants, not a
> durable trading edge or freedom from broker/network/exchange failures.

## Architecture

```text
NSE instrument dump (once at startup)
        ↓
KiteTicker WebSocket shards (QUOTE mode)
        ↓
broad in-memory stock-in-play ranking
        ↓
top preliminary pool → one full /quote request
(spread, depth, and circuit checks)
        ↓
top candidate pool → historical 5-minute candles
        ↓
causal ORB/VWAP/EMA/RSI/ATR/RVOL rules
        ↓
initial live revalidation + after-cost/risk checks
        ↓
AI_MODE=off | synchronous online shadow/gate review
        ↓
final live revalidation + trade rebuild + capacity reservation
        ↘ append-only AI_CANDIDATE record → offline ai_ideas.py
        ↓
post-account-preflight executable quote/freshness gate + final rebuild
        ↓
paper execution or guarded MIS order workflow
```

The bot shards the dynamic NSE EQ universe across up to three WebSocket
connections. The default is 2,800 instruments per socket, below Kite's
documented 3,000-instrument limit. All sockets update one thread-safe tick
store; no fixed watchlist is required.

## Safe setup on macOS

From this project directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Create `.env` from the template only if you do not already have one:

```bash
cp .env.example .env
chmod 600 .env
```

Set the Kite credentials. Add an OpenAI API key only when deliberately running
the offline idea worker or an online `shadow`/`gate` experiment. Keep these
recommended research-safety values unchanged:

```dotenv
AI_MODE=off
AI_IDEA_MODE=shadow
LIVE_TRADING=false
LIVE_TRADING_CONFIRM=
```

Never commit `.env`. The repository's VS Code settings enable
`python.terminal.useEnvFile`, so new integrated terminals load `.env`; restart
an existing terminal after changing the setting or environment file.

Run the offline unit suite before making any connection:

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```

## AI modes

The template defaults to `AI_MODE=off`. This keeps OpenAI out of the market
execution loop and preserves the cleanest deterministic paper baseline.

| Mode | Behavior | Intended use |
| --- | --- | --- |
| `off` | Makes no execution-loop OpenAI call and records the deterministic baseline as `OFF`. | Default paper measurement. |
| `shadow` | Calls OpenAI synchronously. The label is not a direct veto, but latency can cross the entry cutoff or change final revalidation. | Timing-sensitive paper experiment, not an execution-neutral baseline. |
| `gate` | Calls OpenAI synchronously and requires approval plus configured thresholds; errors fail closed. | Guarded paper/live use after execution-path validation. |

Online reviews are capped by `MAX_AI_REVIEWS_PER_SCAN`. Successful API reviews
record an input hash, decision ID, actual response model/ID, latency, and basic
token counts. Skipped, unavailable, or failed reviews record `ERROR` and a
status/reason, but provider response metadata may be blank and usage may be
zero. The completed-trade `ERROR` cohort is not a count of every API failure.

AI receives structured candidate features; it does not predict fills or know
the future. It cannot change quantity, stop distance, maximum position size,
daily loss limits, maximum open positions, or maximum trades. Prompt changes,
model changes, and model-version drift must be treated as strategy changes and
revalidated.

`OPENAI_MODEL=gpt-5.6` is an alias that currently routes to GPT-5.6 Sol; it is
not a promise that behavior will remain fixed. Use an explicitly pinned model
snapshot when one is available to the account, retain the actual response model
in provenance, and never combine changed model/prompt/config cohorts as if they
were one experiment. See the official
[GPT-5.6 Sol model page](https://developers.openai.com/api/docs/models/gpt-5.6-sol).

## Research-only AI trade ideas

After the synchronous mode decision, final live revalidation and final trade
rebuild, the bot attempts to append an `AI_CANDIDATE` for a cost/risk-qualified,
capacity-eligible proposal. Its `idea_id` identifies that final payload. This is
not a complete opportunity tape: entry cutoff, daily-trade, open-position, and
capacity checks truncate the stream. Generate research labels outside the
execution loop from an explicit same-session journal:

```bash
python ai_ideas.py logs/trades_YYYYMMDD.jsonl --dry-run
python ai_ideas.py logs/trades_YYYYMMDD.jsonl --limit 8
```

Replace `YYYYMMDD` with the session being reviewed. `AI_IDEA_MODE=shadow` marks
the intended research posture and prevents live startup, but it does not launch
the worker; running `ai_ideas.py` is the explicit API action and `--limit`
controls that invocation.

The dry run displays the identifier-stripped candidate payload without calling
OpenAI. It is not guaranteed anonymous and is not the complete API request. A
normal run writes owner-only `logs/ai_ideas.jsonl`. The worker removes the
symbol, token, signal timestamp, and composite scores from model input, then
constrains output to `TAKE`, `PASS`, or `ABSTAIN` labels for supplied candidate
IDs. It cannot place orders or change trade parameters or risk. These are
research labels only; do not connect them to execution unless a frozen,
out-of-sample portfolio replay first demonstrates repeatable after-cost value.

## Authentication and connection checks

Kite access tokens expire daily. Run:

```bash
python login.py
```

Complete Zerodha login in the browser and paste the redirect URL into the
terminal. `login.py` writes that day's `KITE_ACCESS_TOKEN` to `.env`.

Then run the connectivity check:

```bash
python test_connections.py
```

This contacts OpenAI and Zerodha, downloads instruments, and requests quote and
historical data. It places no orders. It is not an offline test and does not
validate strategy profitability.

## Paper operation

Confirm these recommended values before every baseline paper run:

```dotenv
AI_MODE=off
AI_IDEA_MODE=shadow
LIVE_TRADING=false
LIVE_TRADING_CONFIRM=
```

Start the bot:

```bash
python bot.py
```

Paper execution applies configured adverse slippage and estimated NSE equity
intraday charges. That is more realistic than using raw candle returns, but it
still cannot reproduce queue position, spread changes, gaps, partial fills,
rejections, broker latency, or every intrabar ordering ambiguity. Paper results
therefore remain estimates.

## Performance reporting

The reporter reads `logs/trades_*.jsonl` by default:

```bash
python performance_report.py
python performance_report.py --json
python performance_report.py 'logs/trades_*.jsonl' --json
```

It reports trade count, win rate, net and gross P&L, fees, expectancy, profit
factor, maximum drawdown, average R, and separate AI `APPROVE`, `REJECT`, and
`ERROR` cohorts. `OFF` trades count overall but never in an AI cohort. Results
are also split into execution-mode/config-fingerprint experiment cohorts; a
mixed overall or AI aggregate is labelled as non-comparable.

Only `CLOSE` records with explicit finite P&L, fees, and R-multiple fields are
used. AI attribution must be `APPROVE`, `REJECT`, `ERROR`, or `OFF`; a missing
decision is accepted only when `ai_mode=off`. Missing or legacy data is
diagnosed and excluded, and values are never reconstructed to make a run appear
profitable.
Negative fees and records where `net_pnl != gross_pnl - fees` are rejected.
Stable event IDs deduplicate a crash replay without double-counting a close.
Fee provenance states whether charges are model-estimated or broker-confirmed;
the bot's current close accounting is an estimate and the contract note remains
authoritative.
The command exits nonzero when there are no complete P&L records. This is an
accounting report, not a leakage-safe backtester. It reports provenance as
compatible, mixed, or unverifiable when execution mode, configuration
fingerprint, response model, or prompt version differ or are missing. The P&L
aggregate remains calculable for legacy records, but a warning does not make
mixed experiments comparable or causal.

## Point-in-time replay (Part 4)

`historical_replay.py` is an offline, deterministic validation layer. It uses
the same pure scalar setup-rule kernel as `bot.detect_setup`, then applies
chronological portfolio capacity, executable bid/ask entry references,
top-of-book participation, adverse slippage, the effective NSE intraday charge
model, an exit-side spread proxy of at least half the recorded entry spread,
the live entry-cutoff guard and circuit buffer, stop/target/circuit geometry,
and a forced 15:10 exit. If an unresolved five-minute bar can contain both stop
and target, the stop wins. If the entry-containing bar's adverse extreme crosses
the stop, it is booked as an explicitly counted worst-case assumption—not an
observed fill—while a favorable target is never granted from that bar. A gap
through a stop receives the worse bar-open price. It imports neither Kite nor
OpenAI and does not read `.env`.

If a realized daily-loss or consecutive-loss limit trips while another replay
position is still open, the session is rejected. Live trading would immediately
kill-switch and flatten that exposure, but bar OHLC has no executable quote for
the emergency liquidation; allowing it to reach a later target would be
optimistic.

The input is deliberately strict and checksummed:

```bash
python historical_replay.py research_data/dataset_name
cp replay_trials.example.json frozen_trials.json
# Give the copied registry a real ID/time and register every planned variant
# before examining the corresponding test/holdout outcomes.
python historical_replay.py research_data/dataset_name \
  --walk-forward --trials frozen_trials.json
```

Walk-forward selection uses only earlier training sessions, leaves a purge gap,
freezes the selected configuration for each test fold, and reserves the final
holdout unless `--evaluate-holdout` is explicitly supplied. Every tried
configuration belongs in the frozen registry; the selection score is the lower
one-sided 95% bound of mean trade R, subject to a minimum training sample.

The current `logs/` directory is intentionally rejected:

```text
INSUFFICIENT_POINT_IN_TIME_DATA: logs has no manifest.json; trade journals are not replay data
```

Those journals contain only selected decisions and ten completed trades. They
lack the historical universe, complete decision set, raw stock/NIFTY bars,
depth, and causal execution path, so using them to optimize thresholds would be
selection-biased. Replay output is always labelled
`DERIVED_LEDGER_BAR_ONLY_LOW_FIDELITY`; even a valid result cannot demonstrate
live fill behavior or guarantee profit. See [REPLAY_DATA_CONTRACT.md](REPLAY_DATA_CONTRACT.md)
for the recorder and dataset requirements.

## Live-order guard

There is intentionally no one-switch path from paper to live. Live startup
requires both:

```dotenv
LIVE_TRADING=true
LIVE_TRADING_CONFIRM=I_UNDERSTAND_REAL_MONEY
```

It also requires an explicit `AI_MODE=off` or `AI_MODE=gate`; `shadow` is
rejected. The process verifies its public IPv4 against `KITE_STATIC_IP` before
live broker operation. These are mistake-prevention controls, not evidence that
the strategy is safe, compliant, or profitable.

### Dedicated-account recovery (V3.5)

If this Zerodha account is used only by this bot for NSE/MIS intraday activity,
you may explicitly set:

```dotenv
DEDICATED_BOT_ACCOUNT=true
MAX_KILL_SWITCH_FLATTEN_ATTEMPTS=3
KILL_SWITCH_RETRY_SECONDS=5
```

Dedicated mode changes **recovery only**, not signal generation. When strict
local order metadata is missing or stale, the bot may use the broker's actual
NSE/MIS order book and signed position for the affected symbol, cancel every
active NSE/MIS order for that symbol, then submit exactly one market-protected
order for the residual signed position. The position is re-read before and after
mutations, and active orders are terminalized again after flat verification so
an orphan stop cannot reverse a flat account. Non-dedicated mode retains the
original fail-closed ownership checks.

V3.5 persists a dedicated-recovery intent before its first broker mutation and
updates it before submission and immediately after an order ID is known. An
unresolved intent survives a process restart and blocks all automatic startup
mutation until orders and positions are reconciled manually. Recovery will not
retry after an ambiguous order lifecycle, a terminal identity mismatch, stale
position settlement, or fills observed around cancellation; these cases can
leave exposure for manual handling, but cannot silently submit a second order.

V3.5 also treats a broker-issued persisted `order_id` as the primary identity of
an order after submission. A missing `tag` in an asynchronous/history payload no
longer makes the bot disown its own stop. A broker-reported `SL` representation
of the intended SL-M is accepted only as armed protection while it is actually
trigger-pending with exact side, quantity, trigger, position and broker order ID.
Terminal MARKET/LIMIT conversion shapes remain owned but are never mistaken for
armed protection. WebSocket/REST order updates share a monotonic cache so stale
payloads cannot regress a terminal order. Startup reconciles account exposure
before starting market-data subscriptions. Kill-switch flatten retries are
bounded; after the configured attempt count the state becomes
`manual_intervention_required=true` and no further automatic broker mutations are
issued.

### Emergency recovery utility

Stop the main bot before running the utility. On a dedicated account, if a live
position/order is left after a halt:

```bash
.venv/bin/python recover_account.py \
  --confirm FLATTEN_DEDICATED_ACCOUNT
```

The utility touches only NSE/MIS orders and positions. It cancels active NSE/MIS
orders per affected symbol, flattens only the verified residual signed position,
and verifies that no NSE/MIS position or active order remains. To back up an old
state file and create fresh state **only after broker-flat verification**:

```bash
.venv/bin/python recover_account.py \
  --confirm FLATTEN_DEDICATED_ACCOUNT \
  --reset-state-after-flat
```

Do not delete `bot_state.json`, reset the kill switch, or launch another bot
process while a broker position/order still exists.

These controls reduce known execution/reconciliation risks; they do not guarantee
that all broker, network, exchange, or market failures are recoverable, and they
do not establish strategy profitability. Start any live deployment with a
small supervised canary.

## Promotion checklist

Do not promote automatically because a backtest or a few paper sessions look
good. Record and review evidence for all of the following:

- Offline unit and failure-path tests pass from a clean environment.
- The universe and every feature are point-in-time; no future bars, revised
  constituents, end-of-day totals, or cross-validation leakage enter a signal.
- A cost-aware walk-forward test includes spread, all charges, adverse
  slippage, latency, partial/unfilled/rejected orders, and conservative
  same-bar stop/target ordering.
- Parameters are frozen before each out-of-sample window. An untouched final
  holdout is evaluated once, and all tried variants are recorded to expose
  selection bias.
- Results are stable across dates, volatility regimes, symbols, and sectors,
  with a broad parameter plateau rather than one optimized point.
- Shadow results show that AI adds repeatable **after-cost** value over the
  exact deterministic baseline. Otherwise use `AI_MODE=off`.
- Kite's no-real-money sandbox is used for what it supports, while a separate
  deterministic local simulator covers MARKET/SL-M behavior, margin paths,
  partial fills, ambiguous submissions, stop or exit rejection, reconnects,
  restarts, reconciliation, orphan cleanup, and end-of-day exit. The repository
  has not yet demonstrated that complete simulation coverage.
- Forward paper trading runs long enough to cover ordinary and stressed market
  conditions, with reconciliation against broker-like fills and charges.
- Deployment checks cover the registered static IP, current broker/exchange
  rules, secrets, clock synchronization, alerting, log retention, and a tested
  manual kill procedure.
- Loss limits and shutdown actions are accepted in advance, then any approved
  live trial starts at the smallest practical canary size under supervision.

Any material strategy, model, prompt, execution, cost, or risk change resets
the relevant evidence. Historical profit never guarantees future profit.

## Strategy and risk controls

The deterministic candidate logic uses:

- dynamic stock-in-play selection and liquidity/spread filters
- a fully closed, contiguous 15-minute opening range
- VWAP and EMA 9/20 trend confirmation
- RSI, ATR, same-clock breakout RVOL, and prior-session 15-minute opening RVOL
- the first directional candle close beyond the range, candle quality, and NIFTY regime
- an ATR stop and configurable R-multiple target
- exchange-timestamp quote freshness, immutable signal-price drift, and live
  spread/VWAP/breakout/circuit revalidation
- no new entries after 14:30 and forced exit at 15:10

The candle request, incomplete-bar filter, and continuity validator share one
captured timestamp. A bar is eligible only after its full five-minute interval
plus `CANDLE_CLOSE_GRACE_SECONDS`; missing, duplicate, misaligned, malformed, or
unexpected current-session bars make the scan fail closed. Historical request
bounds are normalized to IST and serialized explicitly as
`YYYY-MM-DD HH:MM:SS`; this avoids host-timezone differences and the Kite SDK's
incompatible serialization of pandas timestamps. Immediately before entry
intent, LONG uses the best ask and SHORT uses the best bid. Exchange quote age
must remain within `MAX_EXECUTION_QUOTE_AGE_SECONDS`, and every drift check is
measured from the immutable signal-candle close rather than the prior refresh.
These controls remove known look-ahead and stale-entry paths; they do not prove
that the strategy has positive expectancy.

A historical-data failure aborts the entire entry scan rather than repeating
the same failed request for every candidate. Repeated scan-level failures use a
capped exponential schedule up to `FULL_SCAN_ERROR_BACKOFF_MAX_SECONDS`; a
successful scan resets the delay. Position monitoring, P&L enforcement, and
forced-exit handling continue at their independent cadence during this backoff.

Default ceilings for `CAPITAL_LIMIT=100000` are approximately ₹200 planned risk
per stopped trade, ₹25,000 maximum position notional, ₹400 aggregate open-stop
risk, ₹50,000 aggregate gross exposure, two open positions, five trades per
day, and an ₹800 daily kill level. Entries use the remaining daily-loss,
open-stop-risk, and gross-exposure budgets; profits do not expand those limits.
`MIN_AFTER_COST_PAYOFF_RATIO=1.20` rejects modeled net targets that are too small
relative to modeled stopped losses. Stops and targets must remain inside the
directionally relevant circuit with headroom. These are configurable loss
ceilings, not expected returns. Realized loss can exceed a planned stop because
of gaps, slippage, rejection, or market dislocation.

## Runtime state

```text
data/bot_state.json   persistent strategy/order state
data/bot.lock         single-process lock
logs/trades_YYYYMMDD.jsonl
logs/ai_ideas.jsonl         offline structured AI research output
```

State writes are atomic and the lock prevents two local bot processes from
running concurrently. Dedicated recovery intents are stored in
`bot_state.json`; an unresolved one deliberately prevents restart. Do not delete
or manually clear that state while a position or broker order may exist. After a
crash or uncertain order response, reconcile broker orders and positions before
resuming; never assume a timeout means no order was accepted. Realized `CLOSE`
events are committed to a durable state outbox with the trade accounting and
replayed after a journal failure or restart. Replays retain the same event ID.
Repeated identical halt causes are suppressed; distinct follow-on causes are
retained in `halt_details` without replacing the root halt reason.

## Dynamic universe and broker constraints

The bot downloads the current NSE instrument list at startup and includes
normal NSE cash equities while excluding identifiable special or
compulsory-delivery series. Zerodha can still apply changing RMS and product
restrictions; a rejected entry must remain no position.

The bot intentionally limits its preliminary pool to 250 instruments so one
full-quote request remains comfortably bounded; 250 is this bot's safety cap,
not a claim about Kite's endpoint maximum. The design also assumes up to 3,000
instruments per WebSocket connection, quote REST at 1 request/second,
historical candles at 3 requests/second, and API orders at 10 requests/second.
It requests automatic market protection for MARKET and SL-M orders. Broker and
exchange rules can change, so verify the current official documentation before
every readiness review. Realtime WebSocket and historical candle access require
the appropriate paid Kite Connect plan.

The Closing Auction Session (CAS) for eligible equity-cash securities took
effect on 2026-08-03. This bot's configured 15:10 forced exit precedes CAS, but
CAS changes closing-price and end-of-session data semantics. Historical data,
recorders, manifests, and replays must version the applicable exchange-session
regime and must not mix pre-CAS and post-CAS assumptions silently. See the
[SEBI CAS circular](https://www.sebi.gov.in/legal/circulars/jan-2026/introduction-of-closing-auction-session-cas-in-the-equity-cash-segment-and-certain-modifications-in-the-pre-open-auction-session_99122.html)
and the
[NSE Clearing effective-date circular](https://nsearchives.nseindia.com/content/circulars/CMPT74898.pdf).

See [RESEARCH_NOTES.md](RESEARCH_NOTES.md) for the architecture's design
rationale.
