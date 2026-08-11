# Zerodha Kite + GPT AI Intraday Bot V3.1

This project scans NSE cash equities for intraday opening-range breakouts and
manages paper or Zerodha Kite orders through a hard risk layer. GPT is an
optional reviewer; it is not allowed to size positions or relax risk limits.

> **Live-money status:** keep `LIVE_TRADING=false`. No indicator, backtest,
> optimizer, or AI model can guarantee profit. Passing the test suite and
> completing a paper session demonstrate software behavior, not a durable
> trading edge or production readiness.

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
AI_MODE=off | shadow | gate
        ↓
hard position and daily risk limits
        ↓
paper execution or guarded MIS order workflow
```

V3.1 shards the dynamic NSE EQ universe across up to three WebSocket
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
```

Set the Kite credentials and, for `shadow` or `gate`, an OpenAI API key. Keep
these safety values unchanged:

```dotenv
AI_MODE=shadow
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

`AI_MODE=shadow` is the research default and is deliberately incompatible with
live trading.

| Mode | Behavior | Intended use |
| --- | --- | --- |
| `off` | Does not call OpenAI; records the deterministic baseline as `OFF`. | Baseline measurement and an explicit non-AI live candidate after promotion. |
| `shadow` | Records GPT's `APPROVE`/`REJECT` review but does not let it block an otherwise valid paper trade. | Measure incremental AI value against the same deterministic candidates. |
| `gate` | Requires GPT approval and configured confidence/quality thresholds. Errors fail closed as rejection. | Experimental paper use; live only after separate evidence of after-cost improvement. |

AI receives structured candidate features; it does not predict fills or know
the future. It cannot change quantity, stop distance, maximum position size,
daily loss limits, maximum open positions, or maximum trades. Prompt changes,
model changes, and model-version drift must be treated as strategy changes and
revalidated.

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

Confirm these values before every paper run:

```dotenv
AI_MODE=shadow
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
factor, maximum drawdown, average R, and separate AI `APPROVE`/`REJECT` shadow
cohorts. `OFF` trades count in overall performance but never in an AI cohort.

Only `CLOSE` records with explicit finite P&L, fees, and R-multiple fields are
used. AI attribution must be `APPROVE`, `REJECT`, or `OFF`; a missing decision
is accepted only when `ai_mode=off`. Missing or legacy data is diagnosed and
excluded, and values are never reconstructed to make a run appear profitable.
The command exits nonzero when there are no complete P&L records. This is an
accounting report, not a leakage-safe backtester.

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
the strategy is safe or profitable. Do not set them until every promotion item
below has independent evidence.

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
- A broker-provided sandbox or equivalent isolated execution simulator covers
  partial fills, ambiguous submissions, stop or exit rejection, reconnects,
  process restarts, position reconciliation, orphan-order cleanup, and the
  end-of-day exit.
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
- a 15-minute opening range
- VWAP and EMA 9/20 trend confirmation
- RSI, ATR, and same-clock-time historical relative volume
- breakout candle quality and NIFTY regime
- an ATR stop and configurable R-multiple target
- signal-age and entry-drift checks
- no new entries after 14:30 and forced exit at 15:10

Default ceilings for `CAPITAL_LIMIT=100000` are approximately ₹200 planned risk
per stopped trade, ₹25,000 maximum position notional, two open positions, five
trades per day, and an ₹800 daily kill level. These are configurable loss
ceilings, not expected returns. Realized loss can exceed a planned stop because
of gaps, slippage, rejection, or market dislocation.

## Runtime state

```text
data/bot_state.json   persistent strategy/order state
data/bot.lock         single-process lock
logs/trades_YYYYMMDD.jsonl
```

State writes are atomic and the lock prevents two local bot processes from
running concurrently. Do not delete `bot_state.json` while a position or broker
order may exist. After a crash or uncertain order response, reconcile broker
orders and positions before resuming; never assume a timeout means no order was
accepted.

## Dynamic universe and broker constraints

The bot downloads the current NSE instrument list at startup and includes
normal NSE cash equities while excluding identifiable special or
compulsory-delivery series. Zerodha can still apply changing RMS and product
restrictions; a rejected entry must remain no position.

The design assumes Kite limits used by this version: up to 3,000 instruments
per WebSocket connection, 500 instruments per full-quote request, quote REST at
1 request/second, historical candles at 3 requests/second, and API orders at 10
requests/second. It uses automatic market protection for MARKET and SL-M
orders. Broker and exchange rules can change, so verify the current official
documentation before any live-readiness review. Realtime WebSocket and
historical candle access require the appropriate paid Kite Connect plan.

See [RESEARCH_NOTES.md](RESEARCH_NOTES.md) for the architecture's design
rationale.
