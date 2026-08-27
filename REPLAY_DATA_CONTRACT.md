# Point-in-Time Replay Data Contract

This contract defines the minimum evidence required to replay the intraday
strategy without using information that was unavailable at the simulated
decision time. A replay that does not satisfy this contract is exploratory and
must not be used to claim profitability or approve live trading.

## Current-data limitation

The existing `logs/trades_20260812.jsonl` through
`logs/trades_20260826.jsonl` contain 1,258 journal events and 10 completed
trades. They are useful for realized-trade accounting, incident analysis, and
parser fixtures only. They do not contain archived stock or NIFTY candles, the
full scan universe, historical instrument masters, point-in-time bid/ask
depth, or complete order and fill paths. Candidate records are derived and
selection-truncated; most rejected opportunities are absent. These logs are
therefore insufficient to reconstruct or optimize the historical strategy.

Downloading historical candles later does not repair the missing universe,
availability, spread, depth, capacity, or execution history. Any such study
must be labelled an approximation.

## Causal time semantics

All timestamps must be timezone-aware ISO-8601 values. Exchange data uses
`Asia/Kolkata`; UTC may also be stored for interoperability. Every record must
distinguish:

- `event_at`: when the exchange or source says the event occurred.
- `received_at`: when this system received the data.
- `available_at`: earliest time the replay is allowed to use the record,
  including publication/finalization delay.
- `as_of`: immutable decision cutoff shared by every input in one scan.

A replay may consume a record only when `available_at <= as_of`. It must never
substitute an end-of-day value, later correction, current instrument master,
or newer snapshot. A five-minute bar is unavailable before `bar_end` plus the
configured close grace. Revisions remain separate immutable versions; they do
not overwrite the originally available observation.

## Required point-in-time streams

Every stream must be append-only, schema-versioned, partitioned by session
date, and linked by stable `run_id`, `session_id`, and `scan_id` values.

### Session and experiment manifest

Record the exchange calendar and market phases, holidays or special sessions,
CAS/session-rules version, starting cash and positions, strategy/configuration
hash, source and dependency hashes, fee-model version, AI policy, random seed,
and registered research-trial ID. Parameters must be frozen before each
out-of-sample window.

### Daily instrument master and corporate actions

Archive the instrument master as observed before each session, including
capture/availability time, instrument token, symbol, segment, series, type,
tick and lot size, and eligibility status. Keep effective-dated listings,
delistings, symbol/token changes, splits, dividends, and adjustment factors in
a separate versioned stream. Reusing today's universe for past sessions is
survivorship and eligibility leakage.

### Five-minute bars

Store raw stock and NIFTY `open`, `high`, `low`, `close`, and `volume` with
instrument token, `bar_start`, `bar_end`, `received_at`, `available_at`, source,
revision, and adjustment identifiers. Retain enough prior sessions to compute
all EMA, ATR, RVOL, and opening-RVOL baselines using only earlier observations.

### Complete scan-universe snapshots

For every scan, store one row for every eligible instrument, including those
that are missing or stale. Required fields include LTP, day open/high/low,
previous close, cumulative volume, last-trade and tick-receipt times, active or
stale status, blocked status, and an explicit missing reason. The complete
cross-section is required to reproduce filters, percentile ranks, shortlist
truncation, and capacity competition.

### Quote and depth snapshots

Archive every preliminary, pre-AI, post-AI, and pre-submit quote request. Store
the requested and returned key sets; request, response, exchange, receipt, and
availability times; LTP and day fields; upper/lower circuits; and top bid/ask
price, quantity, and order count. Prefer all available depth levels. Missing or
stale snapshots must cause the same rejection in replay as in production.

### Decision trace

For every symbol at every stage, record the exact input data IDs and hashes,
features, thresholds, ranks, pass/fail result, rejection reason, causal cutoff,
portfolio exposure, reserved risk, and capacity. This trace is recorded before
shortlist or AI truncation and must include deterministic non-candidates.

### Execution and portfolio events

Store order intent, request, acknowledgement, update, tradebook fill, partial
fill, rejection, cancellation, and ambiguous outcome as separate ordered
events. Include event/receipt times, side, quantity, price, status, tags,
latency, positions, margin, and reconciliation results. Held-symbol tick or
top-of-book data must cover entry through exit when execution-fidelity results
are required.

### Costs and AI provenance

Archive effective-dated brokerage, taxes, exchange charges, stamp duty, and
slippage/impact model parameters, then reconcile estimates to broker tradebook
and contract-note data when available.

For AI evaluation, store every eligible deterministic candidate, the exact
identifier-stripped input and hash, prompt/schema version, pinned model
snapshot, structured response, error/usage metadata, and completion time. A
new model run on old data is a separate offline experiment, not the historical
decision.

## Completeness and provenance rules

Each record requires `schema_version`, stable event ID, monotonic source
sequence where available, source name, raw-payload hash, and ingestion version.
Daily manifests must state expected and observed record counts, gaps,
duplicates, stale inputs, and recorder outages. Missing data is represented
explicitly and must never be silently forward-filled.

The replay must reject a session when required universe, calendar, manifest,
or causal bar coverage is incomplete. Lower-fidelity execution may continue
only under a named conservative policy and must be reported separately.

### Part 4 derived replay envelope

`historical_replay.py` consumes a checksummed derived cache, not the raw streams
themselves. A dataset directory contains `manifest.json`,
`opportunities.jsonl`, and `bars.jsonl`. The manifest must identify the raw-tape
hash and source-strategy fingerprint and affirm a point-in-time,
survivorship-free, complete pre-rule decision trace. These declarations are
auditable claims by the recorder; setting their booleans manually is not proof
that the underlying data is complete.

Each opportunity stores causal decision, signal-close, feature-availability,
and quote-availability timestamps; executable bid/ask prices and quantities;
tick and circuit data; a raw-source hash; and every scalar `SetupRuleInput`.
The live detector and replay call the same `strategy_rules.py` evaluator. Each
bar stores its start and availability time, raw OHLCV, and source hash. For
every opportunity, the loader requires an unbroken path from the entry-
containing five-minute interval through the 15:10 force-exit boundary. Exact
field names and schema versions are executable in `historical_replay.py` and
covered by `tests/test_historical_replay.py`.

This envelope supports threshold, capacity, cost, and conservative bar-path
research. It does not by itself replay raw cross-sectional ranking or prove
quote availability/fills. The bar-only engine charges at least half the
decision-time spread again on exit, in addition to adverse exit slippage,
because bar OHLC is not an executable quote; the true future spread remains
unknown. Frozen replay configuration also records the live entry-cutoff guard
and minimum circuit buffer; candidates crossing either gate are rejected.
Preserve the complete raw streams above so every derived row can be regenerated
and audited.

## Conservative execution and bar ambiguity

Execution uses the executable ask for buys and bid for sells plus non-negative
adverse slippage and configured latency. It must model rejected, unfilled, and
partial orders; lack of a fresh executable quote is not a fill.

When only OHLC bars are available, apply these rules:

1. Never claim a trigger or fill before the simulated entry time. If the
   entry-containing bar crosses the stop but its pre/post-entry order is
   unknowable, book a pessimistic stop only under the separately counted
   `ENTRY_BAR_CONSERVATIVE_STOP_ASSUMPTION` label; never grant its target.
2. If stop and target are both reachable in a later bar and no finer causal
   path resolves their order, process the stop first.
3. If price gaps through a stop, fill at the first executable price no better
   than the stop; do not grant the stop price automatically.
4. Apply spread, adverse slippage, latency, and all charges to every turnover.
5. Mark bar-only results `LOW_FIDELITY` and keep them separate from tick/depth
   results. They cannot alone justify production promotion.
6. Group exits with the same bar-availability timestamp. Resolve trailing-loss
   state with wins first and losses last, and drawdown with losses first, so
   arbitrary event IDs cannot change portfolio admissions or risk statistics.
7. If a realized loss limit triggers while another position remains open,
   reproduce the live kill-switch liquidation from a causal executable quote.
   The Part 4 bar-only engine has no such quote and therefore rejects that
   session instead of allowing the position to continue or inventing a fill.

## Evaluation boundary

Research trials must be registered, chronological, and cost-aware. Parameter
selection occurs only in training windows; each walk-forward test uses frozen
parameters, and the final holdout is evaluated once. Paper, live, approximate,
and different configuration or AI cohorts must never be pooled as one
experiment. No replay, however complete, guarantees future gains.
