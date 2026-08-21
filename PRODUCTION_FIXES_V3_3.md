# V3.3 live order/recovery fixes

This revision is focused on the live-order lifecycle failure reproduced by the
AEROFLEX session on 2026-08-21. Strategy/indicator logic is intentionally not
retuned here; the changes are broker-state, idempotency, recovery and operational
safety changes.

## Root cause addressed

The entry was accepted and filled. The protective SL-M existed and was reported
as `TRIGGER PENDING` with pending quantity equal to the position quantity, but
strict local identity verification could disown that stop when a broker payload
omitted/changed non-essential metadata such as the local tag. The old emergency
path then correctly refused to blindly market-exit while an apparently unknown
sell stop remained active, because doing so could leave an orphan order that
later reverses a flat position. The kill switch consequently retried the same
blocked recovery continuously.

## Changes

- Broker `order_id` is the primary identity after it has been persisted. Tags are
  still used to recover an ambiguous submission before an order ID is known.
- An SL-M with `TRIGGER PENDING`, the exact broker order ID, correct symbol,
  product, side, quantity and trigger is accepted as armed protection even when a
  tag is absent in an asynchronous/history payload.
- Order verification logs field-level stop identity mismatch reasons.
- `latest_order()` prefers the full current order book and uses WebSocket/history
  only as event/fallback sources. Sparse updates no longer erase known metadata.
- Duplicate/out-of-order WebSocket order updates are idempotently merged. A
  terminal state cannot regress to a stale non-terminal state.
- Tagged order submission remains submit-once. An ambiguous response is resolved
  by exact tag; it is never blindly resubmitted.
- Dedicated-account mode (`DEDICATED_BOT_ACCOUNT=true`) can recover an affected
  NSE/MIS symbol without requiring a local tag: cancel all active NSE/MIS orders
  for the symbol, re-read the signed position, submit one market-protected exit
  for only the residual quantity, verify flat, then verify no active symbol order
  remains.
- A fill/cancel race is re-read from broker state before a second action.
- If a dedicated recovery exit submission itself becomes ambiguous, that symbol
  is marked ambiguous in-process and no second automatic exit is submitted until
  broker state proves it flat.
- Dedicated startup recovery can terminalize unowned NSE/MIS orders and flatten
  untracked NSE/MIS positions. Non-dedicated mode retains fail-closed ownership
  checks.
- Kill-switch flatten attempts are bounded by
  `MAX_KILL_SWITCH_FLATTEN_ATTEMPTS`. Exhaustion sets
  `manual_intervention_required=true` and stops further automatic mutations.
- `recover_account.py` provides an explicit operator recovery path for a
  dedicated account and can optionally back up/reset state only after broker-flat
  verification.

## Tests run

The included suite was run with external network/broker SDK calls stubbed and all
broker mutations simulated locally:

- `python -m unittest discover -s tests -p 'test_*.py' -v`
- 92 tests passed.
- Pytest equivalent: 92 passed plus 25 parameterized/subtests passed.
- Python bytecode compilation succeeded for the bot, helpers, recovery utility
  and tests.

Regression coverage includes the exact failure class: a 44-share protective
SL-M in `TRIGGER PENDING` with an exact persisted order ID and an omitted tag is
recognized as armed protection; duplicate WebSocket updates are idempotent;
dedicated recovery cancels an unknown stop then exits the residual exactly once;
a stop that fills during cancellation does not cause a second exit; ambiguous
recovery submission is not resubmitted; and kill-switch mutations stop after the
retry cap.

## Limits of the test claim

The test suite does not connect to a real Zerodha account or exchange and cannot
simulate every OMS/network/exchange failure. This is an execution-safety software
release, not a guarantee of profitability or zero operational risk. Use a small,
supervised live canary after broker-flat recovery and verify order/position state
in Kite after each first live lifecycle.
