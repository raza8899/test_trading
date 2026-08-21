# Dedicated-account live recovery runbook

Use this only if the Zerodha account is dedicated to this bot for NSE/MIS
intraday activity.

## If a live position/order is currently stuck

1. Stop the old bot process. Do not run two bot versions at once.
2. Verify in Kite the current NSE/MIS position and all active orders.
3. Deploy this V3.3 build and set in `.env`:

```dotenv
LIVE_TRADING=true
LIVE_TRADING_CONFIRM=I_UNDERSTAND_REAL_MONEY
DEDICATED_BOT_ACCOUNT=true
AI_IDEA_MODE=off
```

Keep your existing Kite credentials/static IP configuration. Do not copy the
example file over a working `.env` containing credentials.

4. From the project directory, first run the explicit recovery utility:

```bash
.venv/bin/python recover_account.py \
  --confirm FLATTEN_DEDICATED_ACCOUNT
```

It only operates on NSE/MIS orders and positions. It cancels active orders for
an affected symbol before flattening the signed residual position, then verifies
that both the position and active orders are gone.

5. Re-check Kite manually. Do not proceed unless NSE/MIS is flat and no orphan
stop/exit order remains.

6. If the old `data/bot_state.json` still describes the pre-fix active trade,
run the recovery utility again with the explicit reset option **only after the
broker is verified flat**:

```bash
.venv/bin/python recover_account.py \
  --confirm FLATTEN_DEDICATED_ACCOUNT \
  --reset-state-after-flat
```

The old state is backed up with a timestamp before a fresh state is written.
Historical JSONL journals are not deleted.

7. Start V3.3 in the foreground for the first canary session and watch order
lifecycle logs. Only move back to `nohup`/systemd after you have observed one
complete entry -> armed stop -> exit lifecycle.

## Never do this

- Do not delete/reset state while a real position/order is still present.
- Do not market-exit a long while leaving its sell stop active; the stop could
  later create an unintended short position.
- Do not restart the old process after the new recovery build is running.
- Do not disable the kill switch merely to force new entries.
