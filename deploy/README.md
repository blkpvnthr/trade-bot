# Scheduling the Intraday Reversal Engine

The engine is an **isolated, optional candidate source** (it does not modify `run.py`).
These units run it on a schedule:

1. **Afternoon scan** — after **1:30 PM ET** it selects analyst+oscillator "strong buy"
   names that are weak over 1 week, red today, and **moving toward their Previous Day
   High (PDH)**. Output is dry-run (ranked candidates + proposed entries/calls); live
   routing is the caller's job through `run.AlpacaGateway.submit_market_order`, so every
   existing safety gate still applies. The runner self-gates to the
   `[INTRADAY_REVERSAL_START_ET, INTRADAY_REVERSAL_ENTRY_CUTOFF_ET)` window (default
   `13:30`–`15:55`) and runs **at most once per day** (marker under
   `logs/intraday_reversal/scan_YYYYMMDD.done`).

2. **EOD management** — near the close (~15:55 ET), for each open reversal position:
   - PDH hit intraday → target reached (exit engine takes it);
   - **PDH not hit but position UP → tighten the trailing stop
     (`INTRADAY_REVERSAL_OVERNIGHT_TRAIL_PCT`, default 0.40%) and carry overnight to
     sell into the NEXT DAY'S HIGH** (`next_day_exit_decision` / `INTRADAY_REVERSAL_NEXT_DAY_TRAIL_PCT`);
   - PDH not hit and not up → close at EOD.

## systemd (Linux droplet)

```bash
# Edit WorkingDirectory / User / venv / EnvironmentFile in the .service files first.
sudo cp deploy/intraday-reversal-*.service deploy/intraday-reversal-*.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now intraday-reversal-scan.timer intraday-reversal-eod.timer
systemctl list-timers 'intraday-reversal-*'
```
The `OnCalendar=... America/New_York` suffix needs systemd ≥ v252. On older systemd,
set the host TZ to ET or convert the times to UTC (EDT = ET+4h, EST = ET+5h).

## cron (alternative)

```cron
# m h dom mon dow  command   (set CRON_TZ so these are ET; runner also self-gates)
CRON_TZ=America/New_York
35 13 * * 1-5  cd /root/trade-bot-instance2 && .venv/bin/python intraday_reversal_engine.py --scan       >> logs/intraday_reversal/scan.log 2>&1
55 15 * * 1-5  cd /root/trade-bot-instance2 && .venv/bin/python intraday_reversal_engine.py --manage-eod --force --positions system-data/reversal_positions.json >> logs/intraday_reversal/eod.log 2>&1
```

## Manual / smoke

```bash
python intraday_reversal_engine.py --scan --force          # run the scan now (bypass window/marker)
python intraday_reversal_engine.py --scan --universe my_rows.json --funnel-out /tmp/f.json
python intraday_reversal_engine.py --manage-eod            # demo EOD actions
python intraday_reversal_engine.py --manage-eod --positions system-data/reversal_positions.json
```

`--universe` / `--positions` take a JSON list (or `{rows:[...]}` / `{positions:[...]}`)
whose keys map onto `ReversalInput` / `ManagedPosition` fields. All output is dry-run.
