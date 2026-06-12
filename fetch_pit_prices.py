import os, json, glob
from pathlib import Path
from datetime import date, datetime, timedelta
for line in Path('.env').read_text().splitlines():
    if '=' in line and not line.strip().startswith('#'):
        k,v=line.split('=',1); os.environ.setdefault(k.strip(), v.strip())
key=os.environ.get('ALPACA_API_KEY') or os.environ.get('APCA_API_KEY_ID') or os.environ.get('ALPACA_API_KEY_ID')
sec=os.environ.get('ALPACA_SECRET_KEY') or os.environ.get('APCA_API_SECRET_KEY') or os.environ.get('ALPACA_API_SECRET_KEY')
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed
import pandas as pd
client=StockHistoricalDataClient(key, sec)
BARS=Path("data/cache/alpaca_bars")

ev=json.load(open("data/cache/pit_insider_events.json"))
# eligible: >$1M 2-day cluster
elig=[]
for sym,evs in ev.items():
    rows=sorted((date.fromisoformat(d),v) for d,v in evs)
    for d,_ in rows:
        if sum(v for dd,v in rows if d-timedelta(days=2)<=dd<=d)>1_000_000:
            elig.append(sym); break
elig=[s for s in elig if s.isalpha() and 1<=len(s)<=5]
have={os.path.basename(p).rsplit('_day',1)[0] for p in glob.glob('data/cache/alpaca_bars/*_day.parquet')}
todo=[s for s in elig if s not in have]
print(f"eligible(clean)={len(elig)} already_have={len(elig)-len(todo)} to_fetch={len(todo)}")

start=datetime(2020,6,1); end=datetime(2026,6,7)
fetched=0; empty=0
B=100
for i in range(0,len(todo),B):
    batch=todo[i:i+B]
    try:
        req=StockBarsRequest(symbol_or_symbols=batch, timeframe=TimeFrame.Day, start=start, end=end, feed=DataFeed.IEX)
        df=client.get_stock_bars(req).df
    except Exception as e:
        print(f"  batch {i//B}: ERR {str(e)[:60]}"); continue
    if df is None or df.empty:
        empty+=len(batch); continue
    got=set(df.index.get_level_values(0))
    for s in batch:
        if s not in got: empty+=1; continue
        d=df.xs(s, level=0)[["open","high","low","close","volume"]].copy(); d.index.name="timestamp"
        if len(d)<30: empty+=1; continue
        d.to_parquet(BARS/f"{s}_day.parquet"); fetched+=1
    print(f"  batch {i//B+1}/{(len(todo)+B-1)//B}: fetched so far={fetched} empty={empty}")
print(f"DONE fetched={fetched} empty/unavailable={empty}")
# coverage
have2={os.path.basename(p).rsplit('_day',1)[0] for p in glob.glob('data/cache/alpaca_bars/*_day.parquet')}
priced=[s for s in elig if s in have2]
json.dump({"eligible":len(elig),"priced":len(priced),"drop_rate":round(1-len(priced)/len(elig),3)}, open("data/cache/pit_coverage.json","w"))
print(f"COVERAGE: eligible={len(elig)} priced={len(priced)} unpriceable={len(elig)-len(priced)} drop_rate={1-len(priced)/len(elig):.1%}")
