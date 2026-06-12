import pandas as pd
import glob

tickers = set()

for file in glob.glob("tickers/*.csv"):
    df = pd.read_csv(file)

    if "Symbol" in df.columns:
        tickers.update(
            df["Symbol"].dropna().astype(str).str.strip()
        )

output = ",".join(f'"{ticker}"' for ticker in sorted(tickers))
with open("tickers.txt", "w") as f:
    f.write(output)
print(output)