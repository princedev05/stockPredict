import yfinance as yf
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

def download_btc_hourly(period="2y", interval="1h"):
    df = yf.download(
        tickers="BTC-USD",
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False
    )

    # Flatten MultiIndex if present, lowercase & select OHLCV
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.index.name   = "date"
    df.columns      = df.columns.str.lower()
    df               = df[["open", "high", "low", "close", "volume"]]
    return df

def split_train_test(
    df: pd.DataFrame,
    train_months: int = 12,
    test_months_before: int = 1,
    test_months_after: int = 1
):
    """
    From a long time series `df`, carve out:
     - `test_months_before` immediately before the training window
     - `train_months` of training
     - `test_months_after` immediately after training
    Returns (train, test) where test is the concat of before + after.
    """
    df = df.sort_index()
    first_ts = df.index[0]
    last_ts  = df.index[-1]

    # Define window boundaries
    train_start       = first_ts + pd.DateOffset(months=test_months_before)
    train_end         = last_ts  - pd.DateOffset(months=test_months_after)
    test_before_start = first_ts
    test_before_end   = train_start
    test_after_start  = train_end
    test_after_end    = last_ts

    # Slice
    test_before = df[
        (df.index >= test_before_start) &
        (df.index <  test_before_end)
    ]
    train = df[
        (df.index >= train_start) &
        (df.index <= train_end)
    ]
    test_after = df[
        (df.index >  test_after_start) &
        (df.index <= test_after_end)
    ]

    # Combine the two test pieces
    test = pd.concat([test_before, test_after]).sort_index()
    return train, test

def main():
    print("Downloading BTC-USD hourly data (2 years)…")
    df = download_btc_hourly(period="2y", interval="1h")

    print("Splitting into:")
    print("  • test_before  = 1 month BEFORE the 12 mo train window")
    print("  • train        = 12 months")
    print("  • test_after   = 1 month AFTER the train window")
    train, test = split_train_test(
        df,
        train_months=12,
        test_months_before=1,
        test_months_after=1
    )
    print(f"  • train rows: {len(train)}")
    print(f"  • test rows:  {len(test)} (1 mo before + 1 mo after)")

    print("Saving to CSV…")
    train.to_csv("train.csv")
    test.to_csv("test.csv")

    # (Optionally scale and save your scaler here)
    # scaler = StandardScaler().fit(train)
    # joblib.dump(scaler, "scaler.pkl")
    print("Done!")

if __name__ == "__main__":
    main()



