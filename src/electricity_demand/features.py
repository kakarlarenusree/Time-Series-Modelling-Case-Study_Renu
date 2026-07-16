import pandas as pd


def create_features(df, date_column="date"):

    df = df.copy()

    df[date_column] = pd.to_datetime(df[date_column])

    # Time-based features
    df["year"] = df[date_column].dt.year
    df["month"] = df[date_column].dt.month
    df["day"] = df[date_column].dt.day
    df["hour"] = df[date_column].dt.hour
    df["day_of_week"] = df[date_column].dt.dayofweek

    # Lag features
    df["lag_1"] = df["demand"].shift(1)
    df["lag_24"] = df["demand"].shift(24)
    df["lag_168"] = df["demand"].shift(168)

    # Rolling features
    df["rolling_mean_24"] = (
        df["demand"]
        .rolling(window=24)
        .mean()
    )

    df = df.dropna()

    return df