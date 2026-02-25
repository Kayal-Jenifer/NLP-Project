from .divider import divider


def statistics(df):
    divider("DATASET STATISTICS")

    print(f"Dataset Shape : {df.shape}")
    print(f"Reviews Count : {len(df):,}")
    print(f"Columns Count : {len(df.columns)}")

    print(f"\nColumn names and types :")
    print(df.dtypes)

    print(f"\nDataset info:")
    df.info()

    print(f"\nFirst few rows:")
    print(df.head())

    print(f"\nMissing values :")
    missing = df.isnull().sum()

    print(missing[missing > 0] if missing.sum() > 0 else "No missing values found!")

    print(f"\nBasic statistics:")
    print(df.describe())

    return df
