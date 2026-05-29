from pathlib import Path
import pandas as pd

RAW_DATA_DIR = Path("data/raw/premier_league")
OUTPUT_PATH = Path("data/processed/epl_combined_cleaned.csv")

CORE_COLUMNS = [
    "Season", "Div", "Date", "Time",
    "HomeTeam", "AwayTeam",
    "FTHG", "FTAG", "FTR",
    "HTHG", "HTAG", "HTR",
    "Referee",
    "HS", "AS", "HST", "AST",
    "HF", "AF", "HC", "AC",
    "HY", "AY", "HR", "AR"
]


def get_season_from_filename(file_path: Path) -> str:
    return file_path.stem.replace("epl_", "")


def clean_file(file_path: Path) -> pd.DataFrame:
    df = pd.read_csv(file_path)

    df["Season"] = get_season_from_filename(file_path)

    for col in CORE_COLUMNS:
        if col not in df.columns:
            df[col] = None

    df = df[CORE_COLUMNS]

    df["Date"] = pd.to_datetime(
        df["Date"],
        dayfirst=True,
        errors="coerce"
    )

    df["HomeTeam"] = df["HomeTeam"].astype(str).str.strip()
    df["AwayTeam"] = df["AwayTeam"].astype(str).str.strip()
    df["FTR"] = df["FTR"].astype(str).str.strip()
    df["HTR"] = df["HTR"].astype(str).str.strip()

    return df


def main():
    csv_files = sorted(RAW_DATA_DIR.glob("*.csv"))

    if not csv_files:
        print(f"No CSV files found in {RAW_DATA_DIR}")
        return

    all_data = []

    for file_path in csv_files:
        print(f"Cleaning {file_path.name}")
        df = clean_file(file_path)
        all_data.append(df)

    combined_df = pd.concat(all_data, ignore_index=True)

    combined_df = combined_df.drop_duplicates()
    combined_df = combined_df.sort_values(
        by=["Date", "Time"],
        na_position="last"
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nCleaned dataset saved to: {OUTPUT_PATH}")
    print(f"Total rows: {len(combined_df)}")
    print(f"Total columns: {len(combined_df.columns)}")
    print(f"Seasons included: {combined_df['Season'].nunique()}")
    print("\nMissing values by column:")
    print(combined_df.isna().sum())


if __name__ == "__main__":
    main()