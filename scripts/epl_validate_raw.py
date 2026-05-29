from pathlib import Path
import pandas as pd

RAW_DATA_DIR = Path("data/raw/premier_league")

REQUIRED_COLUMNS = [
    "Div", "Date", "Time", "HomeTeam", "AwayTeam",
    "FTHG", "FTAG", "FTR",
    "HTHG", "HTAG", "HTR",
    "HS", "AS", "HST", "AST",
    "HF", "AF", "HC", "AC",
    "HY", "AY", "HR", "AR"
]

VALID_RESULTS = {"H", "D", "A"}


def validate_file(file_path: Path) -> list[str]:
    issues = []

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        return [f"Could not read file: {e}"]

    # Check row count
    if len(df) != 380:
        issues.append(f"Expected 380 matches, found {len(df)}")

    # Check required columns
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        issues.append(f"Missing required columns: {missing_cols}")

    # Stop deeper checks if required columns are missing
    if missing_cols:
        return issues

    # Check missing values in required columns
    missing_values = df[REQUIRED_COLUMNS].isna().sum()
    missing_values = missing_values[missing_values > 0]

    if not missing_values.empty:
        issues.append(f"Missing values found: {missing_values.to_dict()}")

    # Check duplicate rows
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        issues.append(f"Found {duplicate_count} duplicate rows")

    # Check date parsing
    parsed_dates = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    invalid_dates = parsed_dates.isna().sum()

    if invalid_dates > 0:
        issues.append(f"Found {invalid_dates} invalid dates")

    # Check result values
    invalid_ftr = set(df["FTR"].dropna().unique()) - VALID_RESULTS
    invalid_htr = set(df["HTR"].dropna().unique()) - VALID_RESULTS

    if invalid_ftr:
        issues.append(f"Invalid FTR values: {invalid_ftr}")

    if invalid_htr:
        issues.append(f"Invalid HTR values: {invalid_htr}")

    # Check score/result consistency
    inconsistent_results = 0

    for _, row in df.iterrows():
        home_goals = row["FTHG"]
        away_goals = row["FTAG"]
        result = row["FTR"]

        expected_result = "H" if home_goals > away_goals else "A" if away_goals > home_goals else "D"

        if result != expected_result:
            inconsistent_results += 1

    if inconsistent_results > 0:
        issues.append(f"Found {inconsistent_results} rows where FTHG/FTAG do not match FTR")

    # Check team consistency
    home_teams = set(df["HomeTeam"].dropna().unique())
    away_teams = set(df["AwayTeam"].dropna().unique())

    if home_teams != away_teams:
        issues.append("HomeTeam and AwayTeam lists do not match")

    total_teams = len(home_teams.union(away_teams))
    if total_teams != 20:
        issues.append(f"Expected 20 teams, found {total_teams}")

    return issues


def main():
    csv_files = sorted(RAW_DATA_DIR.glob("*.csv"))

    if not csv_files:
        print(f"No CSV files found in {RAW_DATA_DIR}")
        return

    print("=" * 70)
    print("RAW PREMIER LEAGUE DATA VALIDATION")
    print("=" * 70)

    all_passed = True

    for file_path in csv_files:
        print(f"\nChecking: {file_path.name}")

        issues = validate_file(file_path)

        if issues:
            all_passed = False
            print("Status: FAILED")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("Status: PASSED")

    print("\n" + "=" * 70)

    if all_passed:
        print("All raw data files passed validation.")
    else:
        print("Some files have validation issues. Review the messages above.")

    print("=" * 70)


if __name__ == "__main__":
    main()