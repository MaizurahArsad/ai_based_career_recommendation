"""Create a 365-day heat map of student money spending."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


OUTPUT_DIR = Path("outputs")
DATA_PATH = OUTPUT_DIR / "student_spending_365_days.csv"
HEATMAP_PATH = OUTPUT_DIR / "student_spending_heatmap_365_days.png"


def create_student_spending_data(days=365, start_date="2016-01-01"):
    """Create realistic sample daily spending data for one student."""
    rng = np.random.default_rng(83)
    dates = pd.date_range(start=start_date, periods=days, freq="D")

    weekday = dates.weekday
    month = dates.month

    base_spending = rng.normal(loc=28, scale=8, size=days)
    weekend_extra = np.where(weekday >= 5, rng.normal(12, 4, size=days), 0)
    semester_extra = np.where(np.isin(month, [2, 3, 4, 5, 9, 10, 11]), 8, 0)
    holiday_saving = np.where(np.isin(month, [6, 7, 12]), -6, 0)
    occasional_large_purchase = rng.choice([0, 0, 0, 0, 45, 75], size=days)

    spending = (
        base_spending
        + weekend_extra
        + semester_extra
        + holiday_saving
        + occasional_large_purchase
    )
    spending = np.clip(spending, 2, None).round(2)

    return pd.DataFrame(
        {
            "Date": dates,
            "Year": dates.year,
            "DayOfYear": dates.dayofyear,
            "Daily_Spending_RM": spending,
        }
    )


def save_spending_heatmap(df):
    """Save a year-by-day heat map image from student spending data."""
    heatmap_data = df.pivot(
        index="Year",
        columns="DayOfYear",
        values="Daily_Spending_RM",
    )

    plt.figure(figsize=(22, 7))
    sns.heatmap(
        heatmap_data,
        cmap="YlOrRd",
        cbar_kws={"label": "Daily Spending (RM)"},
        linewidths=0,
    )
    plt.title("Student Money Spending Heat Map Across 365 Days")
    plt.xlabel("Day of Year")
    plt.ylabel("Year")
    plt.tight_layout()
    plt.savefig(HEATMAP_PATH, dpi=200)
    plt.close()


def main():
    """Create the spending dataset and heat map files."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    df = create_student_spending_data()
    df.to_csv(DATA_PATH, index=False)
    save_spending_heatmap(df)

    print(f"Created dataset: {DATA_PATH}")
    print(f"Created heat map: {HEATMAP_PATH}")
    print(f"Total days: {len(df)}")
    print(f"Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")


if __name__ == "__main__":
    main()
