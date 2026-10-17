import pandas as pd
import logging
from pathlib import Path

# ==================================================
# STEP 1: PROJECT PATHS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "outputs" / "Processed_Solar_Dataset.csv"
OUTPUT_DIR = BASE_DIR / "outputs" / "PowerBI"
LOG_PATH = OUTPUT_DIR / "PowerBI_Preparation.log"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def PreparePowerBIData():

    try:
        # ==========================================
        # STEP 2: LOAD DATA
        # ==========================================

        print("\n" + "=" * 60)
        print("SOLAR POWER - POWER BI DATA PREPARATION")
        print("=" * 60)

        Data = pd.read_csv(DATA_PATH)

        print("\nOriginal dataset shape:", Data.shape)

        # ==========================================
        # STEP 3: CLEAN DATA
        # ==========================================

        Required_Columns = [
            "DATE_TIME",
            "AC_POWER",
            "AMBIENT_TEMPERATURE",
            "MODULE_TEMPERATURE",
            "IRRADIATION"
        ]

        Missing_Columns = [
            Column for Column in Required_Columns
            if Column not in Data.columns
        ]

        if Missing_Columns:
            raise ValueError(
                f"Missing required columns: {Missing_Columns}"
            )

        Data["DATE_TIME"] = pd.to_datetime(
            Data["DATE_TIME"], errors="coerce"
        )

        Numeric_Columns = [
            "AC_POWER",
            "AMBIENT_TEMPERATURE",
            "MODULE_TEMPERATURE",
            "IRRADIATION"
        ]

        for Column in Numeric_Columns:
            Data[Column] = pd.to_numeric(
                Data[Column], errors="coerce"
            )

        Data = Data.dropna(
            subset=Required_Columns
        ).copy()

        Data = Data[
            (Data["AC_POWER"] >= 0)
            & (Data["IRRADIATION"] >= 0)
        ].copy()

        # ==========================================
        # STEP 4: CREATE DATE AND TIME FEATURES
        # ==========================================

        Data["DATE"] = Data["DATE_TIME"].dt.date.astype(str)
        Data["YEAR"] = Data["DATE_TIME"].dt.year
        Data["MONTH"] = Data["DATE_TIME"].dt.month
        Data["MONTH_NAME"] = Data["DATE_TIME"].dt.strftime("%B")
        Data["DAY"] = Data["DATE_TIME"].dt.day
        Data["HOUR"] = Data["DATE_TIME"].dt.hour
        Data["MINUTE"] = Data["DATE_TIME"].dt.minute
        Data["DAY_OF_WEEK"] = Data["DATE_TIME"].dt.day_name()

        # Sort month names correctly in Power BI using MONTH
        Data["IS_DAYLIGHT"] = (
            Data["IRRADIATION"] >= 0.2
        ).map({True: "Yes", False: "No"})

        # Categories are analytical labels, not fault diagnoses
        Data["GENERATION_CATEGORY"] = pd.cut(
            Data["AC_POWER"],
            bins=[
                -0.001,
                Data["AC_POWER"].quantile(0.25),
                Data["AC_POWER"].quantile(0.75),
                float("inf")
            ],
            labels=["Low", "Medium", "High"],
            duplicates="drop"
        )

        # ==========================================
        # STEP 5: EXPORT CLEAN POWER BI DATASET
        # ==========================================

        PowerBI_Columns = [
            "DATE_TIME",
            "DATE",
            "YEAR",
            "MONTH",
            "MONTH_NAME",
            "DAY",
            "HOUR",
            "MINUTE",
            "DAY_OF_WEEK",
            "AC_POWER",
            "AMBIENT_TEMPERATURE",
            "MODULE_TEMPERATURE",
            "IRRADIATION",
            "IS_DAYLIGHT",
            "GENERATION_CATEGORY"
        ]

        PowerBI_Data = Data[PowerBI_Columns].copy()

        PowerBI_Data.to_csv(
            OUTPUT_DIR / "Solar_PowerBI_Dataset.csv",
            index=False
        )

        # ==========================================
        # STEP 6: DAILY SUMMARY
        # ==========================================

        Daily_Summary = (
            Data.groupby("DATE", as_index=False)
            .agg(
                Average_AC_Power=("AC_POWER", "mean"),
                Maximum_AC_Power=("AC_POWER", "max"),
                Average_Irradiation=("IRRADIATION", "mean"),
                Average_Module_Temperature=(
                    "MODULE_TEMPERATURE", "mean"
                ),
                Total_Records=("AC_POWER", "count")
            )
        )

        Daily_Summary.to_csv(
            OUTPUT_DIR / "PowerBI_Daily_Summary.csv",
            index=False
        )

        # ==========================================
        # STEP 7: MONTHLY SUMMARY
        # ==========================================

        Monthly_Summary = (
            Data.groupby(
                ["YEAR", "MONTH", "MONTH_NAME"],
                as_index=False
            )
            .agg(
                Average_AC_Power=("AC_POWER", "mean"),
                Maximum_AC_Power=("AC_POWER", "max"),
                Average_Irradiation=("IRRADIATION", "mean"),
                Total_Records=("AC_POWER", "count")
            )
            .sort_values(["YEAR", "MONTH"])
        )

        Monthly_Summary.to_csv(
            OUTPUT_DIR / "PowerBI_Monthly_Summary.csv",
            index=False
        )

        # ==========================================
        # STEP 8: KPI SUMMARY
        # ==========================================

        KPI_Summary = pd.DataFrame([
            {
                "KPI": "Total Records",
                "Value": len(Data)
            },
            {
                "KPI": "Average AC Power",
                "Value": Data["AC_POWER"].mean()
            },
            {
                "KPI": "Maximum AC Power",
                "Value": Data["AC_POWER"].max()
            },
            {
                "KPI": "Average Irradiation",
                "Value": Data["IRRADIATION"].mean()
            },
            {
                "KPI": "Average Module Temperature",
                "Value": Data["MODULE_TEMPERATURE"].mean()
            },
            {
                "KPI": "Daylight Records",
                "Value": int((Data["IRRADIATION"] >= 0.2).sum())
            },
            {
                "KPI": "Zero Power Records",
                "Value": int((Data["AC_POWER"] == 0).sum())
            }
        ])

        KPI_Summary.to_csv(
            OUTPUT_DIR / "PowerBI_KPI_Summary.csv",
            index=False
        )

        # ==========================================
        # STEP 9: COMPLETION SUMMARY
        # ==========================================

        print("\nClean dataset shape:", PowerBI_Data.shape)

        print("\nKPI Summary:")
        print(KPI_Summary.to_string(index=False))

        print("\nFiles created successfully:")
        print("1. Solar_PowerBI_Dataset.csv")
        print("2. PowerBI_Daily_Summary.csv")
        print("3. PowerBI_Monthly_Summary.csv")
        print("4. PowerBI_KPI_Summary.csv")

        print("\nOutput folder:", OUTPUT_DIR)

        logging.info(
            "Power BI data preparation completed successfully."
        )

    except FileNotFoundError:
        print("\nDataset not found:", DATA_PATH)
        logging.exception("Input dataset not found.")

    except (ValueError, KeyError) as Error:
        print("\nData preparation error:", Error)
        logging.exception("Power BI data preparation failed.")

if __name__ == "__main__":
    PreparePowerBIData()