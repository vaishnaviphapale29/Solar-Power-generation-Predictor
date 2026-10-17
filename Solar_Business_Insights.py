import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import logging

# Project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "outputs" / "Processed_Solar_Dataset.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
IMAGE_DIR = BASE_DIR / "images"

OUTPUT_DIR.mkdir(exist_ok=True)
IMAGE_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    filename=OUTPUT_DIR / "business_insights.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def GenerateBusinessInsights():
    try:
        # Load processed dataset
        Data = pd.read_csv(DATA_PATH)
        Data["DATE_TIME"] = pd.to_datetime(Data["DATE_TIME"])

        # Keep valid numeric values
        Numeric_Columns = [
            "AC_POWER",
            "IRRADIATION",
            "MODULE_TEMPERATURE",
            "AMBIENT_TEMPERATURE"
        ]

        for Column in Numeric_Columns:
            Data[Column] = pd.to_numeric(Data[Column], errors="coerce")

        Data = Data.dropna(subset=Numeric_Columns + ["DATE_TIME"])

        # 1. Best-performing date and time
        Peak_Index = Data["AC_POWER"].idxmax()
        Peak_Record = Data.loc[[Peak_Index]]

        Peak_Record.to_csv(
            OUTPUT_DIR / "Peak_Solar_Performance.csv",
            index=False
        )

        # 2. Daily performance analysis
        Daily_Analysis = (
            Data.groupby("DATE")
            .agg(
                Average_AC_Power=("AC_POWER", "mean"),
                Peak_AC_Power=("AC_POWER", "max"),
                Average_Irradiation=("IRRADIATION", "mean"),
                Records=("AC_POWER", "count")
            )
            .reset_index()
        )

        Daily_Analysis.to_csv(
            OUTPUT_DIR / "Advanced_Daily_Analysis.csv",
            index=False
        )

        # 3. Low-generation records during daylight
        Daylight_Data = Data[Data["IRRADIATION"] >= 0.2].copy()

        if not Daylight_Data.empty:
            Low_Threshold = Daylight_Data["AC_POWER"].quantile(0.10)

            Low_Generation = Daylight_Data[
                Daylight_Data["AC_POWER"] <= Low_Threshold
            ].copy()

            Low_Generation["Analysis_Flag"] = (
                "Low generation during relatively high irradiation"
            )

            Low_Generation.to_csv(
                OUTPUT_DIR / "Low_Generation_Records.csv",
                index=False
            )

        # 4. High irradiation but unusually low power
        if not Daylight_Data.empty:
            Irradiation_Threshold = Daylight_Data["IRRADIATION"].quantile(0.75)
            Power_Threshold = Daylight_Data["AC_POWER"].quantile(0.25)

            Unusual_Records = Daylight_Data[
                (Daylight_Data["IRRADIATION"] >= Irradiation_Threshold)
                & (Daylight_Data["AC_POWER"] <= Power_Threshold)
            ].copy()

            Unusual_Records["Analysis_Flag"] = (
                "Review: high irradiation with low power"
            )

            Unusual_Records.to_csv(
                OUTPUT_DIR / "High_Irradiation_Low_Power.csv",
                index=False
            )

        # 5. Business summary
        Best_Day = Daily_Analysis.loc[
            Daily_Analysis["Average_AC_Power"].idxmax()
        ]

        Summary = pd.DataFrame([
            {
                "Metric": "Total Records",
                "Value": len(Data)
            },
            {
                "Metric": "Average AC Power",
                "Value": Data["AC_POWER"].mean()
            },
            {
                "Metric": "Maximum AC Power",
                "Value": Data["AC_POWER"].max()
            },
            {
                "Metric": "Best Average Generation Date",
                "Value": Best_Day["DATE"]
            },
            {
                "Metric": "Average Irradiation",
                "Value": Data["IRRADIATION"].mean()
            }
        ])

        Summary.to_csv(
            OUTPUT_DIR / "Advanced_Business_Summary.csv",
            index=False
        )

        # 6. Daily average generation graph
        Daily_Analysis["DATE"] = pd.to_datetime(Daily_Analysis["DATE"])
        Daily_Analysis = Daily_Analysis.sort_values("DATE")

        plt.figure(figsize=(12, 6))
        plt.plot(
            Daily_Analysis["DATE"],
            Daily_Analysis["Average_AC_Power"],
            marker="o"
        )
        plt.title("Daily Average Solar Power Generation")
        plt.xlabel("Date")
        plt.ylabel("Average AC Power")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(
            IMAGE_DIR / "Advanced_Daily_Generation.png",
            dpi=150
        )
        plt.show()
        plt.close()

        # 7. Print business findings
        print("\n" + "=" * 55)
        print("ADVANCED SOLAR BUSINESS INSIGHTS")
        print("=" * 55)

        print("\nPeak power record:")
        print(Peak_Record[
            ["DATE_TIME", "AC_POWER", "IRRADIATION"]
        ].to_string(index=False))

        print("\nBest average-generation date:")
        print(Best_Day.to_string())

        print("\nBusiness recommendations:")
        print("- Review low-generation records during daylight.")
        print("- Investigate high-irradiation, low-power records.")
        print("- Compare daily generation trends for performance changes.")
        print("- Check equipment and sensor data before concluding a fault.")

        print("\nReports saved in:", OUTPUT_DIR)
        print("Graph saved in:", IMAGE_DIR)

        logging.info("Business insights generated successfully.")

    except FileNotFoundError:
        print("Dataset not found:", DATA_PATH)
        logging.exception("Processed dataset missing.")

    except (ValueError, KeyError) as Error:
        print("Data analysis error:", Error)
        logging.exception("Business analysis failed.")

if __name__ == "__main__":
    GenerateBusinessInsights()