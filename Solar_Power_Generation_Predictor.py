import logging
import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
IMAGE_DIR = BASE_DIR / "images"
OUTPUT_DIR = BASE_DIR / "outputs"

for Folder in [DATA_DIR, IMAGE_DIR, OUTPUT_DIR]:
    Folder.mkdir(parents=True, exist_ok=True)

GENERATION_PATH = DATA_DIR / "Plant_1_Generation_Data.csv"
WEATHER_PATH = DATA_DIR / "Plant_1_Weather_Sensor_Data.csv"

LOG_PATH = OUTPUT_DIR / "project.log"

logging.basicConfig(
    filename=str(LOG_PATH),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8"
)

# ============================================================
# 2. COMMON FUNCTIONS
# ============================================================

def DisplayInfo(Message):

    print("\n" + "=" * 65)
    print(Message)
    print("=" * 65)

    logging.info(Message)

def SaveAndShowGraph(FileName):

    Graph_Path = IMAGE_DIR / FileName

    plt.tight_layout()
    plt.savefig(Graph_Path, dpi=300, bbox_inches="tight")

    plt.show(block=True)
    plt.close()

    print("Graph saved:", Graph_Path)

def SaveCSV(DataFrame, FileName):

    File_Path = OUTPUT_DIR / FileName

    DataFrame.to_csv(File_Path, index=False)

    print("Report saved:", File_Path)

    logging.info("Report saved: %s", File_Path)

# ============================================================
# 3. LOAD DATASETS
# ============================================================

def LoadDataset():

    DisplayInfo("STEP 1: LOAD DATASETS")

    if not GENERATION_PATH.exists():
        raise FileNotFoundError(
            f"Generation dataset not found: {GENERATION_PATH}"
        )

    if not WEATHER_PATH.exists():
        raise FileNotFoundError(
            f"Weather dataset not found: {WEATHER_PATH}"
        )

    Generation_Data = pd.read_csv(GENERATION_PATH)
    Weather_Data = pd.read_csv(WEATHER_PATH)

    print("Generation Shape:", Generation_Data.shape)
    print("Weather Shape:", Weather_Data.shape)

    print("\nGeneration Dataset:")
    print(Generation_Data.head())

    print("\nWeather Dataset:")
    print(Weather_Data.head())

    return Generation_Data, Weather_Data

# ============================================================
# 4. DATA QUALITY REPORT
# ============================================================

def DataQualityReport(Generation_Data, Weather_Data):

    DisplayInfo("STEP 2: DATA QUALITY REPORT")

    Datasets = {
        "Generation": Generation_Data,
        "Weather": Weather_Data
    }

    Numeric_Columns = {
        "Generation": ["AC_POWER"],
        "Weather": [
            "AMBIENT_TEMPERATURE",
            "MODULE_TEMPERATURE",
            "IRRADIATION"
        ]
    }

    Report = []

    for Dataset_Name, Dataset in Datasets.items():

        Duplicate_Count = int(Dataset.duplicated().sum())

        print(f"\n{Dataset_Name} Dataset")
        print("Rows:", len(Dataset))
        print("Columns:", len(Dataset.columns))
        print("Duplicate Rows:", Duplicate_Count)

        for Column in Dataset.columns:

            Missing_Count = int(Dataset[Column].isna().sum())

            Missing_Percent = (
                Missing_Count / len(Dataset) * 100
                if len(Dataset) else 0
            )

            Invalid_Datetime = 0
            Invalid_Numeric = 0

            if Column == "DATE_TIME":

                Converted = pd.to_datetime(
                    Dataset[Column], errors="coerce"
                )

                Invalid_Datetime = int(
                    (Dataset[Column].notna() & Converted.isna()).sum()
                )

            if Column in Numeric_Columns[Dataset_Name]:

                Converted = pd.to_numeric(
                    Dataset[Column], errors="coerce"
                )

                Invalid_Numeric = int(
                    (Dataset[Column].notna() & Converted.isna()).sum()
                )

            Report.append({
                "Dataset": Dataset_Name,
                "Column": Column,
                "Total_Rows": len(Dataset),
                "Missing_Values": Missing_Count,
                "Missing_Percent": round(Missing_Percent, 2),
                "Duplicate_Rows": Duplicate_Count,
                "Invalid_Datetime": Invalid_Datetime,
                "Invalid_Numeric": Invalid_Numeric
            })

    Report_Data = pd.DataFrame(Report)

    SaveCSV(Report_Data, "Data_Quality_Report.csv")

    return Report_Data

# ============================================================
# 5. UNDERSTAND DATA
# ============================================================

def UnderstandDataset(Generation_Data, Weather_Data):

    DisplayInfo("STEP 3: UNDERSTAND DATA")

    for Name, Dataset in [
        ("Generation", Generation_Data),
        ("Weather", Weather_Data)
    ]:

        print(f"\n{Name} Dataset Information:")
        Dataset.info()

        print(f"\n{Name} Missing Values:")
        print(Dataset.isnull().sum())

        print(f"\n{Name} Statistical Summary:")
        print(Dataset.describe(include="all"))

        print(f"\n{Name} Columns:")
        print(Dataset.columns.tolist())

# ============================================================
# 6. DATA PREPROCESSING
# ============================================================

def PreprocessData(Generation_Data, Weather_Data):

    DisplayInfo("STEP 4: DATA PREPROCESSING")

    Generation_Data = Generation_Data.copy()
    Weather_Data = Weather_Data.copy()

    Required_Generation = ["DATE_TIME", "AC_POWER"]

    Required_Weather = [
        "DATE_TIME",
        "AMBIENT_TEMPERATURE",
        "MODULE_TEMPERATURE",
        "IRRADIATION"
    ]

    for Column in Required_Generation:

        if Column not in Generation_Data.columns:
            raise ValueError(
                f"Missing generation column: {Column}"
            )

    for Column in Required_Weather:

        if Column not in Weather_Data.columns:
            raise ValueError(
                f"Missing weather column: {Column}"
            )

    # Convert timestamps
    Generation_Data["DATE_TIME"] = pd.to_datetime(
        Generation_Data["DATE_TIME"], errors="coerce"
    )

    Weather_Data["DATE_TIME"] = pd.to_datetime(
        Weather_Data["DATE_TIME"], errors="coerce"
    )

    # Convert numeric columns
    Generation_Data["AC_POWER"] = pd.to_numeric(
        Generation_Data["AC_POWER"], errors="coerce"
    )

    Numeric_Columns = [
        "AMBIENT_TEMPERATURE",
        "MODULE_TEMPERATURE",
        "IRRADIATION"
    ]

    for Column in Numeric_Columns:

        Weather_Data[Column] = pd.to_numeric(
            Weather_Data[Column], errors="coerce"
        )

    # Remove invalid timestamps and target values
    Generation_Data.dropna(
        subset=["DATE_TIME", "AC_POWER"], inplace=True
    )

    Weather_Data.dropna(
        subset=["DATE_TIME"], inplace=True
    )

    # Remove duplicate rows
    Generation_Data.drop_duplicates(inplace=True)
    Weather_Data.drop_duplicates(inplace=True)

    # Remove physically invalid negative values
    Generation_Data = Generation_Data[
        Generation_Data["AC_POWER"] >= 0
    ].copy()

    Weather_Data = Weather_Data[
        Weather_Data["IRRADIATION"] >= 0
    ].copy()

    # Fill missing weather values with median
    for Column in Numeric_Columns:

        Median_Value = Weather_Data[Column].median()

        if pd.notna(Median_Value):
            Weather_Data[Column] = (
                Weather_Data[Column].fillna(Median_Value)
            )

    # Remove records with remaining required missing values
    Weather_Data.dropna(
        subset=Numeric_Columns, inplace=True
    )

    if Generation_Data.empty or Weather_Data.empty:
        raise ValueError(
            "No valid records remain after preprocessing."
        )

    print("Generation rows after cleaning:", len(Generation_Data))
    print("Weather rows after cleaning:", len(Weather_Data))

    print(
        "Generation missing values:",
        Generation_Data.isnull().sum().sum()
    )

    print(
        "Weather missing values:",
        Weather_Data.isnull().sum().sum()
    )

    logging.info("Data preprocessing completed")

    return Generation_Data, Weather_Data

# ============================================================
# 7. PREPARE ANALYTICS DATASET
# ============================================================

def PrepareAnalyticsData(Generation_Data, Weather_Data):

    DisplayInfo("STEP 5: PREPARE ANALYTICS DATASET")

    # Sum inverter power readings for each timestamp
    Generation_Aggregated = (
        Generation_Data.groupby("DATE_TIME", as_index=False)
        ["AC_POWER"].sum()
    )

    # Average weather sensor readings at each timestamp
    Weather_Aggregated = (
        Weather_Data.groupby("DATE_TIME", as_index=False)[
            [
                "AMBIENT_TEMPERATURE",
                "MODULE_TEMPERATURE",
                "IRRADIATION"
            ]
        ].mean()
    )

    # Join generation and weather data
    Data = pd.merge(
        Generation_Aggregated,
        Weather_Aggregated,
        on="DATE_TIME",
        how="inner",
        validate="one_to_one"
    )

    Data.sort_values("DATE_TIME", inplace=True)
    Data.reset_index(drop=True, inplace=True)

    if Data.empty:
        raise ValueError(
            "No matching timestamps found in the two datasets."
        )

    # Time-based features
    Data["HOUR"] = Data["DATE_TIME"].dt.hour
    Data["MINUTE"] = Data["DATE_TIME"].dt.minute
    Data["DAY"] = Data["DATE_TIME"].dt.day
    Data["MONTH"] = Data["DATE_TIME"].dt.month
    Data["YEAR"] = Data["DATE_TIME"].dt.year
    Data["DATE"] = Data["DATE_TIME"].dt.date

    print("\nMerged Dataset Shape:", Data.shape)
    print(Data.head())

    SaveCSV(Data, "Processed_Solar_Dataset.csv")

    return Data

# ============================================================
# 8. BUSINESS KPI ANALYSIS
# ============================================================

def BusinessKPIAnalysis(Data):

    DisplayInfo("STEP 6: BUSINESS KPI ANALYSIS")

    KPIs = pd.DataFrame([{
        "Total_Records": len(Data),
        "Average_AC_Power": Data["AC_POWER"].mean(),
        "Maximum_AC_Power": Data["AC_POWER"].max(),
        "Minimum_AC_Power": Data["AC_POWER"].min(),
        "Average_Irradiation": Data["IRRADIATION"].mean(),
        "Average_Module_Temperature":
            Data["MODULE_TEMPERATURE"].mean(),
        "Average_Ambient_Temperature":
            Data["AMBIENT_TEMPERATURE"].mean()
    }])

    print("\nSOLAR BUSINESS KPIs")
    print(KPIs.round(3).to_string(index=False))

    SaveCSV(KPIs, "Solar_Business_KPIs.csv")

    # Daily power summary
    Daily_Report = (
        Data.groupby("DATE", as_index=False)
        .agg(
            Average_AC_Power=("AC_POWER", "mean"),
            Peak_AC_Power=("AC_POWER", "max"),
            Minimum_AC_Power=("AC_POWER", "min"),
            Average_Irradiation=("IRRADIATION", "mean"),
            Average_Module_Temperature=(
                "MODULE_TEMPERATURE", "mean"
            ),
            Records=("AC_POWER", "size")
        )
    )

    Daily_Report["DATE"] = Daily_Report["DATE"].astype(str)

    SaveCSV(Daily_Report, "Daily_Solar_Analysis.csv")

    # Monthly power summary
    Monthly_Report = (
        Data.groupby(["YEAR", "MONTH"], as_index=False)
        .agg(
            Average_AC_Power=("AC_POWER", "mean"),
            Peak_AC_Power=("AC_POWER", "max"),
            Average_Irradiation=("IRRADIATION", "mean"),
            Average_Module_Temperature=(
                "MODULE_TEMPERATURE", "mean"
            ),
            Records=("AC_POWER", "size")
        )
    )

    SaveCSV(Monthly_Report, "Monthly_Solar_Analysis.csv")

    # Hourly analysis
    Hourly_Report = (
        Data.groupby("HOUR", as_index=False)
        .agg(
            Average_AC_Power=("AC_POWER", "mean"),
            Peak_AC_Power=("AC_POWER", "max"),
            Average_Irradiation=("IRRADIATION", "mean")
        )
    )

    SaveCSV(Hourly_Report, "Hourly_Solar_Analysis.csv")

    Peak_Day = Daily_Report.loc[
        Daily_Report["Average_AC_Power"].idxmax()
    ]

    Peak_Hour = Hourly_Report.loc[
        Hourly_Report["Average_AC_Power"].idxmax()
    ]

    print("\nBUSINESS INSIGHTS")

    print(
        "Highest average-power date:",
        Peak_Day["DATE"]
    )

    print(
        "Highest average-power hour:",
        int(Peak_Hour["HOUR"])
    )

    print(
        "Average AC power:",
        round(Data["AC_POWER"].mean(), 3)
    )

    print(
        "Average irradiation:",
        round(Data["IRRADIATION"].mean(), 3)
    )

    # Hourly power chart
    plt.figure(figsize=(10, 5))

    plt.plot(
        Hourly_Report["HOUR"],
        Hourly_Report["Average_AC_Power"],
        marker="o"
    )

    plt.title("Average Solar Power by Hour")
    plt.xlabel("Hour of Day")
    plt.ylabel("Average AC Power")
    plt.xticks(range(0, 24, 2))
    plt.grid(True, alpha=0.3)

    SaveAndShowGraph("Hourly_Solar_Power.png")

    # Daily power chart
    plt.figure(figsize=(12, 5))

    plt.plot(
        Daily_Report["DATE"],
        Daily_Report["Average_AC_Power"],
        marker="o"
    )

    plt.title("Daily Average Solar Power")
    plt.xlabel("Date")
    plt.ylabel("Average AC Power")
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)

    SaveAndShowGraph("Daily_Solar_Power.png")

    # Monthly power chart
    Monthly_Labels = (
        Monthly_Report["YEAR"].astype(str)
        + "-"
        + Monthly_Report["MONTH"].astype(str).str.zfill(2)
    )

    plt.figure(figsize=(10, 5))

    plt.bar(
        Monthly_Labels,
        Monthly_Report["Average_AC_Power"]
    )

    plt.title("Monthly Average Solar Power")
    plt.xlabel("Year-Month")
    plt.ylabel("Average AC Power")
    plt.xticks(rotation=45)

    SaveAndShowGraph("Monthly_Solar_Power.png")

    return Daily_Report, Monthly_Report, Hourly_Report

# ============================================================
# 9. EXPLORATORY DATA ANALYSIS
# ============================================================

def PerformEDA(Data):

    DisplayInfo("STEP 7: EXPLORATORY DATA ANALYSIS")

    # Irradiation vs power
    plt.figure(figsize=(8, 5))

    plt.scatter(
        Data["IRRADIATION"],
        Data["AC_POWER"],
        alpha=0.5
    )

    plt.title("Irradiation vs AC Power")
    plt.xlabel("Irradiation")
    plt.ylabel("AC Power")

    SaveAndShowGraph("Irradiation_vs_Power.png")

    # Correlation matrix
    Correlation = Data[
        [
            "AC_POWER",
            "AMBIENT_TEMPERATURE",
            "MODULE_TEMPERATURE",
            "IRRADIATION"
        ]
    ].corr()

    print("\nCORRELATION MATRIX")
    print(Correlation.round(3))

    SaveCSV(
        Correlation.reset_index(names="Feature"),
        "Correlation_Report.csv"
    )

    plt.figure(figsize=(8, 6))

    Heatmap = plt.imshow(
        Correlation,
        cmap="coolwarm",
        vmin=-1,
        vmax=1
    )

    plt.colorbar(Heatmap)

    plt.xticks(
        range(len(Correlation.columns)),
        Correlation.columns,
        rotation=35,
        ha="right"
    )

    plt.yticks(
        range(len(Correlation.columns)),
        Correlation.columns
    )

    for i in range(len(Correlation.columns)):
        for j in range(len(Correlation.columns)):

            plt.text(
                j, i,
                f"{Correlation.iloc[i, j]:.2f}",
                ha="center",
                va="center"
            )

    plt.title("Solar Power Correlation Heatmap")

    SaveAndShowGraph("Correlation_Heatmap.png")

# ============================================================
# 10. PREPARE ML FEATURES
# ============================================================

def PrepareFeatures(Data):

    DisplayInfo("STEP 8: PREPARE ML FEATURES")

    Feature_Columns = [
        "AMBIENT_TEMPERATURE",
        "MODULE_TEMPERATURE",
        "IRRADIATION",
        "HOUR",
        "MINUTE",
        "DAY",
        "MONTH"
    ]

    X = Data[Feature_Columns].copy()
    Y = Data["AC_POWER"].copy()

    print("Features:", Feature_Columns)
    print("Target: AC_POWER")
    print("Feature Shape:", X.shape)

    return X, Y, Feature_Columns

# ============================================================
# 11. TIME-BASED TRAIN TEST SPLIT
# ============================================================

def SplitData(X, Y):

    DisplayInfo("STEP 9: TRAIN TEST SPLIT")

    Split_Index = int(len(X) * 0.80)

    if Split_Index < 2 or Split_Index >= len(X):
        raise ValueError(
            "Not enough records for train-test split."
        )

    X_Train = X.iloc[:Split_Index]
    X_Test = X.iloc[Split_Index:]

    Y_Train = Y.iloc[:Split_Index]
    Y_Test = Y.iloc[Split_Index:]

    print("Training Records:", len(X_Train))
    print("Testing Records:", len(X_Test))

    return X_Train, X_Test, Y_Train, Y_Test

# ============================================================
# 12. BUILD AND COMPARE MODELS
# ============================================================

def BuildModels():

    return {
        "Linear Regression": LinearRegression(),

        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
    }

def CompareModels(Models, X_Train, Y_Train):

    DisplayInfo("STEP 10: TIME SERIES CROSS VALIDATION")

    if len(X_Train) < 3:
        raise ValueError(
            "Not enough training records for cross-validation."
        )

    Number_Of_Splits = min(5, len(X_Train) - 1)

    Time_Split = TimeSeriesSplit(
        n_splits=Number_Of_Splits
    )

    Results = []

    for Model_Name, Model in Models.items():

        print("\nEvaluating:", Model_Name)

        Scores = cross_val_score(
            Model,
            X_Train,
            Y_Train,
            cv=Time_Split,
            scoring="neg_mean_absolute_error"
        )

        Mean_MAE = -Scores.mean()

        Results.append({
            "Model": Model_Name,
            "CV_MAE": Mean_MAE
        })

        print("Fold MAE:", np.round(-Scores, 3))
        print("Mean CV MAE:", round(Mean_MAE, 3))

    Comparison = pd.DataFrame(Results)

    SaveCSV(Comparison, "Model_Comparison.csv")

    Best_Model_Name = Comparison.loc[
        Comparison["CV_MAE"].idxmin(), "Model"
    ]

    print("\nBest Model:", Best_Model_Name)

    return Comparison, Best_Model_Name

# ============================================================
# 13. TRAIN AND EVALUATE BEST MODEL
# ============================================================

def EvaluateModel(
    Model,
    Model_Name,
    X_Train,
    X_Test,
    Y_Train,
    Y_Test
):

    DisplayInfo("STEP 11: MODEL TRAINING AND EVALUATION")

    Model.fit(X_Train, Y_Train)

    Predictions = Model.predict(X_Test)

    MAE = mean_absolute_error(Y_Test, Predictions)

    RMSE = np.sqrt(
        mean_squared_error(Y_Test, Predictions)
    )

    R2 = r2_score(Y_Test, Predictions)

    Metrics = pd.DataFrame([{
        "Model": Model_Name,
        "MAE": MAE,
        "RMSE": RMSE,
        "R2_Score": R2
    }])

    print("\nSelected Model:", Model_Name)
    print(f"MAE: {MAE:.4f}")
    print(f"RMSE: {RMSE:.4f}")
    print(f"R2 Score: {R2:.4f}")

    SaveCSV(Metrics, "Evaluation_Metrics.csv")

    Results = pd.DataFrame({
        "Actual_Power": np.asarray(Y_Test),
        "Predicted_Power": Predictions
    })

    Results["Error"] = (
        Results["Actual_Power"]
        - Results["Predicted_Power"]
    )

    SaveCSV(Results, "Actual_vs_Predicted.csv")

    # Actual vs predicted line chart
    plt.figure(figsize=(12, 5))

    plt.plot(
        Results["Actual_Power"].to_numpy()[:200],
        label="Actual Power"
    )

    plt.plot(
        Results["Predicted_Power"].to_numpy()[:200],
        label="Predicted Power"
    )

    plt.title("Actual vs Predicted Solar Power")
    plt.xlabel("Test Sample")
    plt.ylabel("AC Power")
    plt.legend()

    SaveAndShowGraph("Actual_vs_Predicted.png")

    return Model, Results


# ============================================================
# 14. SAVE MODEL
# ============================================================

def SaveModel(Model, Model_Name, Feature_Columns):

    DisplayInfo("STEP 12: SAVE TRAINED MODEL")

    Model_Path = OUTPUT_DIR / "Solar_Power_Model.pkl"

    Model_Package = {
        "model": Model,
        "model_name": Model_Name,
        "feature_columns": Feature_Columns
    }

    joblib.dump(Model_Package, Model_Path)

    print("Model saved:", Model_Path)

    return Model_Path

# ============================================================
# 15. EXAMPLE PREDICTION
# ============================================================

def PredictPower(Model_Path):

    DisplayInfo("STEP 13: EXAMPLE SOLAR POWER PREDICTION")

    Model_Package = joblib.load(Model_Path)

    Model = Model_Package["model"]
    Feature_Columns = Model_Package["feature_columns"]

    New_Data = pd.DataFrame([{
        "AMBIENT_TEMPERATURE": 25.0,
        "MODULE_TEMPERATURE": 30.0,
        "IRRADIATION": 0.7,
        "HOUR": 12,
        "MINUTE": 0,
        "DAY": 15,
        "MONTH": 6
    }])

    New_Data = New_Data[Feature_Columns]

    Prediction = float(Model.predict(New_Data)[0])

    print("\nExample Input:")
    print(New_Data.to_string(index=False))

    print("\nPredicted AC Power:", round(Prediction, 3))

    print(
        "Note: This is an example prediction, "
        "not a guaranteed real-world output."
    )

# ============================================================
# 16. MAIN FUNCTION
# ============================================================

def main():

    try:

        DisplayInfo("SOLAR POWER GENERATION ANALYSIS STARTED")

        # Load datasets
        Generation_Data, Weather_Data = LoadDataset()

        # Data quality report
        DataQualityReport(
            Generation_Data,
            Weather_Data
        )

        # Understand datasets
        UnderstandDataset(
            Generation_Data,
            Weather_Data
        )

        # Clean datasets
        Generation_Data, Weather_Data = PreprocessData(
            Generation_Data,
            Weather_Data
        )

        # Prepare joined dataset
        Data = PrepareAnalyticsData(
            Generation_Data,
            Weather_Data
        )

        if len(Data) < 10:
            raise ValueError(
                "Too few matching timestamps for analysis."
            )

        # Business analytics
        BusinessKPIAnalysis(Data)

        # EDA
        PerformEDA(Data)

        # Prepare ML data
        X, Y, Feature_Columns = PrepareFeatures(Data)

        X_Train, X_Test, Y_Train, Y_Test = SplitData(X, Y)

        # Compare models with time-series CV
        Models = BuildModels()

        Comparison, Best_Model_Name = CompareModels(
            Models,
            X_Train,
            Y_Train
        )

        Best_Model = Models[Best_Model_Name]

        # Train best model and evaluate test performance
        Trained_Model, Results = EvaluateModel(
            Best_Model,
            Best_Model_Name,
            X_Train,
            X_Test,
            Y_Train,
            Y_Test
        )

        # Save model
        Model_Path = SaveModel(
            Trained_Model,
            Best_Model_Name,
            Feature_Columns
        )

        # Example prediction
        PredictPower(Model_Path)

        DisplayInfo("PROJECT COMPLETED SUCCESSFULLY")

        print("\nImages Folder:", IMAGE_DIR)
        print("Outputs Folder:", OUTPUT_DIR)

    except Exception as Error:

        logging.exception("Project execution failed")

        print("\nERROR:", Error)
        print(
            "Check CSV paths, column names, "
            "and outputs/project.log."
        )

if __name__ == "__main__":
    main()