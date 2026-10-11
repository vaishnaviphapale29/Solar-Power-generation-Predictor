import logging
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================================
# PROJECT PATHS - USING PATHLIB ONLY
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
IMAGE_DIR = BASE_DIR / "images"
OUTPUT_DIR = BASE_DIR / "outputs"

DATA_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GENERATION_PATH = DATA_DIR / "Plant_1_Generation_Data.csv"
WEATHER_PATH = DATA_DIR / "Plant_1_Weather_Sensor_Data.csv"

LOG_PATH = OUTPUT_DIR / "project.log"

# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    filename=str(LOG_PATH),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8"
)

def DisplayInfo(Message):

    print("\n" + "=" * 70)
    print(Message)
    print("=" * 70)

    logging.info(Message)

# ============================================================
# COMMON GRAPH HANDLER
# SAVE GRAPH AND DISPLAY ON SCREEN
# ============================================================

def SaveAndShowGraph(FileName):

    Graph_Path = IMAGE_DIR / FileName

    plt.tight_layout()

    plt.savefig(
        Graph_Path,
        dpi=300,
        bbox_inches="tight"
    )

    print("Graph saved:", Graph_Path)
    logging.info("Graph saved: %s", Graph_Path)

    plt.show()
    plt.close()

# ============================================================
# STEP 1: LOAD DATASET
# ============================================================

def LoadDataset():

    DisplayInfo("STEP 1: LOAD DATASET")

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

    print("Generation Dataset Shape:", Generation_Data.shape)
    print("Weather Dataset Shape:", Weather_Data.shape)

    print("\nGeneration Dataset:")
    print(Generation_Data.head())

    print("\nWeather Dataset:")
    print(Weather_Data.head())

    logging.info("Datasets loaded successfully")

    return Generation_Data, Weather_Data

# ============================================================
# STEP 2: UNDERSTAND DATASET
# ============================================================

def UnderstandDataset(Generation_Data, Weather_Data):

    DisplayInfo("STEP 2: UNDERSTAND DATASET")

    print("\nGeneration Dataset Information:")
    Generation_Data.info()

    print("\nWeather Dataset Information:")
    Weather_Data.info()

    print("\nGeneration Columns:")
    print(Generation_Data.columns.tolist())

    print("\nWeather Columns:")
    print(Weather_Data.columns.tolist())

    print("\nGeneration Missing Values:")
    print(Generation_Data.isnull().sum())

    print("\nWeather Missing Values:")
    print(Weather_Data.isnull().sum())

    print("\nGeneration Duplicate Rows:",
          Generation_Data.duplicated().sum())

    print("Weather Duplicate Rows:",
          Weather_Data.duplicated().sum())

    print("\nGeneration Statistical Summary:")
    print(Generation_Data.describe())

    print("\nWeather Statistical Summary:")
    print(Weather_Data.describe())

# ============================================================
# STEP 3: PREPROCESS DATA
# ============================================================

def PreprocessData(Generation_Data, Weather_Data):

    DisplayInfo("STEP 3: DATA PREPROCESSING")

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

    Generation_Data["DATE_TIME"] = pd.to_datetime(
        Generation_Data["DATE_TIME"],
        errors="coerce"
    )

    Weather_Data["DATE_TIME"] = pd.to_datetime(
        Weather_Data["DATE_TIME"],
        errors="coerce"
    )

    Generation_Data["AC_POWER"] = pd.to_numeric(
        Generation_Data["AC_POWER"],
        errors="coerce"
    )

    Numeric_Columns = [
        "AMBIENT_TEMPERATURE",
        "MODULE_TEMPERATURE",
        "IRRADIATION"
    ]

    for Column in Numeric_Columns:

        Weather_Data[Column] = pd.to_numeric(
            Weather_Data[Column],
            errors="coerce"
        )

    Generation_Data.dropna(
        subset=["DATE_TIME", "AC_POWER"],
        inplace=True
    )

    Weather_Data.dropna(
        subset=["DATE_TIME"],
        inplace=True
    )

    Generation_Data.drop_duplicates(inplace=True)
    Weather_Data.drop_duplicates(inplace=True)

    Generation_Data = Generation_Data[
        Generation_Data["AC_POWER"] >= 0
    ]

    Weather_Data = Weather_Data[
        Weather_Data["IRRADIATION"] >= 0
    ]

    for Column in Numeric_Columns:

        Weather_Data[Column] = Weather_Data[Column].fillna(
            Weather_Data[Column].median()
        )

    print(
        "\nGeneration Shape After Cleaning:",
        Generation_Data.shape
    )

    print(
        "Weather Shape After Cleaning:",
        Weather_Data.shape
    )

    print(
        "\nRemaining Generation Missing Values:",
        Generation_Data.isnull().sum().sum()
    )

    print(
        "Remaining Weather Missing Values:",
        Weather_Data.isnull().sum().sum()
    )

    logging.info("Data preprocessing completed")

    return Generation_Data, Weather_Data

# ============================================================
# STEP 4: EXPLORATORY DATA ANALYSIS
# ============================================================

def PerformEDA(Generation_Data, Weather_Data):

    DisplayInfo("STEP 4: EXPLORATORY DATA ANALYSIS")

    # Graph 1: Power Distribution

    plt.figure(figsize=(8, 5))

    plt.hist(
        Generation_Data["AC_POWER"],
        bins=30,
        edgecolor="black"
    )

    plt.title("AC Power Distribution")
    plt.xlabel("AC Power")
    plt.ylabel("Frequency")

    SaveAndShowGraph("Power_Distribution.png")

    # Graph 2: Irradiation Distribution

    plt.figure(figsize=(8, 5))

    plt.hist(
        Weather_Data["IRRADIATION"],
        bins=30,
        edgecolor="black"
    )

    plt.title("Solar Irradiation Distribution")
    plt.xlabel("Irradiation")
    plt.ylabel("Frequency")

    SaveAndShowGraph("Irradiation_Distribution.png")

    # Graph 3: Irradiation vs Module Temperature

    plt.figure(figsize=(8, 5))

    plt.scatter(
        Weather_Data["IRRADIATION"],
        Weather_Data["MODULE_TEMPERATURE"],
        alpha=0.5
    )

    plt.title("Irradiation vs Module Temperature")
    plt.xlabel("Irradiation")
    plt.ylabel("Module Temperature")

    SaveAndShowGraph("Irradiation_vs_Temperature.png")

    # Prepare timestamp-aligned data for correlation

    Generation_Aggregated = (
        Generation_Data.groupby("DATE_TIME", as_index=False)
        ["AC_POWER"].sum()
    )

    Weather_Aggregated = (
        Weather_Data.groupby("DATE_TIME", as_index=False)[
            [
                "AMBIENT_TEMPERATURE",
                "MODULE_TEMPERATURE",
                "IRRADIATION"
            ]
        ].mean()
    )

    EDA_Data = pd.merge(
        Generation_Aggregated,
        Weather_Aggregated,
        on="DATE_TIME",
        how="inner"
    )

    Correlation = EDA_Data[
        [
            "AC_POWER",
            "AMBIENT_TEMPERATURE",
            "MODULE_TEMPERATURE",
            "IRRADIATION"
        ]
    ].corr()

    print("\nCorrelation Matrix:")
    print(Correlation.round(3))

    # Graph 4: Correlation Heatmap

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
                j,
                i,
                f"{Correlation.iloc[i, j]:.2f}",
                ha="center",
                va="center"
            )

    plt.title("Solar Power Correlation Heatmap")

    SaveAndShowGraph("Correlation_Heatmap.png")

# ============================================================
# STEP 5: FEATURE ENGINEERING
# ============================================================

def FeatureEngineering(Generation_Data, Weather_Data):

    DisplayInfo("STEP 5: FEATURE ENGINEERING")

    Generation_Aggregated = (
        Generation_Data.groupby("DATE_TIME", as_index=False)
        ["AC_POWER"].sum()
    )

    Weather_Aggregated = (
        Weather_Data.groupby("DATE_TIME", as_index=False)[
            [
                "AMBIENT_TEMPERATURE",
                "MODULE_TEMPERATURE",
                "IRRADIATION"
            ]
        ].mean()
    )

    Data = pd.merge(
        Generation_Aggregated,
        Weather_Aggregated,
        on="DATE_TIME",
        how="inner"
    )

    Data.sort_values("DATE_TIME", inplace=True)
    Data.reset_index(drop=True, inplace=True)

    Data["HOUR"] = Data["DATE_TIME"].dt.hour
    Data["MINUTE"] = Data["DATE_TIME"].dt.minute
    Data["DAY"] = Data["DATE_TIME"].dt.day
    Data["MONTH"] = Data["DATE_TIME"].dt.month

    Data.dropna(inplace=True)

    Processed_Path = OUTPUT_DIR / "Processed_Solar_Dataset.csv"

    Data.to_csv(Processed_Path, index=False)

    print("\nProcessed Dataset:")
    print(Data.head(10).to_string(index=False))

    print("\nProcessed Dataset Shape:", Data.shape)
    print("Processed Dataset saved:", Processed_Path)

    logging.info("Feature engineering completed")

    return Data

# ============================================================
# STEP 6: PREPARE FEATURES AND TARGET
# ============================================================

def PrepareData(Data):

    DisplayInfo("STEP 6: PREPARE FEATURES AND TARGET")

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

    print("Feature Columns:", Feature_Columns)
    print("Target Column: AC_POWER")
    print("X Shape:", X.shape)
    print("Y Shape:", Y.shape)

    return X, Y, Feature_Columns

# ============================================================
# STEP 7: TRAIN-TEST SPLIT
# ============================================================

def SplitData(X, Y):

    DisplayInfo("STEP 7: TRAIN-TEST SPLIT")

    Split_Index = int(len(X) * 0.80)

    if Split_Index <= 0 or Split_Index >= len(X):

        raise ValueError(
            "Not enough rows for train-test split."
        )

    X_Train = X.iloc[:Split_Index]
    X_Test = X.iloc[Split_Index:]

    Y_Train = Y.iloc[:Split_Index]
    Y_Test = Y.iloc[Split_Index:]

    print("Training Records:", len(X_Train))
    print("Testing Records:", len(X_Test))

    return X_Train, X_Test, Y_Train, Y_Test

# ============================================================
# STEP 8: BUILD MODELS
# ============================================================

def BuildModels():

    DisplayInfo("STEP 8: BUILD MACHINE LEARNING MODELS")

    Models = {
        "Linear Regression": LinearRegression(),

        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
    }

    print("Models Created:")

    for Model_Name in Models:
        print("-", Model_Name)

    return Models

# ============================================================
# STEP 9: TRAIN MODELS
# ============================================================

def TrainModels(Models, X_Train, Y_Train):

    DisplayInfo("STEP 9: TRAIN MODELS")

    Trained_Models = {}

    for Model_Name, Model in Models.items():

        print(f"\nTraining {Model_Name}...")

        Model.fit(X_Train, Y_Train)

        Trained_Models[Model_Name] = Model

        print(f"{Model_Name} training completed.")

        logging.info("%s training completed", Model_Name)

    return Trained_Models

# ============================================================
# STEP 10: CROSS VALIDATION
# ============================================================

def CrossValidateModels(Models, X_Train, Y_Train):

    DisplayInfo("STEP 10: TIME SERIES CROSS VALIDATION")

    if len(X_Train) < 3:

        raise ValueError(
            "Not enough training rows for cross-validation."
        )

    Number_Of_Splits = min(5, len(X_Train) - 1)

    Time_Split = TimeSeriesSplit(
        n_splits=Number_Of_Splits
    )

    CV_Results = {}

    for Model_Name, Model in Models.items():

        print(f"\nCross-validating {Model_Name}...")

        Scores = cross_val_score(
            Model,
            X_Train,
            Y_Train,
            cv=Time_Split,
            scoring="neg_mean_absolute_error"
        )

        Mean_MAE = -Scores.mean()

        CV_Results[Model_Name] = Mean_MAE

        print("Fold MAE:", np.round(-Scores, 3))
        print("Mean CV MAE:", round(Mean_MAE, 3))

    Best_Model_Name = min(
        CV_Results,
        key=CV_Results.get
    )

    print("\nBest Model Based on CV MAE:", Best_Model_Name)

    return CV_Results, Best_Model_Name

# ============================================================
# STEP 11: EVALUATE MODEL
# ============================================================

def EvaluateModel(Model, X_Test, Y_Test, Model_Name):

    DisplayInfo("STEP 11: MODEL EVALUATION")

    Predictions = Model.predict(X_Test)

    MAE = mean_absolute_error(Y_Test, Predictions)

    RMSE = np.sqrt(
        mean_squared_error(Y_Test, Predictions)
    )

    R2 = r2_score(Y_Test, Predictions)

    Metrics = {
        "Model": Model_Name,
        "MAE": MAE,
        "RMSE": RMSE,
        "R2_Score": R2
    }

    print("Selected Model:", Model_Name)
    print(f"MAE: {MAE:.4f}")
    print(f"RMSE: {RMSE:.4f}")
    print(f"R2 Score: {R2:.4f}")

    return Predictions, Metrics

# ============================================================
# STEP 12: SAVE OUTPUT FILES AND DISPLAY RESULTS
# ============================================================

def SaveResults(Y_Test, Predictions, CV_Results, Metrics):

    DisplayInfo("STEP 12: SAVE AND DISPLAY OUTPUTS")

    Results = pd.DataFrame({
        "Actual_Power": np.asarray(Y_Test),
        "Predicted_Power": Predictions
    })

    Results["Error"] = (
        Results["Actual_Power"] -
        Results["Predicted_Power"]
    )

    Comparison = pd.DataFrame(
        list(CV_Results.items()),
        columns=["Model", "CV_MAE"]
    )

    Metrics_Data = pd.DataFrame([Metrics])

    Results_Path = OUTPUT_DIR / "Actual_vs_Predicted.csv"
    Comparison_Path = OUTPUT_DIR / "Model_Comparison.csv"
    Metrics_Path = OUTPUT_DIR / "Evaluation_Metrics.csv"

    Results.to_csv(Results_Path, index=False)
    Comparison.to_csv(Comparison_Path, index=False)
    Metrics_Data.to_csv(Metrics_Path, index=False)

    print("\nACTUAL VS PREDICTED POWER (FIRST 10 ROWS)")
    print(Results.head(10).round(3).to_string(index=False))

    print("\nMODEL COMPARISON")
    print(Comparison.round(4).to_string(index=False))

    print("\nEVALUATION METRICS")
    print(Metrics_Data.round(4).to_string(index=False))

    print("\nOutput Files Saved:")
    print(Results_Path)
    print(Comparison_Path)
    print(Metrics_Path)

    logging.info("Output CSV files saved successfully")

    return Results

# ============================================================
# STEP 13: VISUALIZE PREDICTIONS
# SAVE AND SHOW EVERY GRAPH
# ============================================================

def VisualizePredictions(Results):

    DisplayInfo("STEP 13: VISUALIZE PREDICTIONS")

    # Graph 5: Actual vs Predicted Line Graph

    plt.figure(figsize=(12, 6))

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

    # Graph 6: Actual vs Predicted Scatter Plot

    plt.figure(figsize=(7, 6))

    plt.scatter(
        Results["Actual_Power"],
        Results["Predicted_Power"],
        alpha=0.6
    )

    Maximum = max(
        Results["Actual_Power"].max(),
        Results["Predicted_Power"].max()
    )

    plt.plot(
        [0, Maximum],
        [0, Maximum],
        linestyle="--"
    )

    plt.title("Actual vs Predicted Power Scatter Plot")
    plt.xlabel("Actual Power")
    plt.ylabel("Predicted Power")

    SaveAndShowGraph("Actual_Predicted_Scatter.png")

    # Graph 7: Prediction Error Distribution

    plt.figure(figsize=(8, 5))

    plt.hist(
        Results["Error"],
        bins=30,
        edgecolor="black"
    )

    plt.title("Prediction Error Distribution")
    plt.xlabel("Actual Power - Predicted Power")
    plt.ylabel("Frequency")

    SaveAndShowGraph("Prediction_Error_Distribution.png")

# ============================================================
# STEP 14: SAVE TRAINED MODEL
# ============================================================

def SaveModel(Model, Model_Name, Feature_Columns):

    DisplayInfo("STEP 14: SAVE TRAINED MODEL")

    Model_Path = OUTPUT_DIR / "Solar_Power_Model.pkl"

    Model_Package = {
        "model": Model,
        "model_name": Model_Name,
        "feature_columns": Feature_Columns
    }

    joblib.dump(Model_Package, Model_Path)

    print("Trained Model Saved:", Model_Path)

    logging.info("Trained model saved successfully")

    return Model_Path

# ============================================================
# STEP 15: PREDICT SOLAR POWER
# ============================================================

def PredictPower(Model_Path):

    DisplayInfo("STEP 15: SOLAR POWER PREDICTION")

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

    Predicted_Power = Model.predict(New_Data)[0]

    print("\nExample Input:")
    print(New_Data.to_string(index=False))

    print(
        "\nPredicted AC Power:",
        round(float(Predicted_Power), 3)
    )

    print(
        "\nNote: This is an example prediction "
        "using sample input values."
    )

    logging.info("Example prediction completed")

    return Predicted_Power

# ============================================================
# MAIN FUNCTION
# ============================================================

def main():

    try:

        DisplayInfo("SOLAR POWER GENERATION PREDICTOR STARTED")

        # Step 1
        Generation_Data, Weather_Data = LoadDataset()

        # Step 2
        UnderstandDataset(
            Generation_Data,
            Weather_Data
        )

        # Step 3
        Generation_Data, Weather_Data = PreprocessData(
            Generation_Data,
            Weather_Data
        )

        # Step 4
        PerformEDA(
            Generation_Data,
            Weather_Data
        )

        # Step 5
        Data = FeatureEngineering(
            Generation_Data,
            Weather_Data
        )

        if len(Data) < 10:

            raise ValueError(
                "Too few merged records. Check DATE_TIME values "
                "in both datasets."
            )

        # Step 6
        X, Y, Feature_Columns = PrepareData(Data)

        # Step 7
        X_Train, X_Test, Y_Train, Y_Test = SplitData(X, Y)

        # Step 8
        Models = BuildModels()

        # Step 9
        Trained_Models = TrainModels(
            Models,
            X_Train,
            Y_Train
        )

        # Step 10
        CV_Results, Best_Model_Name = CrossValidateModels(
            Trained_Models,
            X_Train,
            Y_Train
        )

        # Step 11
        Best_Model = Trained_Models[Best_Model_Name]

        Predictions, Metrics = EvaluateModel(
            Best_Model,
            X_Test,
            Y_Test,
            Best_Model_Name
        )

        # Step 12
        Results = SaveResults(
            Y_Test,
            Predictions,
            CV_Results,
            Metrics
        )

        # Step 13
        VisualizePredictions(Results)

        # Step 14
        Model_Path = SaveModel(
            Best_Model,
            Best_Model_Name,
            Feature_Columns
        )

        # Step 15
        PredictPower(Model_Path)

        DisplayInfo("PROJECT EXECUTED SUCCESSFULLY")

        print("\nImages Folder:", IMAGE_DIR)
        print("Outputs Folder:", OUTPUT_DIR)

    except Exception as Error:

        logging.exception("Project execution failed")

        print("\nERROR:", Error)

        print(
            "\nCheck the dataset paths, CSV column names, "
            "and outputs/project.log."
        )

if __name__ == "__main__":
    main()
