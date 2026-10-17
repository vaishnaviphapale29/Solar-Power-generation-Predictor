import logging
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "outputs" / "Solar_Power_Model.pkl"
OUTPUT_DIR = BASE_DIR / "outputs"
IMAGE_DIR = BASE_DIR / "images"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

PREDICTION_PATH = OUTPUT_DIR / "New_Data_Prediction.csv"
GRAPH_PATH = IMAGE_DIR / "New_Data_Prediction.png"

logging.basicConfig(
    filename=str(OUTPUT_DIR / "test_new_data.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8"
)

# ============================================================
# 2. GET NEW WEATHER DATA
# ============================================================

def GetNewData():

    print("\nENTER WEATHER CONDITIONS")
    print("-" * 40)

    Values = {
        "AMBIENT_TEMPERATURE": 25.0,
        "MODULE_TEMPERATURE": 30.0,
        "IRRADIATION": 0.7,
        "HOUR": 12,
        "MINUTE": 0,
        "DAY": 15,
        "MONTH": 6
    }

    for Feature, Default in Values.items():

        User_Input = input(
            f"{Feature} (default {Default}): "
        ).strip()

        if User_Input:
            Values[Feature] = float(User_Input)

    if not 0 <= Values["IRRADIATION"] <= 1.5:
        raise ValueError("Irradiation must be between 0 and 1.5.")

    if not 0 <= Values["HOUR"] <= 23:
        raise ValueError("Hour must be between 0 and 23.")

    if Values["HOUR"] % 1 != 0:
        raise ValueError("Hour must be a whole number.")

    if not 0 <= Values["MINUTE"] <= 59:
        raise ValueError("Minute must be between 0 and 59.")

    if Values["MINUTE"] % 1 != 0:
        raise ValueError("Minute must be a whole number.")

    if not 1 <= Values["DAY"] <= 31:
        raise ValueError("Day must be between 1 and 31.")

    if Values["DAY"] % 1 != 0:
        raise ValueError("Day must be a whole number.")

    if not 1 <= Values["MONTH"] <= 12:
        raise ValueError("Month must be between 1 and 12.")

    if Values["MONTH"] % 1 != 0:
        raise ValueError("Month must be a whole number.")

    if Values["AMBIENT_TEMPERATURE"] < 0:
        raise ValueError("Ambient temperature cannot be negative.")

    if Values["MODULE_TEMPERATURE"] < 0:
        raise ValueError("Module temperature cannot be negative.")

    return pd.DataFrame([Values])

# ============================================================
# 3. LOAD MODEL AND PREDICT
# ============================================================

def TestNewData():

    print("\n" + "=" * 60)
    print("SOLAR POWER GENERATION - NEW DATA PREDICTION")
    print("=" * 60)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}\n"
            "Run the main project script first."
        )

    Saved_Data = joblib.load(MODEL_PATH)

    Model = Saved_Data["model"]
    Feature_Columns = Saved_Data["feature_columns"]

    New_Data = GetNewData()
    New_Data = New_Data[Feature_Columns]

    Prediction = Model.predict(New_Data)
    Predicted_Power = float(Prediction[0])

    print("\nINPUT WEATHER CONDITIONS")
    print(New_Data.to_string(index=False))

    print("\nPREDICTION RESULT")
    print(f"Predicted AC Power: {Predicted_Power:.2f}")

    logging.info(
        "New data prediction completed: %.2f",
        Predicted_Power
    )

    # Save prediction report
    Result = New_Data.copy()
    Result["PREDICTED_AC_POWER"] = Predicted_Power
    Result.to_csv(PREDICTION_PATH, index=False)

    print("\nPrediction report saved:", PREDICTION_PATH)

    # Plot prediction
    plt.figure(figsize=(7, 5))
    plt.bar(
        ["Predicted AC Power"],
        [Predicted_Power]
    )
    plt.title("Solar Power Generation Prediction")
    plt.ylabel("Predicted AC Power")
    plt.tight_layout()
    plt.savefig(GRAPH_PATH, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    print("Prediction graph saved:", GRAPH_PATH)
    print("\nTEST COMPLETED SUCCESSFULLY")

if __name__ == "__main__":

    try:
        TestNewData()

    except (ValueError, FileNotFoundError, KeyError) as Error:
        print("\nERROR:", Error)
        logging.exception("Prediction failed")
