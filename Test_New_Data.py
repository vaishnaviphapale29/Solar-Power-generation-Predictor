import pandas as pd
import joblib
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "outputs" / "Solar_Power_Model.pkl"

def TestNewData():

    # Load trained model
    Saved_Data = joblib.load(MODEL_PATH)

    Model = Saved_Data["model"]
    Feature_Columns = Saved_Data["feature_columns"]

    # Enter new weather conditions
    New_Data = pd.DataFrame([
        {
            "AMBIENT_TEMPERATURE": 25,
            "MODULE_TEMPERATURE": 30,
            "IRRADIATION": 0.7,
            "HOUR": 12,
            "MINUTE": 0,
            "DAY": 15,
            "MONTH": 6
        }
    ])

    # Keep features in the expected order
    New_Data = New_Data[Feature_Columns]

    # Predict solar power
    Prediction = Model.predict(New_Data)

    print("\nNew Weather Conditions:")
    print(New_Data.to_string(index=False))

    print("\nPredicted Solar Power Generation:")
    print(f"{Prediction[0]:.2f}")

if __name__ == "__main__":
    TestNewData()