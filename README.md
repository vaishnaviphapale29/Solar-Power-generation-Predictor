# Solar Power Generation Predictor

## Project Overview

Solar Power Generation Predictor is a Machine Learning project that predicts solar power generation based on weather conditions.

The project uses historical solar power generation data and weather sensor data to analyze the relationship between environmental conditions and AC power output.

The main goal is to build, evaluate, and save Machine Learning regression models that can predict solar power generation using weather-related and time-based features.

## Objectives

- Analyze solar power generation data.
- Perform data cleaning and preprocessing.
- Conduct Exploratory Data Analysis (EDA).
- Visualize relationships between weather conditions and power generation.
- Perform feature engineering.
- Train and evaluate regression models.
- Compare model performance.
- Save the trained model for future predictions.
- Test the model using new weather data.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Git
- GitHub
- Visual Studio Code

## Machine Learning Algorithms

The project uses the following regression algorithms:

1. **Linear Regression**
2. **Random Forest Regressor**

The models are trained on the prepared dataset and evaluated to compare their prediction performance.

## Dataset Description

This project uses two datasets.

### 1. Plant 1 Generation Data

File: `Plant_1_Generation_Data.csv`

This dataset contains historical solar power generation information, including AC power, DC power, and related generation measurements.

### 2. Plant 1 Weather Sensor Data

File: `Plant_1_Weather_Sensor_Data.csv`

This dataset contains weather measurements, including ambient temperature, module temperature, and irradiation.

Both datasets are used to study how weather conditions relate to solar power generation.

## Project Structure

```text
Solar_Power_Generation/
│
├── Data/
│   ├── Plant_1_Generation_Data.csv
│   └── Plant_1_Weather_Sensor_Data.csv
│
├── images/
│   └── EDA visualization images
│
├── outputs/
│   └── Trained model and generated output files
│
├── Solar_Power_Generation_Predictor.py
├── Test_New_Data.py
├── requirements.txt
├── project.log
└── README.md
```

## Project Workflow

1. Load the generation and weather datasets.
2. Inspect the dataset structure, columns, and data types.
3. Check and handle missing or inconsistent values.
4. Prepare and combine the datasets using suitable keys.
5. Perform Exploratory Data Analysis.
6. Create visualizations to understand data patterns.
7. Extract time-based features.
8. Select input features and the target variable.
9. Split the data into training and testing sets.
10. Train Linear Regression and Random Forest Regression models.
11. Evaluate the models using regression metrics.
12. Compare model performance.
13. Save the trained model.
14. Load the saved model to test new weather conditions.
15. Display the predicted AC power generation.

## Input Features

The model uses the following features:

- `AMBIENT_TEMPERATURE` — Ambient air temperature.
- `MODULE_TEMPERATURE` — Solar panel module temperature.
- `IRRADIATION` — Solar radiation received by the panel.
- `HOUR` — Hour of the day.
- `MINUTE` — Minute of the hour.
- `DAY` — Day of the month.
- `MONTH` — Month of the year.

## Target Variable

`AC_POWER`

The target variable represents the AC power output that the model is trained to predict.

## Model Performance

Two Machine Learning models were trained and compared using Time Series Cross-Validation.

| Model | Cross-Validation MAE |
|---|---:|
| Linear Regression | 514.2837 |
| Random Forest | **333.0444** |

**Best Model:** Random Forest Regressor

### Evaluation Results

The Random Forest model was evaluated on the test dataset.

| Metric | Score |
|---|---:|
| Mean Absolute Error (MAE) | 249.1383 |
| Root Mean Squared Error (RMSE) | 526.4474 |
| R² Score | 0.9956 |

### Dataset Summary

- **Generation Dataset Records:** 68,778
- **Weather Dataset Records:** 3,182
- **Processed Dataset Records:** 3,157
- **Training Records:** 2,525
- **Testing Records:** 632
- **Input Features:** 7
- **Target Variable:** AC_POWER

### Key Findings

- Random Forest performed better than Linear Regression based on cross-validation MAE.
- The model achieved an R² score of 0.9956 on the test dataset.
- Irradiation showed a strong correlation with AC power in the processed dataset.
- The model generated actual-versus-predicted comparisons and prediction visualizations.

### Sample Prediction

**Input:**
- Ambient Temperature: 25°C
- Module Temperature: 30°C
- Irradiation: 0.7
- Hour: 12
- Minute: 0
- Day: 15
- Month: 6

**Predicted AC Power:** 19,190.302

*Note: This is an example prediction using sample input values.*

### Generated Outputs

- `outputs/Solar_Power_Model.pkl` — Trained Random Forest model
- `outputs/Processed_Solar_Dataset.csv` — Processed dataset
- `outputs/Actual_vs_Predicted.csv` — Actual and predicted power values
- `outputs/Model_Comparison.csv` — Model comparison results
- `outputs/Evaluation_Metrics.csv` — Evaluation metrics
- `images/` — Generated charts and visualizations

### Future Improvements

- Experiment with additional regression models and hyperparameter tuning.
- Evaluate the model on data from different solar plants and time periods.
- Develop a dashboard for solar power monitoring and prediction.
- Integrate live weather data for practical power forecasting.

## Installation and Setup

### Prerequisites

- Python 3.10 or later
- pip
- Visual Studio Code or another Python IDE

### Install Required Libraries

Open the project folder in VS Code and run the following command in the terminal:

```bash
pip install -r requirements.txt
```

## Run the Main Project

Run the following command to execute the main script:

```bash
python Solar_Power_Generation_Predictor.py
```

The script performs the configured data processing, analysis, model training, evaluation, and model-saving steps.

Make sure both dataset files are available inside the `Data/` folder.

## Test Prediction on New Data

After the main script has successfully trained and saved the model, run:

```bash
python Test_New_Data.py
```

This script loads the saved model from the `outputs/` folder and predicts AC power generation using sample weather conditions.

You can modify the weather values inside `Test_New_Data.py` to test different input conditions.

**Note:** Predictions depend on the quality of the dataset, the trained model, and the selected input values.

## Model Evaluation

Regression models can be evaluated using the following metrics:

- **Mean Absolute Error (MAE):** Measures the average absolute difference between actual and predicted values.
- **Mean Squared Error (MSE):** Measures the average squared prediction error.
- **Root Mean Squared Error (RMSE):** Measures prediction error in the target variable's units.
- **R-squared (R²):** Measures how well the model explains variation in the target variable.

The actual evaluation results depend on the trained model and the dataset.

## Project Outputs

The project can generate:

- Exploratory Data Analysis visualizations.
- Model evaluation results.
- Prediction and analysis output files.
- A saved trained Machine Learning model.
- Execution logs.

Generated files are stored in the appropriate project folders.

## Applications

This project demonstrates how Machine Learning can be used to:

- Analyze the relationship between weather conditions and solar power output.
- Estimate solar power generation from selected weather inputs.
- Compare regression algorithms for a prediction task.
- Build a reusable model for experimentation and learning.

The project is intended for educational and portfolio purposes. It is not a guarantee of real-world solar power output.

## Future Improvements

- Experiment with additional regression algorithms.
- Perform systematic hyperparameter tuning.
- Improve feature engineering.
- Develop an interactive prediction interface using Streamlit.
- Deploy the model as a web application.
- Evaluate the model on data from additional solar plants.
- Monitor model performance on new data.