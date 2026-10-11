# Solar Power Generation Predictor

## Overview
Solar Power Generation Predictor is a Machine Learning project that predicts solar power generation using historical generation data and weather conditions. It applies data preprocessing, feature engineering, model training, and evaluation to develop a predictive solution.

## Objectives
- Predict solar power generation using weather-related features.
- Analyze relationships between weather conditions and power output.
- Compare different regression algorithms.
- Evaluate model performance using appropriate regression metrics.

## Technologies Used
- **Programming Language:** Python
- **Data Processing:** Pandas, NumPy
- **Data Visualization:** Matplotlib
- **Machine Learning:** Scikit-learn
- **Model Persistence:** Joblib
- **Path Management:** Pathlib
- **Logging:** Python Logging

## Machine Learning Algorithms

### 1. Linear Regression
Used as a baseline regression model to predict solar power generation.

### 2. Random Forest Regressor
An **Ensemble Learning algorithm** that combines predictions from multiple decision trees to improve predictive performance and capture nonlinear relationships in the data.

Both models are trained and evaluated to compare their performance.

## Project Workflow
1. Load generation and weather datasets.
2. Inspect and preprocess the data.
3. Perform Exploratory Data Analysis (EDA).
4. Engineer features and combine relevant datasets.
5. Split the data chronologically into training and testing sets.
6. Train regression models.
7. Perform time-series cross-validation.
8. Evaluate models using MAE, RMSE, and R² score.
9. Compare actual and predicted power generation.
10. Save visualizations, evaluation results, and the trained model.

## Evaluation Metrics
- **MAE (Mean Absolute Error):** Measures the average absolute prediction error.
- **RMSE (Root Mean Squared Error):** Penalizes larger prediction errors more strongly.
- **R² Score:** Measures how well the model explains the variation in the target variable.

## Project Structure
```text
Solar_Power_Generation_Predictor/
├── data/
│   ├── Plant_1_Generation_Data.csv
│   └── Plant_1_Weather_Sensor_Data.csv
├── images/
├── outputs/
├── Solar_Power_Generation_Predictor.py
├── requirements.txt
└── README.md
```

## Installation

Install the required dependencies:

```bash
python -m pip install -r requirements.txt
```

## Dataset Setup
Place the following CSV files inside the `data/` directory:
- `Plant_1_Generation_Data.csv`
- `Plant_1_Weather_Sensor_Data.csv`

## Execution

Run the project using:

```bash
python Solar_Power_Generation_Predictor.py
```

## Outputs
The project generates:
- Processed solar power dataset
- Actual vs. predicted results
- Model comparison and evaluation metrics
- Saved trained model
- Data visualization graphs
- Execution logs

## Key Highlights
- Implements an Ensemble Learning algorithm using Random Forest Regressor.
- Compares baseline and ensemble regression models.
- Uses chronological train-test splitting for time-dependent data.
- Applies time-series cross-validation.
- Saves graphs and model evaluation results for further analysis.
- Uses `pathlib` for cross-platform file and directory management.

## Future Enhancements
- Experiment with additional ensemble algorithms such as Gradient Boosting.
- Tune model hyperparameters.
- Develop a web interface for interactive predictions.