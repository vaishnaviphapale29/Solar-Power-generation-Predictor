# Solar Power Generation Predictor

## Project Overview
The Solar Power Generation Predictor is a Machine Learning project that predicts solar power generation using weather conditions and time-based features. It uses historical solar generation and weather data to train and evaluate regression models.

## Objectives
- Predict solar power generation using weather data.
- Analyze the relationship between weather conditions and AC power.
- Compare Machine Learning regression models.
- Evaluate model performance using regression metrics.
- Visualize actual and predicted power values.

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib

## Dataset Description
The project uses two datasets:

**1. Generation Dataset**
- Date and time
- Plant ID
- Source key
- DC power
- AC power
- Daily yield
- Total yield

**2. Weather Dataset**
- Date and time
- Plant ID
- Source key
- Ambient temperature
- Module temperature
- Irradiation

## Project Workflow
1. Load the generation and weather datasets.
2. Understand dataset structure and statistical information.
3. Clean and preprocess the data.
4. Perform Exploratory Data Analysis (EDA).
5. Merge relevant generation and weather information.
6. Create time-based features such as hour, minute, day, and month.
7. Select input features and the target variable.
8. Split data into training and testing sets.
9. Train Linear Regression and Random Forest models.
10. Compare models using Time Series Cross-Validation.
11. Evaluate the selected model.
12. Save the trained model and output files.
13. Visualize actual and predicted power values.

## Model Performance

Two regression models were trained and compared using Time Series Cross-Validation.

| Model | Cross-Validation MAE |
|---|---:|
| Linear Regression | 514.2837 |
| Random Forest | **333.0444** |

**Selected Model:** Random Forest Regressor

### Evaluation Metrics

| Metric | Score |
|---|---:|
| Mean Absolute Error (MAE) | 249.1383 |
| Root Mean Squared Error (RMSE) | 526.4474 |
| R² Score | 0.9956 |

### Dataset Summary

- Generation dataset records: 68,778
- Weather dataset records: 3,182
- Processed dataset records: 3,157
- Training records: 2,525
- Testing records: 632
- Input features: 7
- Target variable: `AC_POWER`

### Key Findings
- Random Forest achieved a lower cross-validation MAE than Linear Regression.
- The selected model achieved an R² score of 0.9956 on the test dataset.
- Irradiation showed a strong correlation with AC power in the processed dataset.
- The project generated actual-versus-predicted comparisons and prediction visualizations.

## Sample Prediction

**Sample input:**
- Ambient Temperature: 25°C
- Module Temperature: 30°C
- Irradiation: 0.7
- Hour: 12
- Minute: 0
- Day: 15
- Month: 6

**Predicted AC Power:** 19,190.302

*Note: This is an example prediction using sample input values.*

## Visualizations

The project generates the following visualizations:

- `Power_Distribution.png`
- `Irradiation_Distribution.png`
- `Irradiation_vs_Temperature.png`
- `Correlation_Heatmap.png`
- `Actual_vs_Predicted.png`
- `Actual_Predicted_Scatter.png`
- `Prediction_Error_Distribution.png`

These charts help analyze power distribution, weather relationships, model predictions, and prediction errors.

## Output Files

The `outputs/` folder contains:

- `Processed_Solar_Dataset.csv`
- `Actual_vs_Predicted.csv`
- `Model_Comparison.csv`
- `Evaluation_Metrics.csv`
- `Solar_Power_Model.pkl`

## How to Run the Project

1. Clone or download the repository.
2. Install the required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

3. Run the Python script:

```bash
python Solar_Power_Generation_Predictor.py
```

*Ensure the dataset paths in the script match your local folder structure.*

## Future Improvements
- Experiment with additional regression algorithms and hyperparameter tuning.
- Evaluate performance on data from different solar plants and time periods.
- Integrate live weather data for power forecasting.
- Develop an interactive dashboard for monitoring solar power generation.

## Conclusion
This project demonstrates an end-to-end Machine Learning workflow for solar power prediction, including data preprocessing, exploratory analysis, feature engineering, model comparison, evaluation, and visualization.