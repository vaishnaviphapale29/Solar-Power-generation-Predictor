# Solar Power Generation Predictor

## Project Overview

The **Solar Power Generation Predictor** is an end-to-end Machine Learning project that analyzes solar power generation data and predicts AC power output based on weather conditions.

The project combines data preprocessing, exploratory data analysis (EDA), business analysis, Machine Learning model comparison, new data prediction, and an interactive Power BI dashboard.

The main objective is to understand how weather conditions affect solar power generation and build a model that can predict power output accurately.

## Objectives

- Analyze solar power generation and weather data.
- Clean, preprocess, and merge datasets.
- Perform exploratory data analysis (EDA).
- Identify relationships between weather conditions and AC power.
- Compare multiple Machine Learning regression models.
- Predict AC power using new weather conditions.
- Create visualizations and business performance reports.
- Develop an interactive Power BI dashboard.

## Technologies Used

- **Programming Language:** Python
- **Libraries:** Pandas, NumPy, Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn
- **Model:** Random Forest Regressor
- **Visualization:** Power BI
- **Development Environment:** VS Code, Jupyter Notebook (if used)
- **Model Saving:** Joblib / Pickle
- **Version Control:** Git and GitHub

## Dataset

The project uses two datasets:

### 1. Solar Power Generation Data

Contains information about solar power generation, including:

- `DATE_TIME`
- `PLANT_ID`
- `SOURCE_KEY`
- `DC_POWER`
- `AC_POWER`
- `DAILY_YIELD`
- `TOTAL_YIELD`

### 2. Weather Sensor Data

Contains weather measurements, including:

- `DATE_TIME`
- `PLANT_ID`
- `SOURCE_KEY`
- `AMBIENT_TEMPERATURE`
- `MODULE_TEMPERATURE`
- `IRRADIATION`

The datasets are processed and merged using the relevant date-time information to prepare the final dataset for analysis and model training.

## Project Workflow

1. Load the solar generation and weather datasets.
2. Inspect the data structure and data quality.
3. Check missing values and duplicate records.
4. Preprocess and merge the datasets.
5. Perform exploratory data analysis.
6. Generate business KPIs and correlation reports.
7. Extract time-based features such as hour, minute, day, and month.
8. Train and compare Machine Learning regression models.
9. Evaluate the best-performing model.
10. Save the trained model.
11. Predict AC power for new weather conditions.
12. Visualize results in Python and Power BI.

## Machine Learning Model

The project compares the following regression models:

- Linear Regression
- Random Forest Regressor

The models are compared using cross-validation MAE. Based on the reported comparison, **Random Forest Regressor** was selected as the best-performing model.

### Input Features

- Ambient Temperature
- Module Temperature
- Irradiation
- Hour
- Minute
- Day
- Month

### Target Variable

`AC_POWER`

The model predicts AC power output based on the selected weather and time-based features.

## Model Performance

The following results were obtained from the project run:

| Evaluation Metric | Result |
|---|---:|
| Mean Absolute Error (MAE) | 249.14 |
| Root Mean Squared Error (RMSE) | 526.45 |
| R² Score | 0.9956 |

The model achieved an R² score of **0.9956** on the test dataset.

*Note: These metrics represent the reported evaluation on the project's test split. Actual performance may vary on new or unseen data.*

## Sample Prediction

The `Test_New_Data.py` script accepts weather and time-based inputs to predict AC power.

### Sample Input

| Feature | Value |
|---|---:|
| Ambient Temperature | 25.0 |
| Module Temperature | 30.0 |
| Irradiation | 0.7 |
| Hour | 12 |
| Minute | 0 |
| Day | 15 |
| Month | 6 |

### Prediction Result

**Predicted AC Power: 19,190.30**

The prediction report is saved as `outputs/New_Data_Prediction.csv`, and the prediction graph is saved as `images/New_Data_Prediction.png`.

This is a sample model prediction, not a guarantee of actual solar power output.

## Business Analysis

The project generates business-oriented reports to understand solar power generation patterns.

Key metrics from the processed dataset include:

| Business Metric | Result |
|---|---:|
| Total Records | 3,157 |
| Average AC Power | 6,704.83 |
| Maximum AC Power | 29,150.21 |
| Minimum AC Power | 0 |
| Average Irradiation | 0.23 |
| Average Module Temperature | 31.18 |
| Average Ambient Temperature | 25.56 |

The analysis also includes:

- Daily, monthly, and hourly power generation analysis.
- Weather conditions and their relationship with AC power.
- Correlation analysis between input features and power output.
- Actual versus predicted power comparison.
- Prediction error distribution.

## Visualizations

The project generates visualizations to help understand power generation patterns and model performance.

Examples include:

- Hourly Solar Power Analysis
- Daily Solar Power Analysis
- Monthly Solar Power Analysis
- Irradiation vs Power
- Correlation Heatmap
- Actual vs Predicted Power
- New Data Prediction Graph

These visualizations are stored in the `images/` folder.

## Power BI Dashboard

An interactive Power BI dashboard is included in this project.

The dashboard presents solar power generation insights through KPI cards, charts, and date filtering.

### Dashboard Features

- Average AC Power KPI
- Maximum AC Power KPI
- Monthly Solar Power Generation
- Daily Generation Trend
- Irradiation vs AC Power Scatter Chart
- Date Slicer for interactive filtering

The Power BI dashboard file is:

`Solar_Power_Generation_Dashboard.pbix`

Open this file using Microsoft Power BI Desktop to explore the dashboard.

## Project Structure

```text
Solar_Power_Generation/
│
├── Data/
│   ├── Plant_1_Generation_Data.csv
│   └── Plant_1_Weather_Sensor_Data.csv
│
├── images/
│   ├── Hourly_Solar_Power.png
│   ├── Daily_Solar_Power.png
│   ├── Monthly_Solar_Power.png
│   ├── Irradiation_vs_Power.png
│   ├── Correlation_Heatmap.png
│   ├── Actual_vs_Predicted.png
│   └── New_Data_Prediction.png
│
├── outputs/
│   ├── Data_Quality_Report.csv
│   ├── Processed_Solar_Dataset.csv
│   ├── Solar_Business_KPIs.csv
│   ├── Daily_Solar_Analysis.csv
│   ├── Monthly_Solar_Analysis.csv
│   ├── Hourly_Solar_Analysis.csv
│   ├── Correlation_Report.csv
│   ├── Model_Comparison.csv
│   ├── Evaluation_Metrics.csv
│   ├── Actual_vs_Predicted.csv
│   ├── New_Data_Prediction.csv
│   └── Solar_Power_Model.pkl
│
├── Solar_Power_Generation_Predictor.py
├── Test_New_Data.py
├── Solar_Power_Generation_Dashboard.pbix
├── requirements.txt
└── README.md
```

*The structure above represents the intended project layout. Keep only the filenames and folders that actually exist in your repository.*

## Installation and Setup

### 1. Navigate to the Project Folder

```bash
cd Solar-Power-generation-Predictor
```

### 2. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### 3. Run the Main Project

```bash
python Solar_Power_Generation_Predictor.py
```

This script performs data processing, analysis, model training, evaluation, and output generation.

### 4. Test New Data Prediction

```bash
python Test_New_Data.py
```

Enter the weather and time-based values when prompted, or press Enter to use the default values.

The script generates a predicted AC power value, a CSV prediction report, and a prediction graph.

**Important:** Ensure the required datasets are present in the expected `Data/` folder before running the scripts.

## Future Improvements

- Add SQL-based data analysis.
- Improve model validation using additional test periods.
- Experiment with more regression algorithms and hyperparameter tuning.
- Deploy the model as a web application.
- Integrate real-time weather data for updated predictions.
- Add more interactive features to the Power BI dashboard.

## Conclusion

The Solar Power Generation Predictor demonstrates an end-to-end Machine Learning workflow, from raw data preprocessing and exploratory analysis to model evaluation and prediction on new inputs.

By combining Python, Scikit-learn, and Power BI, the project provides insights into solar power generation patterns and demonstrates practical skills in data analysis, regression modeling, visualization, and reporting.