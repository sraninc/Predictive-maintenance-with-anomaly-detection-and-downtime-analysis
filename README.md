# Predictive Maintenance with Anomaly Detection and Downtime Prediction

This project, developed by **Sneha Rani** focuses on detecting anomalies and predicting downtimes in a multi-stage continuous manufacturing process. The data consists of sensor measurements from two stages of the production line, where we aim to detect anomalies in the measurements and predict downtime events for maintenance purposes.

## Table of Contents
- [Introduction](#introduction)
- [Project Workflow](#project-workflow)
- [Anomaly Detection](#anomaly-detection)
- [Downtime Prediction](#downtime-prediction)
- [Models Used](#models-used)
- [Feature Importance](#feature-importance)
- [Results](#results)
- [How to Run](#how-to-run)

---

## Introduction

In this project, I applied machine learning techniques to time series data from a manufacturing process with two stages. The goal was to identify anomalies in sensor measurements and use them to predict downtime events. This type of analysis is critical for **predictive maintenance**, which helps in minimizing equipment failure and improving operational efficiency.

The main focus areas include:
- Detecting anomalies in sensor readings.
- Predicting downtimes based on identified anomalies.
- Tuning models to improve downtime prediction accuracy.

---

## Project Workflow

1. **Data Preprocessing**:
   - Data was first cleaned and resampled to ensure a regular time interval between observations (1-second intervals).
   - We dealt with missing values through linear interpolation.
   
2. **Anomaly Detection**:
   - Applied **Isolation Forest** and **Z-Score** methods for detecting anomalies in the sensor data.
   - Each sensor's anomaly was labeled and marked with a binary indicator.
   
3. **Downtime Indicator Creation**:
   - Downtime events were identified based on the anomalies in Stage 1 and Stage 2 outputs.
   - Separate downtime indicators for both stages were created, which formed the target labels for prediction.

4. **Modeling and Evaluation**:
   - Various baseline models, including **Random Forest** and **Logistic Regression**, were used to predict downtime.
   - **SMOTE (Synthetic Minority Oversampling Technique)** was applied to handle the class imbalance between downtime and normal operations.

---

## Anomaly Detection

### Methods Used:
- **Isolation Forest**: This unsupervised learning algorithm was used to detect anomalies in the sensor readings. It isolates observations that differ significantly from the rest of the data.
- **Z-Score Method**: Another method was applied to mark anomalies by calculating the Z-score for each sensor measurement. Measurements with a Z-score greater than a threshold were marked as anomalies.

### Code Example:
```python
from sklearn.ensemble import IsolationForest
iso_forest = IsolationForest(contamination=0.01)

# Apply Isolation Forest for anomaly detection
for f in sensor_features:
    resampled_data_with_anomalies[f'{f}_anomaly'] = iso_forest.fit_predict(resampled_data[[f]])
```
---

## Downtime Prediction

### Labeling Downtime:
- A downtime indicator was created based on the presence of anomalies in the Stage 1 and Stage 2 output measurements.
- Downtime events were then grouped, and the duration of each event was calculated for further analysis.

### Code Example:
```python
# Create a downtime indicator for Stage 1
esampled_data_with_ano['stage1_downtime_indicator'] = resampled_data_with_ano[stage1_anomaly_columns].max(axis=1)

# Calculate downtime durations
downtime_durations = downtime_events.groupby('downtime_event')['time_stamp'].agg(['min', 'max'])
downtime_durations['duration_seconds'] = (downtime_durations['max'] - downtime_durations['min']).dt.total_seconds()
```
---

## Models used

### Random Forest with SMOTE:
- **Random Forese** was used for downtime prediction due to its ability to handle high-dimensional data and its robustness to overfitting.
- **SMOTE** was applied to address the class imbalance problem by oversampling the minority class (Downtime events).

### Logistic Regression:
- A simpled baseline maodel, **Logistic Regression**, was used as a comparison for the Random Forest. Class weighting and SMOTE were applied to improve performance.

### LSTM (Long Short-Term Memory):
- LSTM, a type of **Recurreint Newural Network (RNN)**, was applied to capture time dependencies in the data and predict downtime events.

### Code Example for Random Forest with SMOTE:
```python
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# Apply SMOTE for class imbalance
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

# Train Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_resampled, y_train_resampled)
```
---

## Feature Importance
Feature importance analysis helps identify which features contributed most to the downtime prediction model. This was done using the **Random Forest's feature importance attribute.**

### Example of Feature Importance:
```python
# Get feature importance from the Random Forest model
feature_importances = rf_model.feature_importances_

# Create a DataFrame to display feature importance
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances}).sort_values(by='Importance', ascending=False)
```
---

## Results
### Stage 1 Downtime Prediction (Random Forest with SMOTE and Hyperparameter Tuning)
- **Accuracy:** 93%
- **Confusion Matrix:**
```python
[[2541   41]]
 [ 152   84]]
```
- **Classification Report:**
  - Precision (Class 1 - Downtime): 67%
  - Recall (Class 1 - Downtime): 36%
  - F1-Score (Class 1 - Downtime): 0.47
- **Top 5 Importance features:**
  1. AmbientConditions.AmbientHumidity.U.Actual
  2. Machine3.MaterialTemperature.U.Actual
  3. Machine1.MotorRPM.C.Actual
  4. Machine2.Zone1TemperatureC.Actual
  5. Machine2.MaterialTemperature.U.Actual
- **Conclusion:**
  - The tuned Random Forest model for Stage 1 downtime prediction achieved high accuracy, with important features including environmental and machine temperature metrics. However, recall for downtime (class 1) events can be improved, as it missed some downtime events.
    
### Stage 2 Downtime Prediction (Random Forest with SMOTE and Hyperparameter Tuning)
- **Accuracy:** 93%
- **Confusion Matrix:**
```python
[[2510   77]
 [ 107   124]]
```
- **Classification Report:**
  - Precision (Class 1 - Downtime): 62%
  - Recall (Class 1 - Downtime): 54%
  - F1-Score (Class 1 - Downtime): 0.57
- **Top 5 Important features:**
  1. Machine4.Temperature3.C.Actual
  2. Machine3.MaterialTemperature.U.Actual
  3. Machine1.MaterialTemperature.U.Actual
  4. Machine5.ExitTemperature.U.Actual
  5. Machine3.MaterialPressure.U.Actual
- **Conclusion:**
  - Stage 2's downtime prediction model performed well with high accuracy and identified critical features such as machine temperatures and exit temperatures. The recall for downtime events is slightly higher than in Stage 1, indicating improved detection of downtime periods.
---

## How to Run
1. Clone the repository:
```python
git clone https://github.com/your-username/anomaly-detection-downtime-prediction.git
cd anomaly-detection-downtime-prediction
```

2. Install required dependencies:
```python
pip install -r requirements.txt
```

3. Run the script:
```python
python main.py
```

4. For visualization dashboards, run:
```python
python visualize_downtime_dashboard.py
```
---

## Future Work:
- **Fine-Tuning Models**: I performed the hyperparameter tuning for Random Forest Classifier with SMOTE but there is always a scope of further hyperparameter tuning to imporve recall for predicting downtime events.
- **Handling Class Imbalance**: While the fine-tuning has improved performance, the imbalance between downtime and normal operations is still impacting the recall for downtime events. I would suggest using more advanced techniques like **cost-sensitive learning**.
   
## License
This project is licensed under the MIT License.



