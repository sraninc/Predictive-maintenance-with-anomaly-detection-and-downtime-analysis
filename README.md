# Predictive Maintenance with Anomaly Detection and Downtime Prediction

This project, developed by **Sneha Rani, PhD** focuses on detecting anomalies and predicting downtimes in a multi-stage continuous manufacturing process. The data consists of sensor measurements from two stages of the production line, where we aim to detect anomalies in the measurements and predict downtime events for maintenance purposes.

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
   - **SMOTE (Synthetic Minority Oversampling Technique)** and **Class Weighting** was applied to handle the class imbalance between downtime and normal operations.

---

## Anomaly Detection

- Values were classified as anomalies if they fall outside a predefined acceptable range (±10%) around the setpoint.
---

## Downtime Prediction

### Labeling Downtime:
- A downtime indicator was created based on the presence of anomalies in the Stage 1 and Stage 2 output measurements using majority rule for downtime detection.
- Downtime events were then grouped, and the duration of each event was calculated for further analysis.

---

## Models used

### Random Forest with SMOTE:
- **Random Forest** was used for downtime prediction due to its ability to handle high-dimensional data and its robustness to overfitting.
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
### Stage 1 Downtime Prediction (Random Forest after Hyperparameter Tuning)
- **Accuracy:** 92%
- **Confusion Matrix:**
```python
[[821   154]]
 [ 73   1770]]
```
- **Classification Report:**
  - Precision (Class 1 - Downtime): 92%
  - Recall (Class 1 - Downtime): 96%
  - F1-Score (Class 1 - Downtime): 0.94
- **Top 5 Importance features:**
  1. Machine3.MaterialTemperature.U.Actual	
  2. Machine3.RawMaterial.Property2
  3. Machine3.MaterialPressure.U.Actual
  4. Machine3.RawMaterial.Property4	
  5. FirstStage.CombinerOperation.Temperature2.U.Actual
- **Conclusion:**
  - The tuned Random Forest model for Stage 1 downtime prediction achieved high accuracy, with important features including machine temperature and raw material properties metrics. 
    
### Stage 2 Downtime Prediction (Random Forest with SMOTE after Hyperparameter Tuning)
- **Accuracy:** 90%
- **Confusion Matrix:**
```python
[[2148   143]
 [ 143   384]]
```
- **Classification Report:**
  - Precision (Class 1 - Downtime): 73%
  - Recall (Class 1 - Downtime): 73%
  - F1-Score (Class 1 - Downtime): 0.73
- **Top 5 Important features:**
  1. Stage1.Output.Measurement1.U.Actual	
  2. FirstStage.CombinerOperation.Temperature2.U.Actual
  3. Stage1.Output.Measurement0.U.Actual	
  4. Stage1.Output.Measurement2.U.Actual
  5. AmbientConditions.AmbientHumidity.U.Actual
- **Conclusion:**
  - Stage 2's downtime prediction model performed well with high accuracy and identified critical features such as machine humidity and stage 1 predictions. The recall for downtime events is slightly higher in Stage 1.
---

## How to Run
1. Clone the repository:
```python
git clone https://github.com/sraninc/predictive_maintenance_liveline_tech.git
cd predictive_maintenance_liveline_tech
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
- **Fine-Tuning Models**: I performed the hyperparameter tuning for Random Forest Classifier but there is always a scope of further hyperparameter tuning to imporve recall for predicting downtime events.
- **Handling Class Imbalance**: While the fine-tuning has improved performance, the imbalance between downtime and normal operations is still impacting the recall for downtime events. I would suggest using more advanced techniques like **cost-sensitive learning**.
   
## License
This project is licensed under the MIT License.



