import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Reading the dataset
dataset = pd.read_csv("obesitydataset.csv")

# DATA CLEANING
dataset = dataset.rename(columns={
    "FAVC":"HighCalorieFood", "FCVC":"VegetablesInMeals", "NCP":"MealsAday", 
    "CAEC":"Foodb/wMeals", "CH20":"WaterIntake", "SCC":"CalorieMonitoring", 
    "FAF":"PhysicalActivity", "TUE":"ScreenTime", "CALC":"AlcoholIntake", 
    "MTRANS":"Transportation", "NObeyesdad":"ObesityLevel", "CH2O":"WaterIntake"})

# Prepare the data
X = dataset.drop(columns=['ObesityLevel'])
y = dataset['ObesityLevel']

# Encode categorical variables
X = pd.get_dummies(X)

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Handle class imbalance with SMOTE
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_scaled, y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, random_state=42)

# Hyperparameter tuning using GridSearchCV for GradientBoostingClassifier
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.1, 0.2]
}

gb_clf = GradientBoostingClassifier(random_state=42)
grid_search = GridSearchCV(estimator=gb_clf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Best parameters
best_params = grid_search.best_params_
print(f'Best Parameters: {best_params}')

# Train the classifier with best parameters
best_gb_clf = grid_search.best_estimator_
best_gb_clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = best_gb_clf.predict(X_test)

# Compute metrics
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'F1 Score: {f1}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Classification Report:\n{classification_rep}')

# Save the model and columns
joblib.dump(best_gb_clf, 'model_gb.pkl')
joblib.dump(X.columns, 'X_train_columns_gb.pkl')

# Example user input
user_input = {
    'Gender': 'Female',
    'Age': 25,
    'Height': 1.65,
    'Weight': 55,
    'family_history_with_overweight': 'yes',
    'HighCalorieFood': 'no',
    'VegetablesInMeals': 3,
    'MealsAday': 3,
    'Foodb/wMeals': 'Sometimes',
    'SMOKE': 'no',
    'WaterIntake': 2,
    'CalorieMonitoring': 'no',
    'PhysicalActivity': 1,
    'ScreenTime': 2,
    'AlcoholIntake': 'Sometimes',
    'Transportation': 'Public_Transportation'  
}

# Convert the input to a DataFrame
user_input_df = pd.DataFrame([user_input])

# Encode categorical variables
user_input_df = pd.get_dummies(user_input_df)

# Ensure all columns in the training data are present
for col in X.columns:
    if col not in user_input_df.columns:
        user_input_df[col] = 0

# Ensure the order of columns is the same as in X_train
user_input_df = user_input_df[X.columns]

# Feature Scaling
user_input_df_scaled = scaler.transform(user_input_df)

# Predict the obesity level
predicted_obesity_level = best_gb_clf.predict(user_input_df_scaled)
print(f'Predicted Obesity Level: {predicted_obesity_level[0]}')

