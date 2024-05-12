# pip install numpy
# pip install pandas
# pip install matplotlib
# pip install -U scikit-learn
# pip install -U imbalanced-learn

import pandas as pd

from sklearn.model_selection import (
    train_test_split,
)

from imblearn.over_sampling import (
    RandomOverSampler,
)

from sklearn.ensemble import (
    RandomForestClassifier,
)

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

import joblib

# Load data
data_df = pd.read_csv(
    "diabetes_prediction_dataset.csv",
    usecols=[
        "smoking_history",
        "HbA1c_level",
        "bmi",
        "diabetes",
    ],
)

# Filter on smoking_history
smoking_class_list = [
    "never",
    "current",
]
data_df = data_df[
    data_df["smoking_history"].isin(
        smoking_class_list
    )
]

data_df["diabetes"].value_counts()

# Format categorical feature to
# one-hot encoding
# leave numeric feature unchanged
x = pd.get_dummies(
    data_df, columns=["smoking_history"]
)

# Extract target to predict
y = x.pop("diabetes")

# Create train and test data
(
    x_train,
    x_test,
    y_train,
    y_test,
) = train_test_split(
    x, y, test_size=0.333
)

# Imbalanced dataset with
# fewer diabetes data points
y_train.value_counts()

# Decided to randomly over
# sample the minority class
ros = RandomOverSampler(random_state=12)
(
    x_train_ros,
    y_train_ros,
) = ros.fit_resample(x_train, y_train)

# Now a balanced dataset
y_train_ros.value_counts()

# Created and trained RF model
# 250 decision trees were trained
# in the model
rf_classifier = RandomForestClassifier(
    min_samples_leaf=50,
    n_estimators=250,
    bootstrap=True,
    oob_score=True,
    n_jobs=-1,
)
rf_classifier.fit(
    x_train_ros, y_train_ros
)

# Save the trained model to a file
joblib.dump(
    rf_classifier,
    "random_forest_model.pkl",
)

# Predicted the test data target diabetes
# Output the accuracy using predictions
# and known test data diabetes values
y_pred = rf_classifier.predict(x_test)
accuracy_score(y_test, y_pred)
print(
    f"Accuracy= {round(accuracy_score(y_test,y_pred),3)*100} %"
)

# Output additional metrics
print(
    classification_report(
        y_test, y_pred
    )
)

confusion_matrix(y_test, y_pred)

# Load the saved Random Forest model
rf_model = joblib.load('random_forest_model.pkl')
print(rf_model)

# Make a prediction for a single
# data point
d = {'bmi': [50],
     'HbA1c_level': [2.2],
     'smoking_history_current': [0],
     'smoking_history_never': [1]}
df = pd.DataFrame(data=d)
pred = rf_model.predict(df)
print(pred)