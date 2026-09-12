import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from catboost import CatBoostClassifier

DATA_DIR = "F://Electric_Vehicle_Purchases//Data"
TRAIN_PATH = f"{DATA_DIR}//train.csv"
TEST_PATH = f"{DATA_DIR}//test.csv"
SUBMISSION_PATH = "F://Electric_Vehicle_Purchases//Submission//final_submission.csv"
MODEL_PATH = "F://Electric_Vehicle_Purchases//Model//final_stacking_model.joblib"


vehicle_df = pd.read_csv(TRAIN_PATH)

vehicle_df = vehicle_df.drop(columns=["Gender", "Current_Car_Type", "City_Type"])

vehicle_df["Will_Buy_EV"] = vehicle_df["Will_Buy_EV"].map({"No": 0, "Yes": 1})
vehicle_df["Subsidy_Available"] = vehicle_df["Subsidy_Available"].map({"Yes": 1, "No": 0})
vehicle_df["Home_Charging_Possible"] = vehicle_df["Home_Charging_Possible"].map({"Yes": 1, "No": 0})

X = vehicle_df.drop(columns=["Will_Buy_EV"])
y = vehicle_df["Will_Buy_EV"]


ordinal_pipeline = Pipeline([
    ("encoder", OrdinalEncoder(categories=[["Low", "Medium", "High"]])),
    ("scaler", StandardScaler()),
])

numeric_pipeline = Pipeline([
    ("scaler", StandardScaler()),
])

numeric_features = [
    "Age",
    "Annual_Income_USD",
    "Daily_Commute_km",
    "Number_of_Cars_Owned",
    "Charging_Stations_Near_Home",
    "Charging_Stations_Near_Work",
    "Environmental_Concern_Level",
    "Home_Charging_Possible",
    "Subsidy_Available",
]

preprocessor = ColumnTransformer([
    ("categorical", ordinal_pipeline, ["Range_Anxiety_Level"]),
    ("numeric", numeric_pipeline, numeric_features),
])


logistic_pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("model", LogisticRegression(
        C=0.01,
        max_iter=1000,
        solver="saga",
        penalty="elasticnet",
        random_state=42,
    )),
])

forest_pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=500,
        min_samples_split=20,
        min_samples_leaf=2,
        max_features="log2",
        max_depth=20,
        criterion="entropy",
        random_state=42,
    )),
])

catboost_pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("model", CatBoostClassifier(
        iterations=200,
        learning_rate=0.2,
        l2_leaf_reg=5,
        depth=4,
        verbose=False,
        random_state=42,
    )),
])


stack_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

stacking_clf = StackingClassifier(
    estimators=[
        ("logistic", logistic_pipeline),
        ("cat", catboost_pipeline),
        ("rf", forest_pipeline),
    ],
    final_estimator=LogisticRegression(max_iter=1000, C=0.1, random_state=42),
    cv=stack_cv,
    n_jobs=1,
    passthrough=False,
)

stacking_clf.fit(X, y)


test_df = pd.read_csv(TEST_PATH)
predictions = stacking_clf.predict_proba(test_df)[:, 1]

submission = pd.DataFrame({
    "id": test_df["id"],
    "Will_Buy_EV": predictions,
})
submission.to_csv(SUBMISSION_PATH, index=False)


joblib.dump(stacking_clf, MODEL_PATH)