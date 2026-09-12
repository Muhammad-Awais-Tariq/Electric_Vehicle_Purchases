# Electric Vehicle Purchase Prediction

An end-to-end machine learning project predicting the probability that a person will buy an electric vehicle (EV), built with a scikit-learn preprocessing pipeline, compared across ten different models/ensembles, and deployed as an interactive Streamlit web app. Built for Kaggle's Playground Series - Season 6, Episode 9.

## Live Demo

Try it here: [https://electricvehicleprediction.streamlit.app/](https://electricvehicleprediction.streamlit.app/)

Enter a person's details (age, income, commute distance, charging access, environmental concern, etc.) and get a live prediction of their EV-purchase probability.

---

## Project Structure

```
Electric_Vehicle_Purchases/
│
├── Data/                              # Raw datasets (gitignored — not pushed to GitHub)
│   ├── train.csv                      # Original Kaggle training data
│   └── test.csv                       # Original Kaggle test data
│
├── Exploration_notebook/
│   └── exploration.ipynb              # Notebook: EDA, preprocessing iteration,
│                                       # model comparison, hyperparameter tuning
│
├── Model/
│   └── final_stacking_model.joblib    # Serialized final trained pipeline (model +
│                                       # preprocessing, all in one)
│
├── Reports/
│   └── data_report.html               # Exploratory data profiling report (sweetviz)
│
├── Submission/
│   └── final_submission.csv           # Kaggle submission — final model trained on full data
│
├── final_pipeline.py                  # Clean, final training script — builds the
│                                       # preprocessing pipeline, trains the final
│                                       # model, and saves it with joblib
├── app.py                             # Streamlit app — loads the saved model and
│                                       # serves the interactive prediction UI
│
├── pyproject.toml                     # Project dependencies (for uv)
├── uv.lock                            # Locked dependency versions
├── .python-version                    # Python version pin
├── .venv/                             # Local virtual environment (gitignored)
├── .gitignore                         # Excludes Data/ and .venv/ from version control
└── README.md                          # This file
```

> **Note on data:** The `Data/` folder (raw CSVs) is excluded via `.gitignore` and is **not** pushed to GitHub. To run this project locally, download `train.csv` and `test.csv` from the [Kaggle Predicting Electric Vehicle Purchases competition](https://www.kaggle.com/competitions/playground-series-s6e9) and place them in a local `Data/` folder.

---

## What Each File Does

### `Exploration_notebook/exploration.ipynb`
The working notebook where all the experimentation happened:
- Exploratory data analysis on the raw dataset (with a sweetviz profiling report)
- Encoding categorical/boolean fields (`Will_Buy_EV`, `Subsidy_Available`, `Home_Charging_Possible` mapped Yes/No → 1/0, `Range_Anxiety_Level` ordinal-encoded Low/Medium/High)
- Dropping low-signal categorical columns (`Gender`, `City_Type`, `Current_Car_Type`, `id`) after checking their correlation with the target
- Building the sklearn `Pipeline` + `ColumnTransformer` (ordinal encoding + scaling for `Range_Anxiety_Level`, scaling for the remaining numeric columns)
- Training and cross-validating ten different models/ensembles
- Hyperparameter tuning with `RandomizedSearchCV` / `GridSearchCV` for each model
- Building and tuning a final `StackingClassifier` on top of the best base learners
- Fitting the final model on the full training set and generating the submission

### `final_pipeline.py`
The clean, production version of the pipeline:
1. Loads `train.csv`
2. Builds the `ColumnTransformer`: ordinal encoding + scaling for `Range_Anxiety_Level`, scaling for the remaining numeric columns
3. Fits the final chosen model (tuned `StackingClassifier`) on the full training set
4. Serializes the trained pipeline with `joblib` for deployment

### `app.py`
The Streamlit web app. Loads `final_stacking_model.joblib`, presents a form for entering a person's details, and returns a live EV-purchase probability prediction.

---

## Features Used

The raw dataset columns are: `id`, `Age`, `Annual_Income_USD`, `Daily_Commute_km`, `Number_of_Cars_Owned`, `Charging_Stations_Near_Home`, `Charging_Stations_Near_Work`, `Environmental_Concern_Level`, `Gender`, `City_Type`, `Current_Car_Type`, `Home_Charging_Possible`, `Subsidy_Available`, `Range_Anxiety_Level`, and the target `Will_Buy_EV`.

`Gender`, `City_Type`, `Current_Car_Type`, and `id` were dropped, as they contributed little to the correlation with the target. The final feature set passed into the model is: `Age`, `Annual_Income_USD`, `Daily_Commute_km`, `Number_of_Cars_Owned`, `Charging_Stations_Near_Home`, `Charging_Stations_Near_Work`, `Environmental_Concern_Level`, `Home_Charging_Possible`, `Subsidy_Available`, `Range_Anxiety_Level`.

By correlation with the target, `Environmental_Concern_Level` and `Subsidy_Available` were the strongest predictors, followed by `Annual_Income_USD` and `Range_Anxiety_Level`.

**Preprocessing (`ColumnTransformer`):**
- **Ordinal categorical** (`Range_Anxiety_Level`, ordered Low < Medium < High) → `OrdinalEncoder` → `StandardScaler`
- **Numeric** (`Age`, `Annual_Income_USD`, `Daily_Commute_km`, `Number_of_Cars_Owned`, `Charging_Stations_Near_Home`, `Charging_Stations_Near_Work`, `Environmental_Concern_Level`, `Home_Charging_Possible`, `Subsidy_Available`) → `StandardScaler`

---

## Evaluation Metric

Every model was scored with **ROC AUC** (`scoring="roc_auc"`) rather than accuracy, since the competition is judged on ranked probability estimates rather than hard class labels. Predictions are submitted as probabilities via `predict_proba` for the same reason.

---

## Models Compared

Ten approaches were trained and evaluated with 5-fold stratified cross-validation (`StratifiedKFold`, `scoring="roc_auc"`), most on a stratified 15% sample of the training data during model selection and tuned with `RandomizedSearchCV` / `GridSearchCV` where applicable:

| Model | CV ROC AUC |
|---|---|
| Decision Tree | ~0.9306 |
| Polynomial SVC (degree 2, calibrated, tuned) | ~0.9363 |
| Linear SVC (calibrated, tuned) | ~0.9380 |
| Logistic Regression (ElasticNet, tuned) | ~0.9381 |
| Random Forest (tuned) | ~0.9380 |
| AdaBoost (Decision Tree base, tuned) | ~0.9391 |
| LightGBM (tuned) | ~0.9402 |
| Voting Classifier (soft, weighted, SVC + Logistic + XGBoost) | ~0.9403 |
| Bagging Classifier (Decision Tree base, tuned) | ~0.9404 |
| XGBoost (tuned) | ~0.9408 |
| CatBoost (tuned, full training data) | ~0.9415 |
| **Stacking Classifier (final)** | **~0.9414** |

**Final model: Stacking Classifier** — base learners are the tuned Logistic Regression, CatBoost, and Random Forest pipelines, combined via a `LogisticRegression` meta-estimator, with the meta-estimator's `C` and `class_weight` tuned via `GridSearchCV`. This is the model saved as `final_stacking_model.joblib` and used for deployment.

**Final Stacking Classifier configuration:**
```python
StackingClassifier(
    estimators=[
        ("logistic", logistic_pipeline),   # ElasticNet, C=0.01
        ("cat", catboost_pipeline),        # depth=4, iterations=200, learning_rate=0.2, l2_leaf_reg=5
        ("rf", forest_pipeline),           # tuned RandomForestClassifier
    ],
    final_estimator=LogisticRegression(C=0.1, class_weight=None, max_iter=1000),
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    passthrough=False
)
```

---

## Submission

The final model is trained on 100% of the available training data (`final_pipeline.py`), so its Kaggle public leaderboard score reported after submission is the true measure of its generalization performance.

**Kaggle competition:** [Predicting Electric Vehicle Purchases — Playground Series S6E9](https://www.kaggle.com/competitions/playground-series-s6e9)

---

## How to Run Locally

### Prerequisites
- Python (see `.python-version`)
- `uv` package manager (or `pip`)

### Setup
```bash
git clone <repository-url>
cd Electric_Vehicle_Purchases

# Download train.csv and test.csv from Kaggle's Predicting Electric Vehicle Purchases competition
# and place them inside a local Data/ folder (not included in this repo)

uv sync
```

### Train the model
```bash
uv run python final_pipeline.py
```
This regenerates `Model/final_stacking_model.joblib`.

### Run the Streamlit app
```bash
uv run streamlit run app.py
```
Then open the URL shown in your terminal (usually `http://localhost:8501`).

---

## Technologies Used

- [scikit-learn](https://scikit-learn.org/) — pipelines, preprocessing, models, cross-validation, hyperparameter search
- [XGBoost](https://xgboost.readthedocs.io/) — gradient-boosted tree model
- [LightGBM](https://lightgbm.readthedocs.io/) — gradient-boosted tree model
- [CatBoost](https://catboost.ai/) — gradient-boosted tree model
- [Pandas](https://pandas.pydata.org/) / [NumPy](https://numpy.org/) — data manipulation
- [Sweetviz](https://github.com/fbdesignpro/sweetviz) — exploratory data profiling
- [Streamlit](https://streamlit.io/) — interactive web app deployment
- [joblib](https://joblib.readthedocs.io/) — model serialization

---

## Key Learnings from This Project

- Building a `Pipeline`/`ColumnTransformer` with mixed ordinal and standard scaling for different column groups
- Deciding which categorical columns to drop based on their correlation with the target rather than encoding everything
- Comparing linear, tree-based, boosted, bagged, voted, and stacked models under a shared evaluation setup
- Hyperparameter tuning with `RandomizedSearchCV` and `GridSearchCV` across multiple model families (Logistic Regression, SVC, Decision Tree, Random Forest, XGBoost, LightGBM, CatBoost, AdaBoost, Bagging)
- Building a `StackingClassifier` from the strongest base learners with a tuned meta-estimator
- Choosing ROC AUC as the evaluation metric and submitting probability scores instead of hard labels
- Serializing a full pipeline (preprocessing + model) for deployment
- Deploying a trained pipeline behind a live Streamlit interface

---

## Author

Muhammad Awais Tariq

## References

- [Kaggle Predicting Electric Vehicle Purchases Competition](https://www.kaggle.com/competitions/playground-series-s6e9)
- [scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Streamlit Documentation](https://docs.streamlit.io/)

---

If you found this project useful, consider giving it a star.