# 🏦 Bank Term Deposit Subscription Predictor

An end-to-end machine learning project that predicts whether a bank client will subscribe to a term deposit, based on client details and marketing campaign data. The project covers data preprocessing, model comparison, hyperparameter tuning, and deployment as an interactive **Streamlit** web app.

**Repository:** [banking_customer_account_prediction_](https://github.com/Palavalasamounika13/banking_customer_account_prediction_)

## Features

- Interactive form for client and campaign inputs (sliders, dropdowns, radio buttons)
- Instant **YES / NO** prediction
- Probability for both classes, with a confidence progress bar
- Model loaded once and cached with `st.cache_resource`

## Project Structure

```
.
├── app.py                                  # Streamlit application
├── model.pkl                               # Trained pipeline (preprocessor + tuned Logistic Regression)
├── requirements.txt                        # Python dependencies
├── sprint_banking_customer_account__.py    # Training notebook exported as a script (Colab)
└── README.md
```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Palavalasamounika13/banking_customer_account_prediction_.git
cd banking_customer_account_prediction_
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate      # macOS / Linux
venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt` pins `scikit-learn==1.6.1`. Keep this version, since pickled models can break across scikit-learn versions.

### 4. Make sure the model file is named `model.pkl`

The training notebook saves the pipeline as `banking_svm_model.pkl`, but `app.py` loads `model.pkl` from the same folder. Rename the file (or change the path in `app.py`):

```bash
mv banking_svm_model.pkl model.pkl
```

### 5. Run the app

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## Input Features

The model expects 16 features (Bank Marketing dataset schema):

### Client information

| Feature     | Description                              | Type        |
|-------------|------------------------------------------|-------------|
| `age`       | Client age (18–95)                       | Numeric     |
| `job`       | Type of job (e.g. management, retired)   | Categorical |
| `marital`   | Marital status                           | Categorical |
| `education` | Education level                          | Categorical |
| `default`   | Has credit in default?                   | yes / no    |
| `balance`   | Account balance in euros                 | Numeric     |
| `housing`   | Has housing loan?                        | yes / no    |
| `loan`      | Has personal loan?                       | yes / no    |

### Campaign details

| Feature    | Description                                         | Type        |
|------------|-----------------------------------------------------|-------------|
| `contact`  | Contact communication type                          | Categorical |
| `month`    | Last contact month                                  | Categorical |
| `day`      | Last contact day of the month (1–31)                | Numeric     |
| `duration` | Last call duration in seconds                       | Numeric     |
| `campaign` | Number of contacts during this campaign             | Numeric     |
| `pdays`    | Days since previous campaign contact (`-1` = never) | Numeric     |
| `previous` | Number of contacts before this campaign             | Numeric     |
| `poutcome` | Outcome of the previous campaign                    | Categorical |

**Target:** `y` (`yes` / `no`), whether the client subscribed to a term deposit.

## Machine Learning Workflow

The full workflow is in `sprint_banking_customer_account__.py` (exported from Google Colab), organized in three sprints.

### Sprint 1: Data understanding and preprocessing

- Loaded `train.csv` (semicolon-separated)
- Inspected shape, dtypes, missing values, and unique counts
- Removed duplicate rows and standardized column names
- Exploratory analysis: histograms, skewness, count plots, correlation heatmap, pair plot
- Outlier analysis with box plots and IQR-based capping
- Preprocessing with a `ColumnTransformer`:
  - `StandardScaler` for numeric features
  - `OneHotEncoder(handle_unknown='ignore')` for categorical features
- 80/20 train-test split (`random_state=42`)

### Sprint 2: Model building and evaluation

Five models were trained and compared on accuracy, precision, recall, F1 score, and confusion matrix, with a train-vs-test accuracy check for overfitting:

| Model                  | Remark                                    |
|------------------------|-------------------------------------------|
| Logistic Regression    | Baseline model                            |
| Decision Tree          | Prone to overfitting, high variance       |
| Random Forest          | Ensemble method, good accuracy/stability  |
| Support Vector Machine | Effective in high-dimensional spaces      |
| Gaussian Naive Bayes   | Simple and fast                           |

### Sprint 3: Optimization and final model

- Hyperparameter tuning of Logistic Regression with `GridSearchCV` (5-fold CV, accuracy scoring):
  - `C`: 0.01, 0.1, 1, 10, 100
  - `penalty`: l1, l2
  - `solver`: liblinear, saga
- Final evaluation: accuracy, classification report, confusion matrix, ROC-AUC
- Preprocessor and tuned classifier bundled into one scikit-learn `Pipeline` and serialized with `pickle`

Because the preprocessing lives inside the pipeline, `app.py` passes raw form inputs straight to the model.

## Notes and Limitations

- **`duration` caveat:** Call duration is only known after a call ends, so it is not available before contacting a client. Models using it are useful for analysis but unrealistic for real pre-call targeting.
- Predictions are statistical estimates, not guarantees. Do not use them as the only basis for business decisions.
- Only load `.pkl` files from sources you trust, since unpickling untrusted files can execute arbitrary code.

## Tech Stack

- [Python](https://www.python.org/)
- [Streamlit](https://streamlit.io/)
- [pandas](https://pandas.pydata.org/) / [NumPy](https://numpy.org/)
- [scikit-learn](https://scikit-learn.org/)
- [Matplotlib](https://matplotlib.org/) / [Seaborn](https://seaborn.pydata.org/) (EDA)

## Dataset

Bank Marketing dataset (direct marketing campaigns of a Portuguese banking institution), available from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/222/bank+marketing).

## License

Add your preferred license here (e.g. MIT).
