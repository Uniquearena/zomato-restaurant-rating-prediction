# Zomato Restaurant Rating Prediction

A machine learning project that predicts a restaurant's rating on Zomato (Bangalore) based on its cost, cuisine, location, service type, and other listing details — cleaned, analyzed, modeled, and deployed as a web app.

## Dataset

`zomato.csv` — 51,717 restaurant listings scraped from Zomato Bangalore, with columns including:

| Column | Description |
|---|---|
| `name` | Restaurant name |
| `online_order` | Whether the restaurant accepts online orders (Yes/No) |
| `book_table` | Whether table booking is available (Yes/No) |
| `rate` | Rating out of 5 (target variable) |
| `votes` | Number of votes/reviews |
| `location` | Neighborhood in Bangalore |
| `rest_type` | Restaurant type (e.g. Casual Dining, Quick Bites) |
| `cuisines` | Cuisines offered |
| `approx_cost(for two people)` | Approximate cost for two |
| `dish_liked` | Popular dishes at the restaurant |
| `listed_in(type)` | Service type (Delivery, Dine-out, Buffet, etc.) |
| `listed_in(city)` | City area the listing falls under |

## What the notebook does

`Restaurant_Deployment_Project.ipynb` covers the full workflow:

1. **Data cleaning** — drops irrelevant columns (`url`, `phone`), removes duplicates and rows with missing values, and renames columns for clarity (`approx_cost(for two people)` → `cost`, `listed_in(type)` → `type`, `listed_in(city)` → `city`).
2. **Feature cleanup** — strips commas from `cost` and converts it to numeric; removes `"NEW"` placeholder ratings and strips the `/5` suffix from `rate`, converting it to numeric.
3. **Exploratory data analysis** — most popular restaurant chains, online ordering and table booking breakdowns, rating distribution and grouped rating bands, service type counts, cost distribution (box plot and density plot), and most-liked dishes parsed from the `dish_liked` column.
4. **Encoding** — maps `online_order`/`book_table` to 0/1, and label-encodes `location`, `rest_type`, `cuisines`, and `menu_item`.
5. **Model training** — trains and compares three regressors to predict `rate`:
   - Linear Regression
   - Random Forest Regressor
   - Extra Trees Regressor
6. **Model export** — serializes the best model (`ExtraTreesRegressor`) to `model.pkl`, and saves the processed feature set to `Zomato_df.csv` for deployment.

## Results

| Model | R² Score |
|---|---|
| Linear Regression | 0.228 |
| Random Forest Regressor | 0.881 |
| Extra Trees Regressor | 0.933 |

The Extra Trees Regressor performs best and is the model used for deployment.

## Deployment

The `my_project` folder contains a Flask web app that serves the trained model:

- `app.py` — Flask application and prediction route
- `model.py` — model loading/prediction helper logic
- `model.pkl` — serialized trained model
- `Zomato_df.csv` — processed feature set used by the app
- `static/`, `templates/` — front-end assets and HTML for the input form

## Tech stack

- Python
- pandas, numpy
- matplotlib, seaborn, plotly
- scikit-learn (`LinearRegression`, `RandomForestRegressor`, `ExtraTreesRegressor`, `LabelEncoder`, `train_test_split`, `r2_score`)
- Flask (deployment)
- pickle (model serialization)

## Getting started

```bash
pip install pandas numpy matplotlib seaborn plotly scikit-learn jupyter
jupyter notebook Restaurant_Deployment_Project.ipynb
```

The notebook expects `zomato.csv` in the working directory — update the file path in the early cells if you place it elsewhere.

To run the deployed app locally:

```bash
cd my_project
python app.py
```

## Project structure

```
.
├── my_project/                      # Flask app for serving predictions
│   ├── app.py
│   ├── model.py
│   ├── model.pkl
│   ├── Zomato_df.csv
│   ├── static/
│   └── templates/
├── Restaurant_Deployment_Project.ipynb   # Main EDA + modeling notebook
├── zomato.csv                       # Source dataset
└── README.md
```
