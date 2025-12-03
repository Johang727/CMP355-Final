import pandas as pd
import os, time, datetime, sys
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_percentage_error
import numpy as np

# variables
# --------------------------------
start = time.time()

DATA_FOLDER:str = "data/"
RANDOM_STATE:int = 136
TEST_SIZE:float = 0.1
SR_BINS:list[int] = [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000, 16000, 17000, 18000]
SR_LABELS:list[str] = ['<1K SR', '1-2K SR', '2-3K SR', '3-4K SR', '4-5K SR', '5-6K SR', '6-7K SR', '7-8K SR', '8-9K SR', '9-10K SR', '10-11K SR', '11-12K SR', '12-13K SR', '13-14K SR', '14-15K SR', '15-16K SR', '16-17K SR', '17K+ SR']
RETRY_SEARCH:bool = False


dataframe_list:list = []
rmse:list[float] = []
r2:list[float] = []
mape:list[float] = []
models:list = []
predictions:list = []

# --------------------------------

# merging CSVs from the data folder
# --------------------------------

print(f"Merging CSV files in {DATA_FOLDER}")

for fn in os.listdir(DATA_FOLDER):
    if fn.endswith(".csv"):
        fp = os.path.join(DATA_FOLDER, fn)
        df = pd.read_csv(fp, sep=";")
        dataframe_list.append(df)
master_dataframe:pd.DataFrame = pd.concat(dataframe_list, ignore_index=True)
# --------------------------------

# dropping outliers
# --------------------------------

old_instances:int = len(master_dataframe)

iqr_mask = (master_dataframe["APM"] > 0.0) # what if i just get rid of the iqr


master_dataframe = master_dataframe[iqr_mask]

print(f"Outliers Removed: {len(master_dataframe) - old_instances}")

# data counting
# --------------------------------

print(f"Instances: {len(master_dataframe)}")

master_dataframe["SRBins"] = pd.cut(master_dataframe["SR"], bins=SR_BINS, labels=SR_LABELS, right=False)

sr_counts:dict[str, int] = master_dataframe["SRBins"].value_counts().sort_index().to_dict()

for key, value in sr_counts.items():
    print(f"{key} has {value} instances")

# --------------------------------  

# make app (it tends to predict lower ratings better this way)
# --------------------------------

master_dataframe["APP"] = master_dataframe["APM"] / master_dataframe['DPM']

# --------------------------------

# data splitting
# --------------------------------

x = master_dataframe[["Date", "DPM", "APM", "APP"]]
y = master_dataframe["SR"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=master_dataframe["SRBins"])

print("\nTraining SR:")
print(f"  Min SR: {np.min(y_train)}")
print(f"  Max SR: {np.max(y_train)}")
print(f"  Mean SR: {np.mean(y_train):.2f}")
print(f"  Median SR: {np.median(y_train):.2f}")
print(f"{len(y_train)} instances.")

print("----")

print("\nTesting SR SR:")
print(f"  Min SR: {np.min(y_test)}")
print(f"  Max SR: {np.max(y_test)}")
print(f"  Mean SR: {np.mean(y_test):.2f}")
print(f"  Median SR: {np.median(y_test)}")
print(f"{len(y_test)} instances.")

# create random forest 
# --------------------------------

if RETRY_SEARCH:
    print("Training many Random Forests to see which is best...")
    # essentially try a ton of things for our random forest and see which is best

    # first run gave me depth:10, features:sqrt, min_samples_split: 2, n_estimators: 50
    # second run which took 20m gave me depth: None, features, sqrt, min_samples: 3, estimators: 79

    # Best Parameters: {'max_depth': None, 'max_features': 'sqrt', 'min_samples_split': 3, 'n_estimators': 79}

    param_grid = {
        'n_estimators': [x for x in range(45, 86, 1)],
        'max_depth': [10, 20, None],
        'min_samples_split': [2,3,4],
        'max_features': ["sqrt"]
    }

    rf = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)

    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        verbose=3
    )

    grid_search.fit(x_train, y_train)

    best_params = grid_search.best_params_

    print(f"Best Parameters: {best_params}")

    best_rf = grid_search.best_estimator_

    models.append(best_rf)
else:
    models.append(RandomForestRegressor(n_jobs=-1, 
                                        random_state=RANDOM_STATE, 
                                        max_depth=None,
                                        max_features="sqrt",
                                        min_samples_split=3,
                                        n_estimators=79
                                        ))


# create the Linear:
# --------------------------------
models.append(LinearRegression(n_jobs=-1))

# create Gradient Boost model:
# --------------------------------

if RETRY_SEARCH:

    # run 1
    # Best Parameters: {'alpha': 0.5, 'learning_rate': 0.01, 'loss': 'squared_error', 'max_depth': 10, 'max_features': 'sqrt', 'min_impurity_decrease': 1.0, 'min_samples_split': 3, 'n_estimators': 500, 'subsample': 0.8}

    # run 2
    # Best Parameters: {'alpha': 0.5, 'learning_rate': 0.005, 'loss': 'squared_error', 'max_depth': 8, 'max_features': 'sqrt', 'min_impurity_decrease': 2.0, 'min_samples_split': 3, 'n_estimators': 1000, 'subsample': 0.5}
    # this one took like 5 hours

    # run 3
    # Best Parameters: {'learning_rate': 0.006, 'loss': 'squared_error', 'max_depth': 7, 'max_features': 'sqrt', 'min_impurity_decrease': 1.75, 'min_samples_split': 3, 'n_estimators': 900, 'subsample': 0.75}
    # Root Mean Squared Error: 712.57

    param_grid_gb = {
        'learning_rate':[0.0055, 0.006, 0.007],
        'n_estimators': [850, 900, 950],
        'max_depth': [6, 7, 8],
        'min_samples_split': [3],
        'max_features': ["sqrt"],
        'min_impurity_decrease':[1.5, 1.75, 1.9],
        'loss':["squared_error"],
        'subsample':[0.6, 0.75, 0.8],
    }

    gb = GradientBoostingRegressor(random_state=RANDOM_STATE)

    grid_search_gb = GridSearchCV(
        estimator=gb,
        param_grid=param_grid_gb,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        verbose=3
    )

    grid_search_gb.fit(xTrain, yTrain)


    best_params = grid_search_gb.best_params_

    print(f"Best Parameters: {best_params}")

    best_gb = grid_search_gb.best_estimator_


    models.append(best_gb)
else:
    models.append(GradientBoostingRegressor(
        learning_rate=0.006,
        loss="squared_error",
        max_depth=7,
        min_samples_split=3,
        max_features="sqrt",
        min_impurity_decrease=1.75,
        n_estimators=900,
        subsample=0.75,
        random_state=RANDOM_STATE
    ))

# create ensemble of the two trees, these are really good at interpolation, not extra.
# --------------------------------

if RETRY_SEARCH:
    sys.exit(0) # i dont think we can make an ensemble with already fitted models?

models.append(VotingRegressor(
    estimators=[
        ("rf", models[0]),
        ("gb", models[2])
    ],
    n_jobs=-1
))



models[0].fit(x_train, y_train) # Random Forest
models[1].fit(x_train, y_train) # Linear
models[2].fit(x_train, y_train) # Gradient Boosting
models[3].fit(x_train, y_train) # Ensemble

# test random forest
# --------------------------------

print("\n----\nTesting Random Forest\n")


predictions.append(models[0].predict(x_test))
rmse.append(root_mean_squared_error(y_test, predictions[0]))
r2.append(r2_score(y_test, predictions[0]))
mape.append(mean_absolute_percentage_error(y_test, predictions[0]))

print(f"Root Mean Squared Error: {rmse[0]:.2f}")
print(f"R-squared: {r2[0]:.4f}")
print(f"Mean Absolute Percentage Error: {mape[0]*100:.2f}%")
# --------------------------------

# test linear 
# --------------------------------

print("\n----\nTesting Linear\n")

predictions.append(models[1].predict(x_test))

rmse.append(root_mean_squared_error(y_test, predictions[1]))
r2.append(r2_score(y_test, predictions[1]))
mape.append(mean_absolute_percentage_error(y_test, predictions[1]))

print(f"Root Mean Squared Error: {rmse[1]:.2f}")
print(f"R-squared: {r2[1]:.4f}")
print(f"Mean Absolute Percentage Error: {mape[1]*100:.2f}%")

# test Gradient Boosting Regressor
# --------------------------------  

print("\n----\nTesting Gradient Boosting\n")

predictions.append(models[2].predict(x_test))

rmse.append(root_mean_squared_error(y_test, predictions[2]))
r2.append(r2_score(y_test, predictions[2]))
mape.append(mean_absolute_percentage_error(y_test, predictions[2]))


print(f"Root Mean Squared Error: {rmse[2]:.2f}")
print(f"R-squared: {r2[2]:.4f}")
print(f"Mean Absolute Percentage Error: {mape[2]*100:.2f}%")

# --------------------------------

# test All Ensemble
# --------------------------------

print("\n----\nTesting RF + GB\n")

predictions.append(models[3].predict(x_test))

rmse.append(root_mean_squared_error(y_test, predictions[3]))
r2.append(r2_score(y_test, predictions[3]))
mape.append(mean_absolute_percentage_error(y_test, predictions[3]))


print(f"Root Mean Squared Error: {rmse[3]:.2f}")
print(f"R-squared: {r2[3]:.4f}")
print(f"Mean Absolute Percentage Error: {mape[3]*100:.2f}%")

# --------------------------------

# make predictions
# --------------------------------

# after playing a bot match... my stats were
# 94.9 DPM
# 76.1 APM

# My rating is : 11257


day:int = 45994
dpm:float = 94.9
apm:float = 76.1
app:float = apm/dpm

# data pred, mid-rank

in_data = pd.DataFrame([[day, dpm, apm, app]])

print("Real world tests:")

print(f"Linear: {round(models[1].predict(in_data)[0])}")

tp = [tree.predict(in_data) for tree in models[0].estimators_]
print(f"Random Forest: {round(np.mean(tp))} ± {round(np.std(tp))}")

print(f"Gradient Boosting: {round(models[2].predict(in_data)[0])}")

print(f"Random Forest + Gradient Boosting: {round(models[3].predict(in_data)[0])}")


# data pred, low rank

dpm:float = 24.4
apm:float = 11.0
app:float = apm/dpm

in_data = pd.DataFrame([[day, dpm, apm, app]])

print("Real world tests:")

print(f"Linear: {round(models[1].predict(in_data)[0])}")

tp = [tree.predict(in_data) for tree in models[0].estimators_]
print(f"Random Forest: {round(np.mean(tp))} ± {round(np.std(tp))}")

print(f"Gradient Boosting: {round(models[2].predict(in_data)[0])}")

print(f"Random Forest + Gradient Boosting: {round(models[3].predict(in_data)[0])}")


end = time.time()
print(f"Runtime: {end-start:.2f}s")