import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

train_dataframe = pd.read_csv("given_data/train.csv")
test_dataframe = pd.read_csv("given_data/test.csv")
test_data = test_dataframe.drop(['Id'], axis=1)

y_train_full = train_dataframe.SalePrice

X_train_dataframe = train_dataframe.drop(['SalePrice', 'Id'], axis=1)

numeric_columns = [col for col in X_train_dataframe.columns if train_dataframe[col].dtype in ['int64', 'float64']]
low_cardinality_col = [col for col in X_train_dataframe.columns if train_dataframe[col].dtype == "object" and train_dataframe[col].nunique() <= 10]

###### categorical imputer

#Train categorial
X_categorical_train  = X_train_dataframe[low_cardinality_col]
categorical_imputer = SimpleImputer(strategy="most_frequent")
X_categorical_train = pd.DataFrame(categorical_imputer.fit_transform(X_categorical_train ))
X_categorical_train.columns = low_cardinality_col

#Test categorial
X_categorical_test  = test_data[low_cardinality_col]
X_categorical_test = pd.DataFrame(categorical_imputer.transform(X_categorical_test))
X_categorical_test.columns = low_cardinality_col

#One Hot Encode the imputer
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
X_categorical_OneHot_train = pd.DataFrame(OH_encoder.fit_transform(X_categorical_train))
X_categorical_OneHOT_test = pd.DataFrame(OH_encoder.transform(X_categorical_test))

# ####### numerical imputer

# Train numerical
X_numerical_train = X_train_dataframe[numeric_columns]
numerical_imputer = SimpleImputer(strategy="mean")
X_numerical_train = pd.DataFrame(numerical_imputer.fit_transform(X_numerical_train))
X_numerical_train.columns = numeric_columns

# Test numerical
X_numerical_test = test_data[numeric_columns]
X_numerical_test = pd.DataFrame(numerical_imputer.transform(X_numerical_test))
X_numerical_test.columns = numeric_columns

# ##### Putting together
X_train_full = pd.concat([X_categorical_OneHot_train, X_numerical_train], axis=1)
X_test_full = pd.concat([X_categorical_OneHOT_test, X_numerical_test], axis=1)

# X_train_full.to_csv("C:\\Users\\jesse\\Documents\\Stuff\\CodeThings\\KaggleHouseLearnComp\\X_train_data.csv", index=False)

X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, random_state=1, train_size=0.8, test_size=0.2)

XGBmodeltrain = XGBRegressor(n_estimators=10000, learning_rate= 0.01) # Your code here

# Fit the model
XGBmodeltrain.fit(X_train,y_train, early_stopping_rounds=10, eval_set=[(X_valid,y_valid)])

valPredictions = XGBmodeltrain.predict(X_valid)

meanError = mean_absolute_error(y_valid, valPredictions)

print("Mean Error: ", meanError)

submissionPred = XGBmodeltrain.predict(X_test_full)
submissionDataframe = pd.DataFrame(
    {
        "Id": test_dataframe.Id,
        "SalePrice": submissionPred
    }
)

submissionDataframe.to_csv("C:\\Users\\jesse\\Documents\\Stuff\\CodeThings\\KaggleHouseLearnComp\\SubmissionPredictions.csv", index=False)