# Housing-Prices-Kaggle-Competition

This is my first kaggle competition that I'm using to learn self learn machine learning.

First thing I did was to Impute missing data. For categorical data I used the "most-frequent" to impute the data and for numerical I used "mean" to impute the data.
Once I had imputed the data I needed one-hot encoded the categorical columns with 10 or less categories. If they had more than 10 categories I droped them from the dataset to keep the dataset a useable size.
The above cleaned te data to be usuable by a machine leanring model. I chose to use the XGBoost model with a max "n_estimator" of 10000 and a "learning_rate" of 0.01.

I then split the train data into parts using 0.8 for training and 0.2 for validating. Using the XGBoost model I got a average mean error of $14421 when predicting the validation data.


My submission to the kaggle competition had an average error of $14995.11723
