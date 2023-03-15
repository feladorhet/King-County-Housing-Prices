import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense
from keras.models import Sequential
from sklearn.metrics import mean_absolute_error, explained_variance_score

modified_df = pd.read_csv("kc_house_data_modified.csv")

#modified_df.columns = ['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors','waterfront', 'view', 'condition', 'grade', 'sqft_above','sqft_basement', 'yr_built', 'yr_renovated', 'lat', 'long','sqft_living15', 'sqft_lot15', 'year', 'month']

X = modified_df.drop("price", axis=1).values
y = modified_df["price"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)

####TODOS
#Create the MinMaxScaler() instance since it is more useful for neural networks.
#Then estimate the transformation parameters from data and then transform it.
#Then transform (but dont fit) the test set.
#Create the neural network model 

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#X_train.shape is (14966, 19), so it makes sense to have 19 neurons. Experiment with layer count.

nn_model = Sequential()

nn_model.add(Dense(units=19, activation="relu"))
nn_model.add(Dense(units=19, activation="relu"))
nn_model.add(Dense(units=19, activation="relu"))
nn_model.add(Dense(units=19, activation="relu"))
nn_model.add(Dense(units=19, activation="relu"))
nn_model.add(Dense(units=19, activation="relu"))
nn_model.add(Dense(units=1))

nn_model.compile(optimizer="adam", loss="mse")

nn_model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=128, epochs=500)

#################
#NOW TO EVALUATE THE MODEL
#################

preds = nn_model.predict(X_test)
#preds.shape = (6415,1)
preds = pd.Series(preds.reshape(6415,))

pred_df = pd.DataFrame(y_test, columns=["Test True Y"])
pred_df = pd.concat([pred_df, preds], axis=1)
pred_df.columns = ["Test True Y", "Predictions"]

#sns.lmplot(data=pred_df, x="Test True Y", y="Predictions")
#plt.show()

mae = mean_absolute_error(y_true=pred_df["Test True Y"], y_pred=pred_df["Predictions"])             ##about 96000$
exp_var = explained_variance_score(y_true=pred_df["Test True Y"], y_pred=pred_df["Predictions"])    ##about 76% accuracy, not great not terrible

##to check, try and guess the price of house[0]
new_house = modified_df.drop("price", axis=1).iloc[0]
new_house = scaler.transform(new_house.values.reshape(-1,19))

newhouse_price = nn_model.predict(new_house) ##actual price of house is 1970000, guessed value is 1109171. Overshooting occurs.


