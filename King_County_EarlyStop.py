####################
#REDO THE ORIGINAL MODEL TO FIND OPTIMAL VAL_LOSS
####################

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import mean_absolute_error, explained_variance_score
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

modified_df = pd.read_csv("kc_house_data_modified.csv")


X = modified_df.drop("price", axis=1).values
y = modified_df["price"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

nn_model = Sequential()
earlyStop = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=100)

nn_model.add(Dense(units=19, activation="relu"))
nn_model.add(Dense(units=19, activation="relu"))
nn_model.add(Dense(units=19, activation="relu"))
nn_model.add(Dense(units=19, activation="relu"))
nn_model.add(Dense(units=19, activation="relu"))
nn_model.add(Dense(units=19, activation="relu"))
nn_model.add(Dense(units=1))

nn_model.compile(optimizer="adam", loss="mse")
nn_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=2000, callbacks=[earlyStop])
losses = pd.DataFrame(nn_model.history.history)
losses.plot()
plt.show()

preds = nn_model.predict(X_test)
#preds.shape = (6415,1)
preds = pd.Series(preds.reshape(6415,))

pred_df = pd.DataFrame(y_test, columns=["Test True Y"])
pred_df = pd.concat([pred_df, preds], axis=1)
pred_df.columns = ["Test True Y", "Predictions"]

mae = mean_absolute_error(y_true=pred_df["Test True Y"], y_pred=pred_df["Predictions"])             ##about 65000$
exp_var = explained_variance_score(y_true=pred_df["Test True Y"], y_pred=pred_df["Predictions"])    ##about 88% accuracy for 500 epochs, 88% for 2000. Significantly better than untuned model.



