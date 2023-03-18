# King-County-Housing-Prices
"King County Housing Prices Project" uses a neural network to guess the prices of the houses in the region up to 2014.

Project consists of two parts:

Part 1: Exploratory Data Analysis
- Created simple graphs using visualization libraries for Python such as SeaBorn and MatPlotLib.
- Got rid of the parameters that could reduce the performance of the neural network such as owner id and zipcode.
- Converted the date of negotiation into a pd.datetime object to simplify the date seperation
- Used the latitude and longitude values to make a scatter plot of the area based on house prices.

Part 2: Training the Neural Network
- Using the SkLearn's MinMaxScaler, scaled the train and test data (MinMaxScaler is commonly used and suitable for deep neural networks)
- Experimented with the number of hidden layers to improve performance (stuck around 75% accuracy, could use more experimenting)
- Obtained a "mean absolute error" of 96000$, not that bad since we are on the scale of millions. 

UPDATE!
- Tuned the model to avoid overshooting, tuned model performs significantly better.

The dataset used in the project: https://www.kaggle.com/datasets/harlfoxem/housesalesprediction
