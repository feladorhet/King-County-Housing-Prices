############################
#EXPLORATORY DATA ANALYSIS ON THE "HOUSE SALES IN KING COUNTY/SEATTLE, USA" DATASET FROM KAGGLE.COM *(1)
############################

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

base_df = pd.read_csv("kc_house_data_RAW.csv")

#base_df.columns = ['id', 'date', 'price', 'bedrooms', 'bathrooms', 'sqft_living','sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade','sqft_above','sqft_basement', 'yr_built', 'yr_renovated', 'zipcode','lat', 'long', 'sqft_living15', 'sqft_lot15']

#base_df.info()
#<class 'pandas.core.frame.DataFrame'>
#RangeIndex: 21597 entries, 0 to 21596
#Data columns (total 21 columns):
# #   Column         Non-Null Count  Dtype  
#---  ------         --------------  -----  
# 0   id             21597 non-null  int64  
# 1   date           21597 non-null  object 
# 2   price          21597 non-null  float64
# 3   bedrooms       21597 non-null  int64  
# 4   bathrooms      21597 non-null  float64
# 5   sqft_living    21597 non-null  int64  
# 6   sqft_lot       21597 non-null  int64  
# 7   floors         21597 non-null  float64
# 8   waterfront     21597 non-null  int64  
# 9   view           21597 non-null  int64  
# 10  condition      21597 non-null  int64  
# 11  grade          21597 non-null  int64  
# 12  sqft_above     21597 non-null  int64  
# 13  sqft_basement  21597 non-null  int64  
# 14  yr_built       21597 non-null  int64  
# 15  yr_renovated   21597 non-null  int64  
# 16  zipcode        21597 non-null  int64  
# 17  lat            21597 non-null  float64
# 18  long           21597 non-null  float64
# 19  sqft_living15  21597 non-null  int64  
# 20  sqft_lot15     21597 non-null  int64  
#dtypes: float64(5), int64(15), object(1)
#memory usage: 3.5+ MB

##THERE IS NO MISSING DATA, PERFECTLY FILLED

##DISTPLOT FOR THE PRICE DISTRIBUTION
sns.distplot(base_df["price"])
plt.title("Density distribution of the house prices")
plt.xlabel("House Price (x1.000.000)")
plt.ylabel("Density")
#plt.show()

##ROOM COUNT
sns.countplot(data=base_df, x="bedrooms")
plt.title("Bedroom count for houses")
plt.ylabel("House count")
#plt.show()

##SEE THE CORRELATION BETWEEN COLUMNS
cmap = sns.cm.rocket_r
sns.heatmap(data=base_df.corr(), cmap=cmap)
plt.title("Correlation between columns")
#plt.show()

price_factors = base_df.corr()["price"].sort_values()
#price_factors is as follows:
#    zipcode         -0.053402
#id              -0.016772
#long             0.022036
#condition        0.036056
#yr_built         0.053953
#sqft_lot15       0.082845
#sqft_lot         0.089876
#yr_renovated     0.126424
#floors           0.256804
#waterfront       0.266398
#lat              0.306692
#bedrooms         0.308787
#sqft_basement    0.323799
#view             0.397370
#bathrooms        0.525906
#sqft_living15    0.585241
#sqft_above       0.605368
#grade            0.667951
#sqft_living      0.701917
#price            1.000000
##THE HIGHEST CORRELATION SEEMS TO BE WITH "sqft_living"

##BOXPLOT FOR PRICE-BEDROOM COUNT
sns.boxplot(data=base_df, x="bedrooms", y="price", palette="viridis")
plt.title("Number of Bedrooms - House Price")
#plt.show()

sns.scatterplot(data=base_df, x="price", y="long")
sns.scatterplot(data=base_df, x="price", y="lat")

##THERE SEEMS TO BE AN AREA IN WHICH HOUSES ARE MORE EXPENSIVE
##THERE ARE ALSO SOME HOUSES THAT ARE EXTREMELY EXPENSIVE THAT NEEDS TO BE DROPPED OUT OF THE DATASET

#base_df.sort_values("price", ascending=False).head(20) shows that the values over 4 million are absurdly expensive, so I trim them.
#base_df.sort_values("price", ascending=True).head(20) shows that there is no such inbalance for cheaper houses. No feature engineering needed.
total_len = len(base_df) #21597 houses total
top_1_perc_dropped = base_df.sort_values("price", ascending=False).iloc[216:]  #trim 1% of the dataset to normalize it

sns.scatterplot(data=top_1_perc_dropped, x="long", y="lat", hue="price", palette="RdYlGn", edgecolor=None, alpha=0.2)
plt.title("King County House Distribution")
plt.xlabel("LONGITUDE")
plt.ylabel("LATIDUDE")
#plt.show()

#########################################################################################################
#FEATURE ENGINEERING STARTS HERE
#########################################################################################################

##DROP USELESS COLUMNS AND TURN DATE INTO "DATETIME"
top_1_perc_dropped.drop("id", axis=1, inplace=True)
top_1_perc_dropped.drop("zipcode", axis=1, inplace=True)
top_1_perc_dropped["date"] = pd.to_datetime(top_1_perc_dropped["date"])
#top_1_perc_dropped["date"] is now in the format 2014-10-13

top_1_perc_dropped["year"] = top_1_perc_dropped["date"].apply(lambda x: x.year)
top_1_perc_dropped["month"] = top_1_perc_dropped["date"].apply(lambda x: x.month)
top_1_perc_dropped.drop("date", axis=1, inplace=True)

##THE RENNOVATION YEAR IS ALSO PROBLEMTAIC SINCE SOME HAVEN'T BEEN RENNOVATED (indicated as 0). HIGHER THE RENNOVATION DATE, HIGHER THE PRICE
#base_df["yr_renovated"] varies between 2014 and 1944. Take the year 2005 to be the division point
top_1_perc_dropped["yr_renovated"] = top_1_perc_dropped["yr_renovated"].apply(lambda x: 1 if x >= 2000 else 0)

#SAME GOES FOR SQFT_BASEMENT BUT NO NEED TO MODIFY IT, SINCE 0 JUST MEANS THERE IS NO BASEMENT.

top_1_perc_dropped.to_csv("kc_house_data_modified.csv", index=False)