#%% md
#The dataset used in this notebook could be found on this link: https://archive.ics.uci.edu/ml/datasets/Air+Quality
#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#%matplotlib inline
#%% md
"""

Atributes info:

    0 Date	(DD/MM/YYYY) 
    1 Time	(HH.MM.SS) 
    2 True hourly averaged concentration CO in mg/m^3 (reference analyzer) 
    3 PT08.S1 (tin oxide) hourly averaged sensor response (nominally CO targeted)	
    4 True hourly averaged overall Non Metanic HydroCarbons concentration in microg/m^3 (reference analyzer) 
    5 True hourly averaged Benzene concentration in microg/m^3 (reference analyzer) 
    6 PT08.S2 (titania) hourly averaged sensor response (nominally NMHC targeted)	
    7 True hourly averaged NOx concentration in ppb (reference analyzer) 
    8 PT08.S3 (tungsten oxide) hourly averaged sensor response (nominally NOx targeted) 
    9 True hourly averaged NO2 concentration in microg/m^3 (reference analyzer)	
    10 PT08.S4 (tungsten oxide) hourly averaged sensor response (nominally NO2 targeted)	
    11 PT08.S5 (indium oxide) hourly averaged sensor response (nominally O3 targeted) 
    12 Temperature in Â°C	
    13 Relative Humidity (%) 
    14 AH Absolute Humidity """
#%%
air_data = pd.read_csv(r'G:\Projects\Python\PyCharmProjects\regression-python-air\Air Quality - Machine Learning\data\AirQualityUCI.csv',sep=';')
#%%
air_data.head()
#%%
air_data.shape
#%%
import seaborn as sns
from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()
sns.set(style='whitegrid')
features_plot = ['C6H6(GT)', 'RH', 'AH', 'PT08.S1(CO)']
#%%
data_to_plot = air_data[features_plot]
data_to_plot = scalar.fit_transform(data_to_plot)
data_to_plot = pd.DataFrame(data_to_plot)
#%%
sns.pairplot(data_to_plot, size=2.0)
plt.tight_layout()
plt.show()
#%% md
## Step 1. Preprocessing data
#%%
air_data.dropna(axis=0, how='all')
#%% md
## Step 2. Features vs Labels
#%%
features = air_data
#%%
features = features.drop('Date', axis='columns')
features = features.drop('Time', axis='columns')
features = features.drop('C6H6(GT)', axis='columns')
features = features.drop('PT08.S4(NO2)', axis='columns')
#%%
labels = air_data['C6H6(GT)'].values
#%%
features = features.values
#%% md
## Step 3. Train and test portions
#%%
from sklearn.model_selection import train_test_split
#%%
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3)
#%%
print("X_trian shape --> {}".format(X_train.shape))
print("y_train shape --> {}".format(y_train.shape))
print("X_test shape --> {}".format(X_test.shape))
print("y_test shape --> {}".format(y_test.shape))
#%% md
## Step 4. Regression
#%% md
### Step 4.1 Linear Regression
#%%
from sklearn.linear_model import LinearRegression
#%%
regressor = LinearRegression()
regressor.fit(X_train, y_train)
#%%
print("Predicted values:", regressor.predict(X_test))
#%%
print("R^2 score for liner regression: ", regressor.score(X_test, y_test))
#%% md
### Step 4.2  Support Vector Regression
#%%
from sklearn.svm import SVR
#%%
support_regressor = SVR(kernel='rbf', C=1000)
support_regressor.fit(X_train, y_train)
#%%
print("Coefficient of determination R^2 <-- on train set: {}".format(support_regressor.score(X_train, y_train)))
#%%
print("Coefficient of determination R^2 <-- on test set: {}".format(support_regressor.score(X_test, y_test)))
#%% md
### Step 4.3 Decision tree regression
#%%
from sklearn.tree import DecisionTreeRegressor
#%%
dtr = DecisionTreeRegressor()
dtr.fit(X_train, y_train)
#%%
print("Coefficient of determination R^2 <-- on train set: {}".format(dtr.score(X_train, y_train)))
#%%
print("Coefficient of determination R^2 <-- on test set: {}".format(dtr.score(X_test, y_test)))
#%% md
### Step 4.4 Lasso regression
#%%
from sklearn.linear_model import Lasso
#%%
indiana_jones = Lasso(alpha=1.0)
indiana_jones.fit(X_train, y_train)
#%%
print("Coefficient of determination R^2 <-- on train set : {}".format(indiana_jones.score(X_train, y_train)))
#%%
print("Coefficient of determination R^2 <-- on test set: {}".format(indiana_jones.score(X_test, y_test)))

#%% md
## Step 5. Feature selection
#%%
from sklearn.ensemble import ExtraTreesRegressor
#%%
etr = ExtraTreesRegressor(n_estimators=300)
etr.fit(X_train, y_train)
#%%
print(etr.feature_importances_)
indecis = np.argsort(etr.feature_importances_)[::-1]
#%%
plt.figure(num=None, figsize=(14, 10), dpi=80, facecolor='w')
plt.title("Feature importances")
plt.bar(range(X_train.shape[1]), etr.feature_importances_[indecis],
       color="r", align="center")
plt.xticks(range(X_train.shape[1]), indecis)
plt.show()