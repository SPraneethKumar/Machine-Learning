import quandl
import numpy as np
import matplotlib.pyplot as plt
import datetime


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing,svm
df = quandl.get("WIKI/AMZN")


forecast_out = int(30) # predicting 30 days into future
df['Prediction'] = df[['Adj. Close']].shift(-forecast_out) #label column with data shifted 30 units up
X = np.array(df.drop(['Prediction'], 1))
X = preprocessing.scale(X)

X_forecast = X[-forecast_out:] # set X_forecast equal to last 30
X = X[:-forecast_out] # remove last 30 from X
y = np.array(df['Prediction'])
y = y[:-forecast_out]  #remove last 30 and put the rest in y


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)  #splitting the data into training and testing sets based on test_size


clf = LinearRegression() #create linear regression object
clf.fit(X_train,y_train) #train the model using the training set
# Testing
mean_squared_error = clf.score(X_test, y_test)
print("mean_squared_error: ", mean_squared_error)

forecast_prediction = clf.predict(X_forecast) #make predictions using test datad
print("Forecasted Values are:")
print((forecast_prediction))

df['Forecast'] = np.nan  #creating a new column
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_prediction:    #to add dates to predicted values
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += 86400
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Dates')
plt.ylabel('Price')
plt.show()







