import numpy as np
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Loading the dataset and creating a DataFrame
data=datasets.load_boston()
df=pd.DataFrame(data.data)
df['target']=data.target

# Splitting the data into training and testing sets
X_train=df[5][:-20]
X_test=df[5][-20:]



# Splitting the target into training and testing sets
Y_train=df['target'][:-20]

Y_test=df['target'][-20:]

# Calculating mean
X_mean=np.mean(X_train)
Y_mean=np.mean(Y_train)

length=len(X_train)

num=0
deno=0

for i in range(length):
    num=num+((X_train[i]-X_mean)*(Y_train[i]-Y_mean))
    deno=deno+(X_train[i]-X_mean)**2

# Calculating the values of coeff0 and coeff1 in Y=coeff0+coeff1*X
coeff1=num/deno
coeff0=Y_mean-(coeff1*X_mean)

# Making the prediction
pred=coeff0+coeff1*X_test


# Plotting a line
plt.plot(X_test, pred, color='#58b970', label='Regression Line')

# Plotting Scatter points
plt.scatter(X_test, Y_test, c='#ef5423', label='Scatter Plot')

plt.legend()
plt.show()
print("mean squared error is")
print(mean_squared_error(pred,Y_test))