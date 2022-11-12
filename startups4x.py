from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
#import seaborn as sns

from timeit import default_timer as timer
start = timer()
# take a look at the dataset
df = pd.read_csv("dataset.csv")
# print(df.columns)

# summarize the data
# print(df.describe())

# select first four colums as independed variables
x = df.iloc[:, :4]

# select the last colums as depended variable
y = df.iloc[:, 4]


# ss = StandardScaler()
# ss.fit(x)
# x = ss.transform(x)

# convert the categorical variable using onehotencoder and the reset using standardScaler 
col_transformer = make_column_transformer((OneHotEncoder(), ['State']),remainder=StandardScaler())

x = col_transformer.fit_transform(x)

# print(X)

# select 80% for training and 20% for testing
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


#shapes of splitted data
print("X_train:",x_train.shape)
print("X_test:",x_test.shape)
print("Y_train:",y_train.shape)
print("Y_test:",y_test.shape)


linreg=LinearRegression()
model = linreg.fit(x_train,y_train)


y_pred = linreg.predict(x_test)
print(y_pred)

Accuracy=r2_score(y_test,y_pred)*100
print(" Accuracy of the model is %.2f" %Accuracy)

end = timer()

print(end - start)
plt.scatter(y_test,y_pred)
plt.xlabel('Actual')
plt.ylabel('Predicted')
# plt.show()


# from sklearn.setup import job

# import os

# if not os.path.exists('Model'):
#         os.mkdir('Model')
# if not os.path.exists('Scaler'):
#         os.mkdir('Scaler')

# joblib.dump(model, r'Model/model.pickle') 
# joblib.dump(col_transformer, r'Scaler/scaler.pickle')

