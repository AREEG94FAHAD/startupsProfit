# startupsProfit
A multi-linear regression system to estimate the profit of startups in three states in America.


#### Independent variables are
- R&D Spend
- Administration
- Marketing Spend
- State,Profit

#### Dependent variables include 
- Profit


#### This project's dataset is available for download at this link.  
[Dataset](https://raw.githubusercontent.com/arib168/data/main/50_Startups.csv)

#### Tools
To work with this project, multiple libraries and frameworks need to be installed. The following is a list of them.

- [Pandas](https://pandas.pydata.org/)
- [Numpy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)
- [scikit-learn](https://scikit-learn.org/stable/)


#### Code implemention 
1- Import the main packets for the project.
```
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
```

2- Take a look at the dataset

```
df = pd.read_csv("dataset.csv")
```

3- Select the columns for training and testing.

a- Select the first four columns as independent variables (x).
```
x = df.iloc[:, :4]
```


b- Select the last column as a dependent variable (y).

```
y = df.iloc[:, 4]
```

4- The categorical variable is converted using onehotencoder, and the reset is performed using standardScaler. 

```
col_transformer = make_column_transformer((OneHotEncoder(), ['State']),remainder=StandardScaler())

x = col_transformer.fit_transform(x)
```



5- Select 80% for training and 20% for testing.

```
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
```

6- call the linear regression and compute the results.
```
linreg=LinearRegression()
linreg.fit(x_train,y_train)

y_pred = linreg.predict(x_test)
print(y_pred)
```

7- Compute the Accuracy
```
Accuracy=r2_score(y_test,y_pred)*100
print(" Accuracy of the model is %.2f" %Accuracy)

plt.scatter(y_test,y_pred)
plt.xlabel('Actual')
plt.ylabel('Predicted')
```
8- export the model

```
import joblib

import os

if not os.path.exists('Model'):
        os.mkdir('Model')
if not os.path.exists('Scaler'):
        os.mkdir('Scaler')

joblib.dump(model, r'Model/model.pickle') 
joblib.dump(col_transformer, r'Scaler/scaler.pickle')
```

9- Using the exported model for new data

```
import joblib
model = joblib.load(r'Model/model.pickle')
scaler = joblib.load(r'Scaler/scaler.pickle')


import pandas as pd
new_data = pd.DataFrame([{'R&D Spend':'155349.2', 'Administration':'126897.8', 'Marketing Spend':'471784.1', 'State':'New York'},{'R&D Spend':'165349.2', 'Administration':'136897.8', 'Marketing Spend':'471784.1', 'State':'New York'}])

# new_data = pd.DataFrame([])


new_data = new_data[['R&D Spend','Administration','Marketing Spend','State']]

print(new_data)

# Normalize and add the missing values automatically
new_data = scaler.transform(new_data)

pre = model.predict(new_data)

print(pre)
```




Results
<hr>

![delay](https://user-images.githubusercontent.com/30151596/201482255-53bc6945-a384-4524-b5cc-734b818b8036.png)
![Figure_1](https://user-images.githubusercontent.com/30151596/201482262-7b3766f3-0dd1-41dc-acb5-4c9b8fcf21c8.png)

