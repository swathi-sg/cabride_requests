import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt

seg=pd.read_csv('/home/swat/Desktop/seg.csv')

seg=seg.sort_values(['T_block','Loc_Cluster']).reset_index(drop=True)

d1=seg[(seg['T_block']==1) & (seg['Loc_Cluster']==0)].reset_index(drop=True)

d1['Date'] = pd.to_datetime(d1.Date , format = '%Y-%m-%d')
data = d1.drop(['Date','T_block','Loc_Cluster'], axis=1)
data.index = d1.Date

#creating the train and validation set
train = data[:int(0.8*(len(data)))]
valid = data[int(0.8*(len(data))):]

#fit the model
from statsmodels.tsa.vector_ar.var_model import VAR

model = VAR(endog=train)
model_fit = model.fit()

# make prediction on validation
prediction = model_fit.forecast(model_fit.y, steps=len(valid))

#converting predictions to dataframe
cols = data.columns
pred = pd.DataFrame(index=valid.index,columns=[cols])
for j in range(0,2):
    for i in range(0, len(prediction)):
       pred.iloc[i][j] = prediction[i][j]

#check rmse
for i in cols:
    print('rmse value for', i, 'is : ', round(sqrt(mean_squared_error(pred[i], valid[i])),2))

#make final predictions
model = VAR(endog=data)
model_fit = model.fit()
yhat = model_fit.forecast(model_fit.y, steps=5)
print(yhat)

plt.plot(train['Count'])
plt.plot(pred['Count'])
plt.plot(valid['Count'])
plt.xlabel('Date')
plt.ylabel('Count')
plt.title('No of rides requested')
plt.legend()
#plt.savefig('3.png')
plt.show()

plt.plot(train['Distance'])
plt.plot(pred['Distance'])
plt.plot(valid['Distance'])
plt.xlabel('Date')
plt.ylabel('Distance')
plt.title('Total distance covered in rides')
plt.legend()
#plt.savefig('4.png')
plt.show()


