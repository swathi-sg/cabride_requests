import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from math import sqrt

seg=pd.read_csv('/home/swat/Desktop/seg.csv')

seg=seg.sort_values(['T_block','Loc_Cluster']).reset_index(drop=True)

d1=seg[(seg['T_block']==1) & (seg['Loc_Cluster']==0)].reset_index(drop=True)

d1['Date'] = pd.to_datetime(d1.Date , format = '%Y-%m-%d')
data = d1.drop(['Date','T_block','Loc_Cluster'], axis=1)
data.index = d1.Date

#creating the train and validation set
#train = data[:int(0.8*(len(data)))]
#valid = data[int(0.8*(len(data))):]
#------------------------------For #Rides prediction-----------------
pd.plotting.register_matplotlib_converters()
y=data['Count']
y.plot(figsize=(15, 6))
plt.show()

from pylab import rcParams
rcParams['figure.figsize'] = 18, 8

decomposition = sm.tsa.seasonal_decompose(y, model='additive')
fig = decomposition.plot()
plt.show()

p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 4) for x in list(itertools.product(p, d, q))]



for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(y,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)

            results = mod.fit()

            #print('ARIMA{}x{}4 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue
        
mod = sm.tsa.statespace.SARIMAX(y,
                                order=(1, 1, 1),
                                seasonal_order=(1, 1, 1, 4),
                                enforce_stationarity=False,
                                enforce_invertibility=False)

results = mod.fit()

#print(results.summary().tables[1])

#results.plot_diagnostics(figsize=(16, 8))
#plt.show()

pred = results.get_prediction(start=pd.to_datetime('2018-07-01'), dynamic=False)

pred_ci = pred.conf_int()

pred._results
ax = y['2018':].plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='Predicted', alpha=.7, figsize=(14, 7))

ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)

ax.set_xlabel('Date')
ax.set_ylabel('Count')
ax.set_title('No of rides requested')
plt.legend()
#plt.savefig('1.png')
plt.show()

y_forecasted = pred.predicted_mean
y_truth = y['2018-07-01':]

# Compute the mean square error
err=sqrt(mean_squared_error(y_forecasted, y_truth))
print('The Root Mean Squared Error of our forecasts is {}'.format(round(err,2)))


#-------------------------Distance prediction---------------------
pd.plotting.register_matplotlib_converters()
y=data['Distance']
y.plot(figsize=(15, 6))
plt.show()

from pylab import rcParams
rcParams['figure.figsize'] = 18, 8

decomposition = sm.tsa.seasonal_decompose(y, model='additive')
fig = decomposition.plot()
plt.show()

p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 4) for x in list(itertools.product(p, d, q))]


for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(y,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)

            results = mod.fit()

            #print('ARIMA{}x{}4 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue
        
mod = sm.tsa.statespace.SARIMAX(y,
                                order=(1, 1, 1),
                                seasonal_order=(1, 1, 1, 4),
                                enforce_stationarity=False,
                                enforce_invertibility=False)

results = mod.fit()

#print(results.summary().tables[1])

#results.plot_diagnostics(figsize=(16, 8))
#plt.show()

pred = results.get_prediction(start=pd.to_datetime('2018-07-01'), dynamic=False)

pred_ci = pred.conf_int()

pred._results
ax = y['2018':].plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='Predicted', alpha=.7, figsize=(14, 7))

ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)

ax.set_xlabel('Date')
ax.set_ylabel('Distance')
ax.set_title('Distance covered in ride')
plt.legend()
plt.savefig('2.png')
plt.show()

y_forecasted = pred.predicted_mean
y_truth = y['2018-07-01':]

# Compute the mean square error
err=sqrt(mean_squared_error(y_forecasted, y_truth))
print('The Root Mean Squared Error of our forecasts is {}'.format(round(err,2)))





