import numpy as np
import pandas as pandas
import statsmodels.api as sm

from returns_data import read_goog_sp500_logistic_data

xData, yData = read_goog_sp500_logistic_data()

logit = sm.Logit(xData, yData)

result = logit.fit()

predictions = (result.predict(xData) > 0.5)

num_accurate_predictions = (list(yData == predictions)).count(True)

pctAccuracy = float(num_accurate_predictions) / floar(len(predictions))

print(pctAccuracy)