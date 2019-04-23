import pandas as pd
import numpy as np

def read_goog_sp500_data():
    googFile = 'data/GOOG.csv'
    spFile = 'data/SP_500.csv'
    
    
    goog = pd.read_csv(googFile, sep=",", usecols=[0,5], names=['Date', 'Goog'], header=0)
    sp = pd.read_csv(spFile, sep=",", usecols=[0,5], names=['Date', 'SP500'], header=0)
    goog['SP500'] = sp['SP500']
    goog['Date'] = pd.to_datetime(goog['Date'], format='%Y-%m-%d')
    
    goog = goog.sort_values(['Date'], ascending=[True])
    
    returns = goog[[key for key in dict(goog.dtypes) if dict(goog.dtypes)[key] in ['float64', 'int64']]]\
                .pct_change()
                
    xData = np.array(returns["SP500"])[1:]
    yData = np.array(returns["Goog"])[1:]

    return (xData, yData)

def read_goog_sp500_dataframe():
    googFile = 'data/GOOG.csv'
    spFile = 'data/SP_500.csv'
    
    
    goog = pd.read_csv(googFile, sep=",", usecols=[0,5], names=['Date', 'Goog'], header=0)
    sp = pd.read_csv(spFile, sep=",", usecols=[0,5], names=['Date', 'SP500'], header=0)
    goog['SP500'] = sp['SP500']
    goog['Date'] = pd.to_datetime(goog['Date'], format='%Y-%m-%d')
    
    goog = goog.sort_values(['Date'], ascending=[True])
    
    returns = goog[[key for key in dict(goog.dtypes) if dict(goog.dtypes)[key] in ['float64', 'int64']]]\
                .pct_change()

    return returns

def read_goog_sp500_logistic_data():

    returns = read_goog_sp500_dataframe()

    retunrs['Intercept'] = 1

    xData = np.array(returns[["SP500", "Intercept"]][1:-1])

    yData = (returns["Goog"] > 0)[1:-1] 

    return = (xData, yData)


def read_xom_oil_nasdaq_data():
    def readFile(filename):
        data = pd.read_csv(filename, sep=",", usecols=[0,5], names=['Date', 'Price'], header=0)
        data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d')
        data = data.sort_values(['Date'], ascending=[True])

        returns = data[[key for key in dict(data.dtypes) if dict(data.dtypes)[key] in ['float64', 'int64']]]\
            .pct_change()
        
        return np.array(returns["Price"])[1:]

    nasdaqData = readFile('data/NASDAQ.csv')
    oilData = readFile('data/USO.csv')
    xomData = readFile('data/XOM.csv')

    return (nasdaqData, oilData, xomData)