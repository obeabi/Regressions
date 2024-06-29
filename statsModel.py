"""
This script performs statistical approach to perform regression
"""

import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd
from Logger import CustomLogger

logs = CustomLogger()

file ='Advertising'
if file == 'Advertising':
    df = pd.read_csv('Advertising.csv')
    df = df.drop(df.columns[0], axis=1)
    X = df[['TV', 'radio', 'newspaper']]
    y = df.sales
else:
    df = pd.read_csv('ai4i2020.csv')
    df = df.drop(df.columns[0], axis=1)
    y = df['Air temperature [K]']
    X = df[['Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]', 'Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF', 'Type']]


if __name__ =='__main__':
    try :
        print("Welcome World!")
        #lm = smf.ols(formula='sales~TV', data=df).fit()
        #print(lm.summary())
        #lm2 = smf.ols(formula='sales~TV+radio', data=df).fit()
        #print(lm2.summary())
        #lm3 = smf.ols(formula='sales~TV+radio+newspaper', data=df).fit()
        #print(lm3.summary())
    except Exception as e:
        raise ValueError(f"Error in preprocessing data: {e}")
        logs.log("Something went wrong while training statistical model", level='ERROR')

