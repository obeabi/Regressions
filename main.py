from RegressorClass import regressor
from ExploratoryAnalysis import extensive_eda
import pandas as pd

file ='Advertisin'
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


# Perform EDA using extensive_eda class
#eda = extensive_eda()
#eda.save_eda_html(df)



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Create regressor object
    model = regressor()
    #x_mc,vif_data = model.check_multicollinearity(X)
    #print(x_mc.columns)
    x= model.pre_process(X=X, categorical_features=['Type'])
    #print(type(x))

    X_train, X_test, y_train, y_test = model.split_train_test(x, y, test_size=0.2)
    model.train_models(X_train, y_train)
    results = model.evaluate_models(X_test, y_test)
    print(results)
    print()
    print(model.select_best_model())
    model.save_best_model()
    print(model.get_feature_names())
    #model.plot_feature_importances(df[['Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]', 'Machine failure', 'TWF']])
    model.plot_features_importance(X_train)
    #print(X_train[:2])


