from RegressorClass import regressor
from ExploratoryAnalysis import extensive_eda
import pandas as pd

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


# Perform EDA using extensive_eda class
#eda = extensive_eda()
#eda.save_eda_html(df)



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Create regressor object
    model = regressor()
    print(model)
    new_columns, _ = model.correlation_multicollinearity(X)
    print(f"\nImportant columns after performing the mult-collinearity test are :{new_columns}\n")
    x = model.pre_process(X[new_columns])
    X_train, X_test, y_train, y_test = model.split_train_test(x, y, test_size=0.2)
    model.train_models(X_train, y_train)
    results = model.evaluate_models(X_test, y_test)
    best_model, evaluation_results = model.select_best_model()
    print("The best model to put in production is: ", best_model)
    print()
    print(evaluation_results)
    #model.save_best_model()
    #print("\nFinal feature names are :")
    #print()
    #print(model.get_feature_names())
    #print()
    #model.plot_features_importance(X_train)
    #print(X_train[:2])


