from RegressorClass import regressor
from ExploratoryAnalysis import extensive_eda
import pandas as pd

file ='Advertisin'
#file ='Admission'
if file == 'Advertising':
    df = pd.read_csv('Advertising.csv')
    df = df.drop(df.columns[0], axis=1)
    X = df[['TV', 'radio', 'newspaper']]
    y = df.sales
elif file == 'Admission':
    df = pd.read_csv('Admission_Prediction.csv')
    df = df.drop(df.columns[0], axis=1)
    X = df.iloc[:, 0:-1]
    y = df.iloc[:, -1]

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
    #new_columns, _ = model.correlation_multicollinearity(X)
    #print(f"\nImportant columns after performing the mult-collinearity test are :{new_columns}\n")
    new_columns, _, _ = model.vif_multicollinearity(X)
    print(f"\nImportant columns after performing the mult-collinearity test using VIF approach are :{new_columns}\n")
    numerical_cols, category_cols = model.find_numericategory_columns(X)
    print("\nThe numeric columns are :", numerical_cols)
    print("\nThe categorical columns are :", category_cols)
    x = model.pre_process(X[new_columns])
    X_train, X_test, y_train, y_test = model.split_train_test(x, y, test_size=0.2)
    model.fit_all_models(X_train, y_train)
    best_model_name, best_model = model.select_best_model(X_train, X_test, y_train, y_test)
    print("\nThe best model to put in production based on test-set is: ", best_model_name)
    print("\nThe evaluation metrics from the best model based on train set are :", model.evaluation_results[best_model_name]["Train"])
    print("\nThe evaluation metrics from the best model based on test set are :", model.evaluation_results[best_model_name]["Test"])
    #model.save_best_model()
    #print("\nFinal feature names are :", model.get_feature_names())
    #model.plot_features_importance(X_train)
    #print(model.get_feature_importance())
    #print(X_train[:2])



