from RegressorClass import regressor
from ExploratoryAnalysis import extensive_eda
import pandas as pd

file = 'Advertisin'
# file = 'Admission'
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
    X = df[['Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]', 'Machine failure',
            'TWF', 'HDF', 'PWF', 'OSF', 'RNF', 'Type']]

# Perform EDA using extensive_eda class
# eda = extensive_eda()
# eda.save_eda_html(df)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Create regressor object
    model = regressor()
    print(model)
    new_columns, columns_to_drop = model.correlation_multicollinearity(X)
    _, categroical_cols = model.find_numericategory_columns(X)
    print(
        f"\nImportant columns after performing the mult-collinearity test using CORRELATION approach are :{new_columns}\n")
    print("\nCategorical columns in the dataset are ", categroical_cols)
    #label_encode_cols = ['Type']
    #one_hot_encode_cols = ['Type']
    X_m = X[new_columns]
    X_train, X_test, y_train, y_test = model.split_train_test(X_m, y, test_size=0.2)
    model.preprocessor_fit(X_train, one_hot_encode_cols=categroical_cols, label_encode_cols=None)
    # Transform X_train
    X_train_transformed = model.preprocessor_transform(X_train).values
    model.fit_all_models(X_train_transformed, y_train)
    # Prediction
    X_test_transformed = model.preprocessor_transform(X_test).values
    best_model_name, best_model = model.select_best_model(X_train_transformed, X_test_transformed, y_train, y_test)
    print("\nThe best model to put in production based on test-set is: ", best_model_name)
    print(f"\nThe evaluation metrics from the {best_model_name} based on train set are :",
          model.evaluation_results[best_model_name]["Train"])
    print(f"\nThe evaluation metrics from the {best_model_name} based on test set are :",
          model.evaluation_results[best_model_name]["Test"])
    # Make Prediction
    # test1 = pd.DataFrame({
    #     'GRE Score': [324.000000],
    #     'TOEFL Score': [107.0],
    #     'University Rating': [4.0],
    #     'SOP': [4.0],
    #     'LOR': [4.5],
    #     'CGPA': [8.87],
    #     'Research': [1]})
    #
    # test1 = model.preprocessor_transform(test1).values
    # print(test1)
    # y_pred = model.predict(test1, model.best_model)
    # print("The chance of admission is :", y_pred[0])
    # model.save_best_model()
