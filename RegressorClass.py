import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, LabelEncoder, MinMaxScaler, \
    FunctionTransformer
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso, RidgeCV, LassoCV, ElasticNet, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error
import joblib

import matplotlib.pyplot as plt
from Logger import CustomLogger

logs = CustomLogger()


class regressor:
    """
    This class object handles regression problems
    """

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {
            "LinearRegression": LinearRegression(),
            "RandomForestRegressor": RandomForestRegressor(random_state=self.random_state),
            "SVR": SVR(),
            "CatBoostRegressor": CatBoostRegressor(random_state=self.random_state, verbose=10),
            "XGBRegressor": XGBRegressor(random_state=self.random_state, verbosity=1)
        }
        self.best_model = None
        self.best_model_name = None
        self.preprocessor = None
        self.scaler = StandardScaler()
        self.max_scaler = MinMaxScaler()
        self.trained_models = {}
        self.results = {}
        self.evaluation_results = {}
        self.lasso_alpha = None
        self.ridge_alpha = None
        self.elasticnet_alpha = None
        self.elasticnet_l1_ratio = None
        self.wcss = None
        self.numerical_cols = None
        self.categorical_cols = None
        self.column_transformer = None
        self.fit_status = False
        self.preprocessing_pipeline = None
        self.fit_status = False
        self.feature_names_out = None  # Store feature names after transformation

    def preprocessor_fit(self, X, one_hot_encode_cols=None, label_encode_cols=None):
        """
        Fit the preprocessor on the data.

        Args:
            X (pd.DataFrame): Input data containing both numerical and categorical columns.
            one_hot_encode_cols (list): List of categorical columns to one-hot encode.
            label_encode_cols (list): List of categorical columns to label encode.
        """
        try:
            self.numerical_cols = X.select_dtypes(include=['number']).columns.tolist()
            self.categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

            transformers = []

            if self.numerical_cols:
                num_pipeline = Pipeline([
                    ('num_imputer', SimpleImputer(strategy='mean')),
                    ('scaler', StandardScaler())
                ])
                transformers.append(('num', num_pipeline, self.numerical_cols))

            if self.categorical_cols:
                if one_hot_encode_cols:
                    cat_pipeline = Pipeline([
                        ('cat_imputer', SimpleImputer(strategy='most_frequent')),
                        ('onehot', OneHotEncoder(handle_unknown='ignore'))
                    ])
                    transformers.append(('cat_onehot', cat_pipeline, one_hot_encode_cols))

                if label_encode_cols:
                    for col in label_encode_cols:
                        transformers.append((f'{col}_label', FunctionTransformer(self.label_encode), [col]))

            self.preprocessing_pipeline = ColumnTransformer(transformers, remainder='passthrough')
            self.preprocessing_pipeline.fit(X)
            self.fit_status = True
            self.feature_names_out = self.get_feature_names_out()
            logs.log("Successfully fitted the pre-processing pipeline!")
        except Exception as e:
            logs.log(f"Error during fit: {str(e)}")

    def preprocessor_transform(self, X):
        """
        Transform the input data using the fitted preprocessor.

        Args:
            X (pd.DataFrame): Input data to transform.

        Returns:
            pd.DataFrame: Transformed data with original column names.
        """
        try:
            if not self.fit_status:
                raise ValueError("Preprocessor must be fit on data before transforming.")
            transformed_data = self.preprocessing_pipeline.transform(X)
            transformed_df = pd.DataFrame(transformed_data, columns=self.feature_names_out)
            logs.log("Successfully transformed the dataset using the pre-processing pipeline")
            return transformed_df
        except Exception as e:
            logs.log(f"Error during transform: {str(e)}")

    def get_feature_names_out(self):
        """
        Get feature names after transformation.

        Returns:
            list: List of feature names after transformation.
        """
        try:
            if self.preprocessing_pipeline is None:
                return []

            feature_names_out = []
            for name, trans, column_names in self.preprocessing_pipeline.transformers_:
                if trans == 'drop' or trans == 'passthrough':
                    continue
                if isinstance(trans, Pipeline):
                    if name.startswith('cat_onehot') and hasattr(trans.named_steps['onehot'], 'get_feature_names_out'):
                        feature_names_out.extend(trans.named_steps['onehot'].get_feature_names_out())
                    else:
                        feature_names_out.extend(column_names)
                elif isinstance(trans, FunctionTransformer):
                    feature_names_out.extend(column_names)
                else:
                    feature_names_out.extend(column_names)
            logs.log("Successfully retrieved features name!")
            return feature_names_out

        except Exception as e:
            logs.log(f"Error during get_feature_names_out: {str(e)}")

    def label_encode(self, X):
        """
        Apply label encoding to the input data.

        Args:
            X (pd.Series or pd.DataFrame): Input data to encode.

        Returns:
            np.ndarray: Label encoded data reshaped to 2D.
        """
        try:
            le = LabelEncoder()
            logs.log("Successfully performed label encoding!")
            return le.fit_transform(X.squeeze()).reshape(-1, 1)
        except Exception as e:
            raise ValueError(f"Something went wrong while performing label encoding: {e}")
            logs.log("Something went wrong while finding the numeric and categorical columns", level='ERROR')

    def find_numericategory_columns(self, X):
        """
        Find numerical and categorical columns from dataset
        :param X:
        :return: numeric and categorical column names
        """
        try:
            numeric_cols = X.select_dtypes(include=[float, int]).columns.tolist()
            categoric_cols = X.select_dtypes(exclude=[float, int]).columns.tolist()
            logs.log("Successfully extracted numerical and categorical columns!")

            return numeric_cols, categoric_cols
        except Exception as e:
            raise ValueError(f"Something went wrong while finding the numeric and categorical columns: {e}")
            logs.log("Something went wrong while finding the numeric and categorical columns", level='ERROR')

    def find_optimal_clusters(self, X, max_k=5):
        """
        Find optimal number of clusters using the K-means and WCSS method
        :param X:
        :param max_k:
        :return:
        """
        try:
            x_scaled = self.scale_features(X)
            self.wcss = []
            for k in range(1, max_k + 1):
                kmeans = KMeans(n_clusters=k, init="k-means++", n_init=12, random_state=self.random_state)
                kmeans.fit(x_scaled)
                self.wcss.append(kmeans.inertia_)

            diffs = np.diff(self.wcss)
            diffs_ratio = diffs[:-1] / diffs[1:]
            optimal_k = np.argmin(diffs_ratio) + 2  # +2 because of zero-based indexing and the diff shifts results by 1
            logs.log("Successfully found optimal clusters!")
            return optimal_k
        except Exception as e:
            raise ValueError(f"Error in finding optimal clusters {e}")
            logs.log("Something went wrong while finding optimal clusters", level='ERROR')

    def plot_elbow_curve(self, max_k=5):
        """
        Plot elbow plot to visualize optimal number of clusters
        :param max_k:
        :return:
        """
        try:
            plt.figure(figsize=(8, 6))
            plt.plot(range(1, max_k + 1), self.wcss, marker='o')
            plt.title('Elbow Method')
            plt.xlabel('Number of Clusters')
            plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
            plt.xticks(range(1, max_k + 1))
            plt.grid(True)
            plt.show(block=False)
        except Exception as e:
            raise ValueError(f"Error in rendering plot of elbow curve {e}")
            logs.log("Something went wrong while rendering plot of elbow curve", level='ERROR')

    def find_clusters(self, X, n_clusters=4):
        """
        Fit dataset to specified number of clusters using KMeans algorithm
        :param X:
        :param n_clusters:
        :return:
        """
        try:
            x_scaled = self.scale_features(X)
            kmeans = KMeans(n_clusters=n_clusters, init="k-means++", n_init=12, random_state=self.random_state)
            logs.log("Successfully fitted dataset with specified clusters!")
            return kmeans.fit_predict(x_scaled)
        except Exception as e:
            raise ValueError(f"Error in fitting data to clusters {e}")
            logs.log("Something went wrong while fitting data to cluster", level='ERROR')

    def split_train_test(self, X, y, test_size=0.2):
        """
        Split dataset into train and test split using test size of 20%
        :param X:
        :param y:
        :param test_size:
        :return:
        """
        try:
            logs.log("Successfully created the train-test set!")
            return train_test_split(X, y, test_size=test_size, random_state=self.random_state)
        except Exception as e:
            raise ValueError(f"Error in splitting dataset into train-test sets {e}")
            logs.log("Something went wrong while splitting dataset into train-test sets", level='ERROR')

    def correlation_multicollinearity(self, X, threshold=0.9):
        """
        Checks for multi-collinearity between features using pearson correlation
        :param X:
        :param threshold:
        :return: non-collinear features
        """
        try:
            numeric_features = X.select_dtypes(include=[float, int]).columns.tolist()
            x_num = X[numeric_features].dropna()
            correlation_matrix = x_num.corr().abs()
            upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
            high_correlation_pairs = [(column, row) for row in upper_triangle.index for column in upper_triangle.columns
                                      if upper_triangle.loc[row, column] > threshold]
            columns_to_drop = {column for column, row in high_correlation_pairs}
            df_reduced = X.drop(columns=columns_to_drop)
            logs.log("Successfully dropped mult-collinear columns!")
            return df_reduced.columns, columns_to_drop
        except Exception as e:
            raise ValueError(f"Error in checking multi-collinearity: {e}")
            logs.log("Something went wrong while checking multi-collinearity:", level='ERROR')

    def fit_lasso_cv(self, X, y):
        """
        Perform lasso regression on train-set using specified cross validation
        :param X:
        :param y
        :return: trained model
        """
        try:
            lasso_cv = LassoCV(cv=10, random_state=self.random_state)
            lasso_cv.fit(X, y)
            self.lasso_alpha = lasso_cv.alpha_
            self.models["Lasso"] = Lasso(alpha=self.lasso_alpha, random_state=self.random_state)
            logs.log("Successfully found the best alpha for Lasso regression")
        except Exception as e:
            raise ValueError(f"Error in fitting LassoCV: {e}")
            logs.log("Something went wrong while fitting LassoCV", level='ERROR')

    def fit_ridge_cv(self, X, y):
        """
         Perform Ridge regression on train-set using specified cross validation
        :param X:
        :param y
        :return: trained model

        """
        try:
            ridge_cv = RidgeCV(alphas=np.random.uniform(0, 10, 50), cv=10)
            ridge_cv.fit(X, y)
            self.ridge_alpha = ridge_cv.alpha_
            self.models["Ridge"] = Ridge(alpha=self.ridge_alpha, random_state=self.random_state)
            logs.log("Successfully found the best alpha for Lasso regression")
        except Exception as e:
            raise ValueError(f"Error in fitting RidgeCV: {e}")
            logs.log("Something went wrong while fitting RidgeCV", level='ERROR')

    def fit_elasticnet_cv(self, X, y):
        """
         Perform Elastic-Net regression on train-set using specified cross validation
        :param X:
        :param y
        :return: trained model

        """
        try:
            elastic_cv = ElasticNetCV(alphas=None, cv=10)
            elastic_cv.fit(X, y)
            self.elasticnet_alpha = elastic_cv.alpha_
            self.elasticnet_l1_ratio = elastic_cv.l1_ratio_
            self.models["ElasticNet"] = ElasticNet(alpha=self.elasticnet_alpha, l1_ratio=self.elasticnet_l1_ratio,
                                                   random_state=self.random_state)
            logs.log("Successfully found the best alpha and l1 for ElasticNet regression")
        except Exception as e:
            raise ValueError(f"Error in fitting ElasticNetCV: {e}")
            logs.log("Something went wrong while fitting ElasticNetCV", level='ERROR')

    def fit_model(self, X, y, model):
        """
         Perform  regression on train-set using specified model
        :param X:
        :param y
        :param model
        :return: trained model
        """
        try:
            model.fit(X, y)
            logs.log("Successfully trained the model")
        except Exception as e:
            raise ValueError(f"Error in fitting model: {e}")
            logs.log("Something went wrong while training model", level='ERROR')

    def predict(self, X, model):
        """
         Perform  prediction on trained regression model
        :param X:
        :param model
        :return: prediction
        """
        try:
            return model.predict(X)
        except Exception as e:
            raise ValueError(f"Error in making predictions: {e}")
            logs.log("Something went wrong while making predictions", level='ERROR')

    def adjusted_r2(self, r2, n, p):
        """
        calculates the adjusted r-squared for regression model(s)
        :param r2:
        :param n:
        :param p:
        :return:
        """
        try:
            return 1 - ((1 - r2) * (n - 1) / (n - p - 1))
        except Exception as e:
            raise ValueError(f"Error in preprocessing data: {e}")
            logs.log("Something went wrong while estimating adjusted r-squared", level='ERROR')

    def evaluate_model(self, y_true, y_pred, X=None):
        """
        Evaluate all trained models
        :param y_true:
        :param y_pred:
        :param X: Optional, required for Adjusted R2 calculation
        :return:
        """
        try:
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_true, y_pred)
            adj_r2 = None
            if X is not None:
                adj_r2 = 1 - (1 - r2) * (len(y_true) - 1) / (len(y_true) - len(X[0, :]) - 1)
            return {
                "MSE": mse,
                "RMSE": rmse,
                "R2": r2,
                "Adjusted R2": adj_r2
            }
        except Exception as e:
            raise ValueError(f"Error in evaluating model: {e}")
            logs.log("Something went wrong while evaluating model", level='ERROR')

    def fit_all_models(self, X, y):
        """
        Fit all specified ML model at once
        :param X:
        :param y:
        """
        try:
            self.fit_lasso_cv(X, y)  # Fit LassoCV and add Lasso using best alpha to models
            self.fit_ridge_cv(X, y)  # Fit RidgeCV and add Ridge using best alpha to models
            self.fit_elasticnet_cv(X, y)  # Fit ElasticNetCV and add ElasticNet using best alpha and l1_ratio to models
            for model_name, model in self.models.items():
                self.fit_model(X, y, model)
                self.trained_models[model_name] = model
            logs.log("Successfully trained all models")
        except Exception as e:
            raise ValueError(f"Error in fitting all models: {e}")
            logs.log("Something went wrong while training all models", level='ERROR')

    def select_best_model(self, X_train, X_test, y_train, y_test):
        """
        Select the best ML model based on adjusted r2 score
        :param X_train:
        :param X_test:
        :param y_train:
        :param y_test:
        :return:
        """
        try:
            best_score = -np.inf
            best_model_name = None
            best_model = None
            for model_name, model in self.trained_models.items():
                y_train_pred = self.predict(X_train, model)
                y_test_pred = self.predict(X_test, model)
                train_scores = self.evaluate_model(y_train, y_train_pred, X_train)
                test_scores = self.evaluate_model(y_test, y_test_pred, X_test)
                self.evaluation_results[model_name] = {"Train": train_scores, "Test": test_scores}
                if test_scores["Adjusted R2"] > best_score:
                    best_score = test_scores["Adjusted R2"]
                    best_model_name = model_name
                    best_model = model
            self.best_model = best_model
            # self.test_best_model = self.evaluation_results[best_model_name]["Test"]
            logs.log("Successfully selected the best model")
            return best_model_name, best_model
        except Exception as e:
            raise ValueError(f"Error in selecting the best model: {e}")
            logs.log("Something went wrong while selecting the best model", level='ERROR')

    def save_best_model(self, file_path='linear_reg.pickle'):
        """
        Save the trained ML model as a pickle or sav file.
        :param file_path:
        :return:
        """
        try:
            with open(file_path, 'wb') as file:
                pickle.dump(self.best_model, file)
                #pickle.dump(self.gridsearch_best_model, file)
            print("\nModel saved as a pickle file successfully!")
            logs.log("Successfully saved the best model")
        except Exception as e:
            raise ValueError(f"Error in saving the best model: {e}")
            logs.log("Something went wrong while saving the best model", level='ERROR')

    def load_model(self, filename='linear_reg.pickle'):
        """
        Load the saved model file.
        :param filename:
        :return : saved model
        """
        try:
            saved_model = pickle.load(open(filename, 'rb'))
            logs.log("Model loaded successfully!")
            return saved_model
        except Exception as e:
            raise ValueError(f"Error in loading the best saved model: {e}")
            logs.log("Something went wrong while loading the best saved model ", level='ERROR')

    def plot_features_importance(self, model_name='LinearRegression'):
        """
        Plot features importance for the best model.
        : param model_name:
        :return:
        """
        try:
            if model_name in ["RandomForestRegressor", "CatBoostRegressor", "XGBRegressor"]:
                model = self.trained_models[model_name]
                feature_importances = model.feature_importances_
            elif model_name in ["LinearRegression", "Lasso", "Ridge", "ElasticNet"]:
                model = self.trained_models[model_name]
                feature_importances = np.abs(model.coef_)  # Use absolute values of coefficients

                # Extract feature names from the preprocessor
                features = self.get_feature_names_out()
                print(f"Features: {features}")
                print(f"Feature Importances: {feature_importances}")

                if len(features) != len(feature_importances):
                    print(
                        f"Lengths of features ({len(features)}) and feature_importances ({len(feature_importances)}) do not match.")
                    raise ValueError("Lengths of features and feature_importances do not match.")

                # If using one-hot encoding, aggregate importance scores per original feature
                original_features = []  # List to store original feature names without one-hot encoding suffixes
                importance_scores = []  # List to store aggregated importance scores per original feature

                for feature_name, importance_score in zip(features, feature_importances):
                    # Example: Extract original feature name from one-hot encoded feature
                    original_feature_name = feature_name.split('_')[0]  # Assuming '_encoded' suffix

                    if original_feature_name not in original_features:
                        original_features.append(original_feature_name)
                        importance_scores.append(importance_score)
                    else:
                        index = original_features.index(original_feature_name)
                        importance_scores[index] += importance_score

                plt.figure(figsize=(10, 6))
                plt.barh(original_features, importance_scores)
                plt.xlabel('Feature Importance')
                plt.ylabel('Feature')
                plt.title(f'Feature Importance in {model_name}')
                plt.show(block=True)
            else:
                raise ValueError(f"Feature importance are not available for the best model: {self.test_best_model}")
        except Exception as e:
            raise ValueError(f"Error in plotting feature importance: {e}")
            logs.log("Something went wrong while plotting feature importance ", level='ERROR')

    def get_feature_importance(self, model_name='LinearRegression'):
        """
        Establish feature importance
        :param model_name:
        :return:
        """
        try:
            if model_name not in self.trained_models:
                raise ValueError(f"Model {model_name} not found. Train the model first.")
            model = self.trained_models[model_name]
            if model_name == 'LinearRegression':
                # Coefficients as feature importance for linear regression
                importances = model.coef_
            elif model_name == 'Lasso':
                # Coeffcients as feature importance for Lasso Regression
                importances = model.coef_
            elif model_name == 'Ridge':
                # Coeffcients as feature importance for Ridge Regression
                importances = model.coef_
            elif model_name == 'ElasticNet':
                # Coeffcients as feature importance for ElasticNet Regression
                importances = model.coef_
            elif hasattr(model, 'feature_importances_'):
                # Feature importances for tree-based models
                importances = model.feature_importances_
            else:
                raise ValueError(f"Feature importance not supported for {model_name}")

            feature_names = self.get_feature_names_out()
            feature_importances = pd.Series(importances, index=feature_names)
            sorted_importances = feature_importances.sort_values(ascending=False)
            return sorted_importances
        except Exception as e:
            raise ValueError(f"Error in getting feature importance: {e}")
            logs.log("Something went wrong while getting feature importance", level='ERROR')

    def tune_parameters(self, model_name, X, y, cv=5, scoring='r2'):
        """
        Perform parameter tuning of parameters using Grid Search CV
        :param model_name:
        :param X:
        :param y:
        :param cv:
        :param scoring: 'neg_mean_squared_error', 'r2'
        :return:

        """
        try:
            # Define models and their parameter grids
            models = {'Lasso': (Lasso(alpha=self.lasso_alpha, random_state=self.random_state), {
                        'max_iter': [1000, 1500, 2000, 5000, 10000]}),

                      'ElasticNet': (ElasticNet(alpha=self.elasticnet_alpha, l1_ratio=self.elasticnet_l1_ratio,
                                                random_state=self.random_state), {
                                         'max_iter': [1000, 1500, 2000, 3000, 5000, 10000]}),

                      'LinearRegression': (LinearRegression(), {
                          'fit_intercept': [True, False],
                          'normalize': [True, False]}),

                      'RandomForestRegressor': (RandomForestRegressor(random_state=self.random_state), {
                          'n_estimators': [100, 200, 500],
                          'max_depth': [None, 10, 20, 30],
                          'min_samples_split': [2, 5, 10]}),

                      'Ridge': (Ridge(alpha=self.ridge_alpha, random_state=self.random_state), {
                          'max_iter': [1000, 5000, 10000]}),

                      'SVR': (SVR(), {
                          'C': [0.1, 1, 10],
                          'gamma': ['scale', 'auto'],
                          'kernel': ['linear', 'rbf']}),

                      'XGBRegressor': (XGBRegressor(random_state=self.random_state, verbosity=1), {
                          'n_estimators': [100, 200, 500],
                          'learning_rate': [0.01, 0.1, 0.3],
                          'max_depth': [3, 5, 7]}),

                      'CatBoostRegressor': (CatBoostRegressor(random_state=self.random_state, verbose=10), {
                          'iterations': [100, 200, 500],
                          'learning_rate': [0.01, 0.1, 0.3],
                          'depth': [3, 5, 7]})
                      }

            # Select the model and parameter grid
            model, param_grid = models[model_name]
            # Initialize GridSearchCV
            grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=cv)
            grid_search.fit(X, y)
            # Get the best parameters and the best score
            best_params = grid_search.best_params_
            best_score = grid_search.best_score_
            self.best_model = grid_search.best_estimator_
            logs.log("Successfully tuned the parameters using GridSearchCV")
            return grid_search.best_estimator_, best_params, best_score
        except Exception as e:
            raise ValueError(f"Error in tuning parameters using GridSearchCV: {e}")
            logs.log("Something went wrong while tuning parameters using GridSearchCV", level='ERROR')

    def cross_validate_models(self, model_name, X, y, n_splits=5):
        """
        Perform cross validation of specified model
        :param X:
        :param y:
        :param n_splits:
        :param model_name:
        :return:
        """
        try:
            # Perform grid search to get the best model
            best_model, best_params, best_score = self.tune_parameters(model_name, X, y)
            # Initialize KFold
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
            #kf = KFold(n_splits=cv, shuffle=True, random_state=self.random_state)
            r2_scores = cross_val_score(best_model, X, y, cv=kf, scoring='r2')
            adj_r2_scores = 1 - (1 - r2_scores) * (len(y) - 1) / (len(y) - len(X[0, :]) - 1)
            logs.log("Successfully performed cross-validation")
            return np.mean(adj_r2_scores), np.std(adj_r2_scores)
        except Exception as e:
            raise ValueError(f"Error in cross-validating models: {e}")

    def get_feature_names(self):
        """This method extracts feature names from the preprocessing pipeline."""
        try:
            feature_names = []
            for name, transformer, columns in self.preprocessor.transformers:
                if transformer == 'drop' or transformer is None:
                    continue

                # Check if the transformer is a pipeline
                if hasattr(transformer, 'named_steps'):
                    steps = transformer.named_steps

                    # Handle case where OneHotEncoder is in the pipeline
                    if 'encoder' in steps and isinstance(steps['encoder'], OneHotEncoder):
                        encoder = steps['encoder']
                        feature_names.extend(encoder.get_feature_names_out(columns))

                    # Handle case where LabelEncoder or OrdinalEncoder is in the pipeline
                    elif 'encoder' in steps and isinstance(steps['encoder'], (LabelEncoder, OrdinalEncoder)):
                        feature_names.extend(columns)
                    # Handle case where SimpleImputer is in the pipeline
                    elif 'imputer' in steps and isinstance(steps['imputer'], SimpleImputer):
                        feature_names.extend(columns)

                    # Handle case where StandardScaler is in the pipeline
                    elif 'scaler' in steps and isinstance(steps['scaler'], (StandardScaler, MinMaxScaler)):
                        feature_names.extend(columns)
                    else:
                        feature_names.extend(columns)

                # Handle standalone OneHotEncoder
                elif isinstance(transformer, OneHotEncoder):
                    feature_names.extend(transformer.get_feature_names_out(columns))

                # Handle standalone LabelEncoder or OrdinalEncoder
                elif isinstance(transformer, (LabelEncoder, OrdinalEncoder)):
                    feature_names.extend(columns)

                # Handle standalone StandardScaler
                elif isinstance(transformer, (StandardScaler, MinMaxScaler)):
                    feature_names.extend(columns)

                # Default case: For transformers like SimpleImputer or others that don't change feature names
                else:
                    feature_names.extend(columns)

            return feature_names
        except Exception as e:
            raise ValueError(f"Error in getting feature names: {e}")
            logs.log("Something went wrong while getting feature names", level='ERROR')

    def vif_multicollinearity(self, X, threshold=10.0):
        """
        Checks for multi-collinearity between features doesn't work well
        :param X should be normailzed:
        :param threshold:
        :return: non-collinear features
        """
        try:
            numeric_features, _ = self.find_numericategory_columns(X)
            x_num = self.pre_process(X[numeric_features])
            vif_data = pd.DataFrame()
            vif_data["feature"] = X[numeric_features].columns
            vif_data["VIF"] = [variance_inflation_factor(x_num, i) for i in range(X[numeric_features].shape[1])]

            # Drop columns with VIF above the threshold
            high_vif_features = vif_data[vif_data["VIF"] > threshold]["feature"].tolist()
            x_dropped = X.drop(columns=high_vif_features)
            logs.log("Successfully performed the multi-collinearity check step!")

            return x_dropped.columns, high_vif_features, vif_data
        except Exception as e:
            raise ValueError(f"Error in checking multi-collinearity: {e}")
            logs.log("Something went wrong while checking multi-collinearity:", level='ERROR')

    def __str__(self):
        return "This is my custom regressor class object"


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print("Installed successfully!")
