import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler, OneHotEncoder ,OrdinalEncoder ,LabelEncoder
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
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
            "XGBRegressor": XGBRegressor(random_state=self.random_state, verbosity=10)
        }
        self.best_model = None
        self.preprocessor = None
        self.scaler = StandardScaler()
        self.trained_models = {}
        self.results = {}
        self.wcss = None

    def pre_process(self,X, categorical_features):
        """

        :param X:
        :param categorical_features:
        :return:
        """
        try:
            numeric_features = [col for col in X.columns if col not in categorical_features]
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('encoder', OneHotEncoder(handle_unknown='ignore'))
            ])
            self.preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numeric_features),
                    ('cat', categorical_transformer, categorical_features)
                ])
            x_preprocessed = self.preprocessor.fit_transform(X)
            logs.log("Successfully performed the pre-processing step!")
            return x_preprocessed

        except Exception as e:
            raise ValueError(f"Error in preprocessing data: {e}")
            logs.log("Something went wrong while pre-processing the dataset",level='ERROR')

    def find_optimal_clusters(self, X, max_k=5):
        """

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
            return optimal_k
        except Exception as e:
            raise ValueError(f"Error in finding optimal clusters {e}")
            logs.log("Something went wrong while finding opimal clusters", level='ERROR')


    def plot_elbow_curve(self, max_k=5):
        """
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

        :param X:
        :param n_clusters:
        :return:
        """
        try:
            x_scaled = self.scale_features(X)
            kmeans = KMeans(n_clusters=n_clusters, init="k-means++", n_init=12, random_state=self.random_state)
            return kmeans.fit_predict(x_scaled)
        except Exception as e:
            raise ValueError(f"Error in fitting data to clusters {e}")
            logs.log("Something went wrong while fitting data to cluster", level='ERROR')

    def split_train_test(self, X, y, test_size=0.2):
        """

        :param X:
        :param y:
        :param test_size:
        :return:
        """
        try:
            return train_test_split(X, y, test_size=test_size, random_state=self.random_state)
        except Exception  as e:
            raise ValueError(f"Error in splitting dataset into train-test sets {e}")
            logs.log("Something went wrong while splitting dataset into train-test sets", level='ERROR')


    def check_multicollinearity(self, X, threshold=10.0):
        """

        :param X:
        :param threshold:
        :return:
        """
        try:
            vif_data = pd.DataFrame()
            vif_data["feature"] = X.columns
            vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

            # Drop columns with VIF above the threshold
            high_vif_features = vif_data[vif_data["VIF"] > threshold]["feature"].tolist()
            X_dropped = X.drop(columns=high_vif_features)

            return X_dropped, vif_data
        except Exception as e:
            raise ValueError(f"Error in checking multicollinearity: {e}")

    def adjusted_r2(self, r2, n, p):
        """
        :param r2:
        :param n:
        :param p:
        :return:
        """
        try:
            return 1 - (1 - r2) * (n - 1) / (n - p - 1)
        except Exception as e:
            raise ValueError(f"Error in preprocessing data: {e}")
            logs.log("Something went wrong while estimating adjusted r-squared", level='ERROR')

    def train_models(self, X, y):
        try:
            n, p = X.shape
            for name, model in self.models.items():
                model.fit(X, y)
                self.trained_models[name] = model
                y_pred = model.predict(X)
                mse = mean_squared_error(y, y_pred)
                r2 = r2_score(y, y_pred)
                adj_r2 = self.adjusted_r2(r2, n, p)
                self.results[name] = {'mse': mse, 'r2': r2, 'adj_r2': adj_r2}

            self.select_best_model()
        except Exception as e:
            raise ValueError(f"Error in training models: {e}")

    def evaluate_models(self, X_test, y_test):
        """
        :param X_test:
        :param y_test:
        :return:
        """
        try:
            n, p = X_test.shape
            evaluation_results = {}
            for name, model in self.trained_models.items():
                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                adj_r2 = self.adjusted_r2(r2, n, p)
                evaluation_results[name] = {'mse': mse, 'r2': r2, 'adj_r2': adj_r2}
            return evaluation_results
        except Exception as e:
            raise ValueError(f"Error in evaluating models: {e}")

    def select_best_model(self):
        """
        :return:
        """
        try:
            self.best_model = max(self.trained_models, key=lambda k: self.results[k]['adj_r2'])
        except Exception as e:
            raise ValueError(f"Error in selecting the best model: {e}")


    def save_best_model(self, filename):
        """
        :param filename:
        :return:
        """
        try:
            best_model_instance = self.trained_models[self.best_model]
            joblib.dump(best_model_instance, filename)
        except Exception as e:
            raise ValueError(f"Error in saving the best model: {e}")

    def plot_feature_importances(self):
        """
        :return:
        """
        try:
            if self.best_model in ["RandomForestRegressor", "CatBoostRegressor", "XGBRegressor"]:
                model = self.trained_models[self.best_model]
                feature_importances = model.feature_importances_
                features = (self.preprocessor.transformers_[0][2] +
                            self.preprocessor.transformers_[1][1]['encoder'].categories_[0].tolist())
                plt.figure(figsize=(10, 6))
                plt.barh(features, feature_importances)
                plt.xlabel('Feature Importance')
                plt.ylabel('Feature')
                plt.title(f'Feature Importances in {self.best_model}')
                plt.show()
            else:
                raise ValueError(f"Feature importances are not available for the best model: {self.best_model}")
        except Exception as e:
            raise ValueError(f"Error in plotting feature importances: {e}")

    def tune_parameters(self, param_grid, X, y):
        try:
            model = self.models[self.best_model]
            grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
            grid_search.fit(X, y)
            self.best_model = grid_search.best_estimator_
            self.trained_models[self.best_model.__class__.__name__] = self.best_model
            self.results[self.best_model.__class__.__name__] = {
                'mse': -grid_search.best_score_,
                'r2': r2_score(y, self.best_model.predict(X)),
                'adj_r2': self.adjusted_r2(r2_score(y, self.best_model.predict(X)), X.shape[0], X.shape[1])
            }
            self.select_best_model()
        except Exception as e:
            raise ValueError(f"Error in tuning parameters: {e}")

    def cross_validate_models(self, X, y, cv=5):
        """
        :param X:
        :param y:
        :param cv:
        :return:
        """
        try:
            cv_results = {}
            kf = KFold(n_splits=cv, shuffle=True, random_state=self.random_state)
            for name, model in self.models.items():
                mse_scores = -cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error')
                r2_scores = cross_val_score(model, X, y, cv=kf, scoring='r2')
                n, p = X.shape
                adj_r2_scores = [self.adjusted_r2(r2, n, p) for r2 in r2_scores]
                cv_results[name] = {'mse': mse_scores.mean(), 'r2': r2_scores.mean(), 'adj_r2': np.mean(adj_r2_scores)}
            return cv_results
        except Exception as e:
            raise ValueError(f"Error in cross-validating models: {e}")

    def __str__(self):
        return "This is my custom regressor class object"


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print("Installed successfully!")