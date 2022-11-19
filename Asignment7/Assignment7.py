import sys
from Statistics import *
import pandas
import sqlalchemy
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, hamming_loss


def data_loader():
    db_user = "root"
    db_pass = "root"
    db_host = "localhost"
    db_database = "baseball"
    connect_string = f"mariadb+mariadbconnector://{db_user}:{db_pass}@{db_host}/{db_database}"

    sql_engine = sqlalchemy.create_engine(connect_string)

    query = """
                    SELECT * FROM baseball_game_stats
                    """
    df = pandas.read_sql_query(query, sql_engine)

    df['HT_Wins'] = df['HT_Wins'].astype('int')

    return df


def models_to_train(X_train, y_train):
    """Returns the list of models"""

    """Decision Tree Classifier"""
    parameters = {
        "max_depth": range(1, 7),
        "criterion": ["gini", "entropy"],
    }
    decision_tree_grid_search = GridSearchCV(
        DecisionTreeClassifier(random_state=1234), parameters, n_jobs=4
    )
    decision_tree_grid_search.fit(X=X_train, y=y_train)

    model_1 = decision_tree_grid_search.best_estimator_

    """Decision tree with out grid search"""

    model_01 = DecisionTreeClassifier(random_state=1234)

    """Support Vector machine"""

    param_grid = {'C': [0.1],
                  'gamma': [1],
                  'kernel': ['rbf']}

    svc_grid_search = GridSearchCV(SVC(), param_grid, refit=True, verbose=3)

    svc_grid_search.fit(X_train, y_train)

    model_2 = svc_grid_search.best_estimator_

    """Support Vector machine with out grid search"""

    model_02 = SVC()

    """K Nearest Neighbor"""

    leaf_size = list(range(1, 3))
    n_neighbors = list(range(1, 10))
    p = [1, 2]
    # Convert to dictionary
    hyper_parameters = dict(leaf_size=leaf_size, n_neighbors=n_neighbors, p=p)
    # Create new KNN object
    knn = KNeighborsClassifier()
    # Use GridSearch
    knn_grid_search = GridSearchCV(knn, hyper_parameters, cv=5)
    # Fit the model
    knn_grid_search.fit(X_train, y_train)

    model_3 = knn_grid_search.best_estimator_

    """K Nearest Neighbor with out tuning"""

    model_03 = KNeighborsClassifier()

    """Logistic Regression"""

    model_4 = LogisticRegression(solver='liblinear', random_state=0)

    """Naive Bayes"""

    model_5 = GaussianNB()

    """Random Forest"""

    model_6 = RandomForestClassifier(random_state=42)

    models = [model_1, model_01, model_2, model_02, model_3, model_03, model_4, model_5, model_6]
    model_names = ['Decision Tree(GridSearch)', 'Decision Tree(Basic)',
                   'SVM(GridSearch)', 'SVM(Basic)',
                   'KNN(GridSearch)', 'KNN(Basic)',
                   'Logistic Regression',
                   'Naive Bayes',
                   'Random Forest']

    return models, model_names


class Metrics:

    @staticmethod
    def model_scores(X_train, X_test, y_train, y_test):
        table = pandas.DataFrame(
            columns=[
                "Model",
                "Training_Accuracy",
                "Testing_Accuracy",
                "Hamming_Loss",
                "Precision",
                "Recall",
                "F1_Score",
                "Roc_Curve",
            ]
        )

        models, model_names = models_to_train(X_train, y_train)

        for i in range(0, len(models)):
            Model = models[i].fit(X_train, y_train)

            y_pred_train = Model.predict(X_train)

            y_pred_test = Model.predict(X_test)

            table = table.append(
                {
                    "Model": model_names[i],
                    "Training_Accuracy": accuracy_score(y_train, y_pred_train),
                    "Testing_Accuracy": accuracy_score(y_test, y_pred_test),
                    "Hamming_Loss": hamming_loss(y_test, y_pred_test),
                    "Precision": precision_recall_fscore_support(y_test, y_pred_test, average='weighted')[0],
                    "Recall": precision_recall_fscore_support(y_test, y_pred_test, average='weighted')[1],
                    "F1_Score": precision_recall_fscore_support(y_test, y_pred_test, average='weighted')[2],
                    "Roc_Curve": Plot.roc_curve(y_test, y_pred_test),
                },
                ignore_index=True
            )

        table = table.sort_values(
            by="Testing_Accuracy", ascending=False
        ).reset_index(drop=True)

        table = table.style.format(
            {
                "Roc_Curve": make_clickable,
            }
        )

        table = table_style(table)

        return table


def main():
    df = data_loader()

    predictors = ['HT_BA',
                  'AT_BA',
                  'HT_home_runs_per_hit',
                  'AT_home_runs_per_hit',
                  'HT_PA',
                  'AT_PA',
                  'HT_BABIP',
                  'AT_BABIP',
                  'HT_BAIP',
                  'AT_BAIP',
                  'HT_K_to_BB',
                  'AT_K_to_BB',
                  'HT_BB_to_K',
                  'AT_BB_to_K',
                  'HT_OBP',
                  'AT_OBP',
                  'HT_HCP',
                  'AT_HCP',
                  'HT_SP_WHIP',
                  'AT_SP_WHIP',
                  'HT_SP_K_per_9inn',
                  'AT_SP_K_per_9inn',
                  'HT_SP_BB_per_9_inn',
                  'AT_SP_BB_per_9_inn',
                  'HT_SP_H_per_9_inn',
                  'AT_SP_H_per_9_inn']

    response = "HT_Wins"

    Stats.get_all_tables(df, predictors, response)

    """From the statistics I had observed the following,

        - Correlation Tables

        1. H_per_9_inn pitched is highly correlated with other predictors
        2. As BABIP and BAIP are related to PA(pitching average) and BA(batting average) these are also correlated

        SO from the above observations I am not using H_per_9inn and PA(as it is not a good statistic for a pitcher) 
        predictors in the model building

        Mean square difference tables

        1. Starting pitcher stats are performing good both individually and in pairs as these are in the top of the 
           tables     
        """

    """Splitting the data to test and train set

        Note: As I had ordered the final table by local date, I am directly splitting the data

        """

    new_predictors = ['HT_BA',
                      'AT_BA',
                      'HT_home_runs_per_hit',
                      'AT_home_runs_per_hit',
                      'HT_BABIP',
                      'AT_BABIP',
                      'HT_BAIP',
                      'AT_BAIP',
                      'HT_K_to_BB',
                      'AT_K_to_BB',
                      'HT_BB_to_K',
                      'AT_BB_to_K',
                      'HT_OBP',
                      'AT_OBP',
                      'HT_HCP',
                      'AT_HCP',
                      'HT_SP_WHIP',
                      'AT_SP_WHIP',
                      'HT_SP_K_per_9inn',
                      'AT_SP_K_per_9inn',
                      'HT_SP_BB_per_9_inn',
                      'AT_SP_BB_per_9_inn', ]

    X = df[new_predictors]
    y = df[response]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)

    """creating a html file for output of models"""

    file_html = open("Baseball_Models_Output.html", "w")

    file_html.write(
        "<h1><center>Metrics of different models trained and tested on Baseball data</center></h1>"
    )

    file_html.write("<h2><center>Metrics with plots</center></h2>")

    metrics_table = Metrics.model_scores(X_train, X_test, y_train, y_test).to_html()

    file_html.write("<body><center>%s</center></body>" % metrics_table)

    file_html.close()


if __name__ == "__main__":
    sys.exit(main())