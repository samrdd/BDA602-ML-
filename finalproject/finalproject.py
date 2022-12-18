import sys
import warnings

import pandas
import sqlalchemy
from Plots import Plot
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sqlalchemy import text
from Statistics import Stats
from tablestyle import make_clickable, table_style


def data_loader():

    """Loads the data from Mariadb"""

    db_user = "root"
    db_pass = "password"  # pragma: allowlist secret
    db_host = "mariadb1"
    db_database = "baseball"

    connect_string = (
        f"mariadb+mariadbconnector://{db_user}:{db_pass}@{db_host}/{db_database}"
    )

    sql_engine = sqlalchemy.create_engine(connect_string)

    query = """SELECT * FROM baseball_team_stats"""

    with sql_engine.begin() as connection:
        df = pandas.read_sql_query(text(query), connection)

    return df


def clean_data(df):

    """atlers the data for furthur usage"""

    df["HT_Wins"] = df["HT_Wins"].astype("int")
    df["temperature"] = df["temperature"].astype("float")
    df["wind_speed"] = df["wind_speed"].astype("float")

    # droping abnormal temperature

    df.drop(df[df["temperature"] == 7882].index, inplace=True)

    df = df.dropna()

    return df


def models_to_train(X_train, y_train):

    """Returns the list of traines models"""

    """Decision Tree Classifier"""

    pipe1 = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", DecisionTreeClassifier(random_state=1234)),
        ]
    )

    pipe1 = pipe1.fit(X_train, y_train)

    """Support Vector machine"""

    pipe2 = Pipeline([("scaler", StandardScaler()), ("model", SVC())])

    pipe2 = pipe2.fit(X_train, y_train)

    """K Nearest Neighbor"""

    pipe3 = Pipeline([("scaler", StandardScaler()), ("model", KNeighborsClassifier())])

    pipe3 = pipe3.fit(X_train, y_train)

    """Logistic Regression"""

    pipe4 = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(solver="liblinear", random_state=0)),
        ]
    )

    pipe4 = pipe4.fit(X_train, y_train)

    """Naive Bayes"""

    pipe5 = Pipeline([("scaler", StandardScaler()), ("model", GaussianNB())])

    pipe5 = pipe5.fit(X_train, y_train)

    """Random Forest"""

    pipe6 = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", RandomForestClassifier(random_state=42)),
        ]
    )

    pipe6 = pipe6.fit(X_train, y_train)

    """Neural Networks"""

    NN_model = MLPClassifier(
        solver="lbfgs", alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1
    )

    pipe7 = Pipeline([("scaler", StandardScaler()), ("model", NN_model)])

    pipe7 = pipe7.fit(X_train, y_train)

    GBclf = GradientBoostingClassifier(
        n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0
    )

    pipe8 = Pipeline([("scaler", StandardScaler()), ("model", GBclf)])

    pipe8 = pipe8.fit(X_train, y_train)

    model_names = [
        "Decision Tree(Basic)",
        "SVM",
        "KNN",
        "Logistic Regression",
        "Naive Bayes",
        "Random Forest",
        "Neural Networks",
        "Gradient Boosting",
    ]

    pipelines = [pipe1, pipe2, pipe3, pipe4, pipe5, pipe6, pipe7, pipe8]

    return model_names, pipelines


class Metrics:
    @staticmethod
    def model_scores(X_train, X_test, y_train, y_test):
        table = pandas.DataFrame(
            columns=[
                "Model",
                "Training_Accuracy",
                "Testing_Accuracy",
                "Matthews_Corrcoef",
                "Precision",
                "Recall",
                "F1_Score",
                "Roc_Curve",
            ]
        )

        model_names, pipelines = models_to_train(X_train, y_train)

        for i in range(0, len(pipelines)):

            Model = pipelines[i]

            y_pred_train = Model.predict(X_train)

            y_pred_test = Model.predict(X_test)

            table = table.append(
                {
                    "Model": model_names[i],
                    "Training_Accuracy": accuracy_score(y_train, y_pred_train),
                    "Testing_Accuracy": accuracy_score(y_test, y_pred_test),
                    "Matthews_Corrcoef": matthews_corrcoef(y_test, y_pred_test),
                    "Precision": precision_recall_fscore_support(
                        y_test, y_pred_test, average="weighted"
                    )[0],
                    "Recall": precision_recall_fscore_support(
                        y_test, y_pred_test, average="weighted"
                    )[1],
                    "F1_Score": precision_recall_fscore_support(
                        y_test, y_pred_test, average="weighted"
                    )[2],
                    "Roc_Curve": Plot.roc_curve(y_test, y_pred_test),
                },
                ignore_index=True,
            )

        table = table.sort_values(by="Testing_Accuracy", ascending=False).reset_index(
            drop=True
        )

        table = table.style.format(
            {
                "Roc_Curve": make_clickable,
            }
        )

        table = table_style(table)

        return table


def split_data(df, predictors, response):

    X = df[predictors]
    y = df[response]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, shuffle=False
    )

    return X_train, X_test, y_train, y_test


def main():

    warnings.filterwarnings("ignore")

    """loading the data"""

    """Use your own password in data_loader function"""
    df = data_loader()

    print("data loaded to python data frame")

    df = clean_data(df)

    # query = """SELECT * FROM baseball_team_stats"""

    # df = load_data(guery)

    # df = pandas.read_csv("baseball.csv")

    # Path("./ouput/plots/").mkdir(parents=True, exist_ok=True)

    predictors = df.drop(
        [
            "game_id",
            "game_date",
            "HT_Wins",
            "HT_id",
            "AT_id",
            "wind_direction",
            "stadium_id",
            "HT_100_SBR",
            "AT_100_SBR",
            "HT_100_SBP",
            "AT_100_SBP",
        ],
        axis=1,
    ).columns.values.tolist()

    response = "HT_Wins"

    print("started calculation feature importance it may take around 20 min")

    Stats.get_all_tables(df, predictors, response)

    X_train, X_test, y_train, y_test = split_data(df, predictors, response)

    results1 = Metrics.model_scores(X_train, X_test, y_train, y_test).to_html()

    # removing the predictors with p>0.05

    predictors1 = df.drop(
        [
            "game_id",
            "game_date",
            "HT_Wins",
            "HT_id",
            "AT_id",
            "wind_direction",
            "stadium_id",
            "HT_100_SBR",
            "AT_100_SBR",
            "HT_100_SBP",
            "HT_100_SBP",
            "AT_SP_100_DGR",
            "HT_50_OBP",
            "HT_100_OBP",
            "AT_SP_50_DGR",
            "AT_SP_50_AB_PT",
            "AT_50_OBP",
            "temperature",
            "HT_SP_100_DGR",
            "AT_100_OBP",
            "HT_SP_100_AB_PT",
            "PE_WHIP_100_diff",
            "AT_100_BAIP",
            "AT_100_BA",
            "AT_50_BA",
            "HT_100_BABIP",
            "HT_SP_50_BB_9IP",
            "AT_SP_50_K_9IP",
            "AT_50_BABIP",
            "AT_50_BAIP",
            "HT_SP_100_K_9IP",
            "wind_speed",
            "AT_SP_100_IF",
            "HT_100_BB_9IP",
            "HT_50_BABIP",
            "AT_SP_100_AB_PT",
            "AT_SP_100_BB_9IP",
            "AT_100_BABIP",
            "AT_SP_50_AB_PT",
            "AT_SP_50_IF",
        ],
        axis=1,
    ).columns.values.tolist()

    print("started training the models mean while you can se the analytics")
    print("results will be primted soon")

    X_train1, X_test1, y_train1, y_test1 = split_data(df, predictors1, response)

    results2 = Metrics.model_scores(X_train1, X_test1, y_train1, y_test1).to_html()

    # removing the predictors with corr > 0.95

    predictors2 = df.drop(
        [
            "game_id",
            "game_date",
            "HT_Wins",
            "HT_id",
            "AT_id",
            "wind_direction",
            "stadium_id",
            "HT_100_SBR",
            "AT_100_SBR",
            "HT_100_SBP",
            "HT_100_SBP",
            "AT_SP_100_DGR",
            "HT_50_OBP",
            "HT_100_OBP",
            "AT_SP_50_DGR",
            "AT_SP_50_AB_PT",
            "AT_50_OBP",
            "temperature",
            "HT_SP_100_DGR",
            "AT_100_OBP",
            "HT_SP_100_AB_PT",
            "PE_WHIP_100_diff",
            "AT_100_BAIP",
            "AT_100_BA",
            "AT_50_BA",
            "HT_100_BABIP",
            "HT_SP_50_BB_9IP",
            "AT_SP_50_K_9IP",
            "AT_50_BABIP",
            "AT_50_BAIP",
            "HT_SP_100_K_9IP",
            "wind_speed",
            "AT_SP_100_IF",
            "HT_100_BB_9IP",
            "HT_50_BABIP",
            "AT_SP_100_AB_PT",
            "AT_SP_100_BB_9IP",
            "AT_100_BABIP",
            "AT_SP_50_AB_PT",
            "AT_SP_50_IF",
            "PE_K_50_diff",
            "AT_50_K_9IP",
            "AT_100_K_9IP",
            "AT_100_OBA",
            "HT_50_OBA",
            "AT_50_OBA",
            "HT_100_H_9IP",
            "AT_100_HR_per_H",
            "HT_100_HR_per_H",
            "HT_50_HR_per_H",
            "AT_50_HR_per_H",
            "AT_50_K_to_BB",
        ],
        axis=1,
    ).columns.values.tolist()

    X_train2, X_test2, y_train2, y_test2 = split_data(df, predictors2, response)

    results3 = Metrics.model_scores(X_train2, X_test2, y_train2, y_test2).to_html()

    # to 20 predictors
    predictors_20 = df[
        [
            "AT_100_WHIP",
            "PE_K_50_diff",
            "AT_50_PE",
            "AT_100_K_9IP",
            "HT_100_OBA",
            "AT_50_K_9IP",
            "HT_50_WHIP",
            "AT_100_K_BB",
            "HT_50_OBA",
            "HT_100_WHIP",
            "AT_50_ERA",
            "AT_50_WHIP",
            "AT_100_ERA",
            "100_PE_diff",
            "50_PE_diff",
            "HT_100_PE",
            "PE_K_100_diff",
            "AT_100_PE",
            "HT_50_H_9IP",
            "HT_50_PE",
            "HT_100_H_9IP",
            "AT _50_K_BB",
            "PE_K_diff",
        ]
    ].columns.values.tolist()

    X_train3, X_test3, y_train3, y_test3 = split_data(df, predictors_20, response)

    results4 = Metrics.model_scores(X_train3, X_test3, y_train3, y_test3).to_html()

    """creating a html file for output of models"""

    file_html = open("./results/Baseball_Models_Output.html", "w")

    file_html.write(
        "<h1><center>Metrics of different models trained and tested on Baseball data</center></h1>"
    )

    file_html.write("<h2><center>Metrics with plots</center></h2>")

    file_html.write("<body><center>%s</center></body>" % results1)

    file_html.write(
        "<h2><center>Metrics after removing the features with p>0.05 </center></h2>"
    )

    file_html.write("<body><center>%s</center></body>" % results2)

    file_html.write(
        "<h2><center>Metrics after removing the features with correlation > 0.95</center></h2>"
    )

    file_html.write("<body><center>%s</center></body>" % results3)

    file_html.write("<h2><center>Metrics of top 20 performing features</center></h2>")

    file_html.write("<body><center>%s</center></body>" % results4)

    file_html.close()

    print("Awsome everything done, now you can check the results")


if __name__ == "__main__":
    sys.exit(main())
