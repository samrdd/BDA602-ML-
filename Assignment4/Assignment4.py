from statsmodels.tools.web import webbrowser
from sklearn.metrics.pairwise import linear_kernel

# importing necessary libraries
import sys
import pandas as pd
import numpy
from plotly import graph_objects as go
from sklearn.metrics import confusion_matrix
import plotly.express as px
import statsmodels.api
from sklearn.ensemble import RandomForestRegressor
from pathlib import Path


# Using the different data to test the code

def Cont_resp():
    # loading the data
    csv_path = '/Users/naveenreddysama/Downloads/auto-mpg.csv'
    df = pd.read_csv(csv_path)
    # Separating the data to target and response variables
    Predictor = df[["cylinders", "displacement", "weight", "acceleration", "model year", "origin"]]  # PREDICTOR
    Response = df[["mpg"]]  # RESPONSE

    return df, Predictor, Response


def Cat_resp():
    # loading the data
    csv_path = '/Users/naveenreddysama/Downloads/heart_failure_clinical_records_dataset.csv'
    df = pd.read_csv(csv_path)
    # Separating the data to target and response variables
    Predictor = df[
        ["age", "anaemia", "creatinine_phosphokinase", "diabetes", "ejection_fraction", "high_blood_pressure",
         "platelets",
         "sex", "smoking", "time"]]  # PREDICTOR
    Response = df[["DEATH_EVENT"]]  # RESPONSE

    return df, Predictor, Response

def Cat_resp2():
    csv_path='/Users/naveenreddysama/Downloads/archive/heart.csv'
    df = pd.read_csv(csv_path)
    # Separating the data to target and response variables
    Predictor = df[["age", "sex", "cp", "trtbps", "chol", "fbs", "restecg", "thalachh", "exng", "oldpeak", "slp", "caa",
              "thall"]]  # PREDICTOR
    Response = df[["output"]]  # RESPONSE

    return df, Predictor, Response

# defining a function to heck the response variable

def response_checker(col):
    if len(pd.unique(col.iloc[:, 0])) <= 2:
        return True
    else:
        return False

# defining a function to check the predictor variable

def predictor_checker (dataframe, column):

    if len(pd.unique(dataframe[column])) <= 2:

        return True
    else:
        return False


# function to store the plot as link

def plot_link(fig, name):
    Path('./plots_a4/').mkdir(parents=True, exist_ok=True)
    link = f"./plots_a4/{name}.html"
    fig.write_html(link)

    return link


# functions to plot the graphs

# continuous predictor vs continuous response
def cat_response_cat_predictor(X, Y, col):

    x = numpy.array(X[col])
    y = numpy.array(Y)

    X_2 = [1 if abs(x_) > 0.5 else 0 for x_ in x]
    Y_2 = [1 if abs(y_) > 0.5 else 0 for y_ in y]

    conf_matrix = confusion_matrix(X_2, Y_2)

    fig = go.Figure(
        data=go.Heatmap(z=conf_matrix, zmin=0, zmax=conf_matrix.max())
    )
    title = f"Categorical Predictor ({col}) by Categorical Response ({Y.columns[0]})"
    fig.update_layout(
        title=title,
        xaxis_title="Response",
        yaxis_title="Predictor",
    )
    # saving the url of the plot
    url = plot_link(fig, title)
    return url


# Cat response continuous predictor

def cat_resp_cont_predictor(X, col, Y, df):
    fig = px.histogram(df, x=X[col], color=Y.columns[0], marginal="rug", hover_data=X.columns)
    title = f"Continuous Predictor ({col}) by Categorical Response({Y.columns[0]})"
    fig.update_layout(
        title=title,
        xaxis_title="Predictor",
        yaxis_title="Distribution",
    )
    # saving the url of the plot
    url = plot_link(fig, title)
    return url


# Cont Response Cat Predictor

def cont_resp_cat_predictor(df, col, X, Y):
    fig = px.violin(df, y=Y.columns[0], x=X[col], box=True, hover_data=X.columns)

    title = f"Categorical Predictor ({col}) by Continous Response ({Y.columns[0]})"
    fig.update_layout(
        title=title,
        xaxis_title="Predictor",
        yaxis_title="Distribution",
    )
    # saving the url of the plot
    url = plot_link(fig, title)

    return url


# Cont Response Cont Predictor

def cont_response_cont_predictor(X, Y, col):
    print(col)
    x = X[col]
    y = Y.iloc[:, 0]

    fig = px.scatter(x=x, y=y, trendline="ols")
    title = f"Continuous Response({col}) by Continuous Predictor({Y.columns[0]})"
    fig.update_layout(
        title=title,
        xaxis_title="Predictor",
        yaxis_title="Response",
    )

    # saving the url of the plot
    url = plot_link(fig, title)

    return url


# defining a functions to calculate t value and p value

def p_t_values(df, col, Y):

    if response_checker(Y):
        predict = statsmodels.api.add_constant(df[col])
        logistic_regression_model = statsmodels.api.Logit(Y, predict)
        logistic_regression_model_fitted = logistic_regression_model.fit()
        t = round(logistic_regression_model_fitted.tvalues[1], 6)
        p = "{:.6e}".format(logistic_regression_model_fitted.pvalues[1])

    else:
        predict = statsmodels.api.add_constant(df[col])
        linear_regression_model = statsmodels.api.OLS(Y, predict)
        linear_regression_model_fitted = linear_regression_model.fit()
        t = round(linear_regression_model_fitted.tvalues[1], 6)
        p = "{:.6e}".format(linear_regression_model_fitted.pvalues[1])

    return t, p



# Random forest Variable importance score

def RFVImp_score(Y, X):

    x = X

    model = RandomForestRegressor()
    model.fit(x, Y)
    feature_imp = model.feature_importances_

    value = []

    for i, col in enumerate(X.columns):

        if predictor_checker(X, col):
            value.append("NA")
        else:
            value.append(feature_imp[i])

    return value


# initiating the lists to store the data in a list wrt

Response, Predictor, Plot, p_value, t_score, RF_VarImp, MWR_Unweighted, MWR_Weighted = [], [], [], [], [], [], [], []


def main():
    # reading the data
    df, X, Y = Cat_resp2()

    # reading the response name

    R_columns = list(Y.columns)

    # checking the response
    if response_checker(Y):

        for col in X.columns:

            # appending the response value to list
            Response.append(R_columns[0])
            # checking the type of predictor
            if predictor_checker(X, col):

                # appending the predictor
                Predictor.append(col + '(Categorical)')

                # appending the url to the list
                url = cat_response_cat_predictor(X, Y, col)
                Plot.append(url)

                # appending rest values
                t_score.append("NA")
                p_value.append("NA")
                MWR_Unweighted.append("NA")
                MWR_Weighted.append("NA")

            else:
                # appending the predictor
                Predictor.append(col + '(Continuous)')

                # appending the url to the list
                url = cat_resp_cont_predictor(X, col, Y, df)
                Plot.append(url)

                # appending the p and t values to the table
                t, p = p_t_values(df, col, Y)

                t_score.append(t)
                p_value.append(p)

                MWR_Unweighted.append("NA")
                MWR_Weighted.append("NA")

    else:

        for col in X.columns:
            Response.append(R_columns[0])

            if predictor_checker(X, col):

                # appending the predictor
                Predictor.append(col + '(Categorical)')

                # appending the url to the list
                url = cont_resp_cat_predictor(df, col, X, Y)
                Plot.append(url)


                t_score.append("NA")
                p_value.append("NA")
                MWR_Unweighted.append("NA")
                MWR_Weighted.append("NA")

            else:
                cont_list_pred.append(col)

                # appending the predictor
                Predictor.append(col + '(Continuous)')

                # appending the url to the list
                url = cont_response_cont_predictor( X, Y, col)
                Plot.append(url)

                # appending the p and t values to the table
                t, p = p_t_values(df, col, Y)

                t_score.append(t)
                p_value.append(p)

                MWR_Unweighted.append("NA")
                MWR_Weighted.append("NA")

    # initiating a data frame
    Summary_Table = pd.DataFrame()
    # storing all the results in a data frame and showing

    if response_checker(Y):
        Summary_Table['Response(Boolean)'] = Response
    else:
        Summary_Table['Response(Continuous)'] = Response

    Summary_Table['Predictor'] = Predictor
    Summary_Table['Plot'] = Plot
    Summary_Table['p-value'] = p_value
    Summary_Table['t-score'] = t_score
    Summary_Table['RF VarImp'] = RFVImp_score(Y, X)
    Summary_Table['MWR Unweighted'] = MWR_Unweighted
    Summary_Table['MWR Weighted'] = MWR_Weighted

    # print(Summary_Table)

    # saving the results in a html file

    f = open('finalfile.html', 'w')
    f.write(Summary_Table.to_html())
    f.close()

    #webbrowser.open_new_tab('/Users/naveenreddysama/Desktop/BDA602/BDA602-ML-/Assignment4/finalfile.html')

if __name__ == "__main__":
    sys.exit(main())