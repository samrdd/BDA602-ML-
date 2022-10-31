# importing necessary libraries
import sys
import pandas as pd
import numpy
from plotly import express as px
from plotly import figure_factory as ff
from plotly import graph_objects as go
from sklearn.metrics import confusion_matrix
import plotly.express as px
import statsmodels.api
from sklearn.ensemble import RandomForestRegressor

# Using the different data to test the code

def Cont_resp():
    # loading the data
    data = pd.read_csv("/content/drive/MyDrive/auto-mpg.csv")

    # Separating the data to target and response variables
    X = data[["cylinders", "displacement", "weight", "acceleration", "model year", "origin"]]  # PREDICTOR
    Y = data[["mpg"]]  # RESPONSE

    return data, X, Y


def Cat_resp():
    # loading the data
    data = pd.read_csv("/content/drive/MyDrive/heart_failure_clinical_records_dataset.csv")

    # Separating the data to target and response variables
    X = data[
        ["creatinine_phosphokinase", "diabetes", "ejection_fraction", "high_blood_pressure", "platelets", "platelets",
         "sex", "smoking", "time"]]  # PREDICTOR
    Y = data[["DEATH_EVENT"]]  # RESPONSE

    return data, X, Y


def Cat_resp2():
    data = pd.read_csv("/content/drive/MyDrive/heart.csv")

    # Separating the data to target and response variables
    X = data[["age", "sex", "cp", "trtbps", "chol", "fbs", "restecg", "thalachh", "exng", "oldpeak", "slp", "caa",
              "thall"]]  # PREDICTOR
    Y = data[["output"]]  # RESPONSE

    return data, X, Y


# Calling the Continous response function

data, X, Y = Cat_resp2()

# Creating a table to store the results
Summary_Table = pd.DataFrame()

# initiating the lists to store the data in a list wrt
Response, Predictor, Plot, p_value, t_score, RF_VarImp, MWR_Unweighted, MWR_Weighted = [], [], [], [], [], [], [], []

# calculatin the random forest variable importance
model = RandomForestRegressor()
model.fit(X, Y)


# continuous predictor vs continuous response
def cat_response_cat_predictor():
    x = numpy.array(X.iloc[:, i])
    y = numpy.array(Y)

    X_2 = [1 if abs(x_) > 0.5 else 0 for x_ in x]
    Y_2 = [1 if abs(y_) > 0.5 else 0 for y_ in y]

    conf_matrix = confusion_matrix(X_2, Y_2)

    fig = go.Figure(
        data=go.Heatmap(z=conf_matrix, zmin=0, zmax=conf_matrix.max())
    )
    fig.update_layout(
        title="Categorical Predictor by Categorical Response (with relationship)",
        xaxis_title="Response",
        yaxis_title="Predictor",
    )
    fig.show()

    return


# Cat response continuous predictor

def cat_resp_cont_predictor():
    fig = px.histogram(data, x=X.columns[i], color=Y.columns[0], marginal="rug", hover_data=X.columns)

    fig.update_layout(
        title="Continuous Predictor by Categorical Response",
        xaxis_title="Predictor",
        yaxis_title="Distribution",
    )
    fig.show()
    return


# Cont Response Cat Predictor

def cont_resp_cat_predictor():
    fig = px.violin(data, y=Y.columns[0], x=X.columns[i], box=True, hover_data=X.columns)
    fig.show()
    return


# Cont Response Cont Predictor

def cont_response_cont_predictor():
    x = X.iloc[:, i]
    y = Y.iloc[:, 0]

    fig = px.scatter(x=x, y=y, trendline="ols")
    fig.update_layout(
        title="Continuous Response by Continuous Predictor",
        xaxis_title="Predictor",
        yaxis_title="Response",
    )
    fig.show()

    return


# looping the columns to verify the data type

P_Columns = list(X.columns)
R_columns = list(Y.columns)
Y1 = Y.iloc[:, 0]

if len(pd.unique(Y1)) <= 2:
    for i in range(0, X.shape[1]):
        Response.append(R_columns[0])
        n = len(pd.unique(X.iloc[:, i]))

        if n <= 4:

            # appending the column name along with its type
            Predictor.append(P_Columns[i] + '(Categorical)')

            # calling the plot for this type
            cat_response_cat_predictor()

            t_score.append("NA")
            Plot.append("NA")
            p_value.append("NA")
            RF_VarImp.append("NA")
            MWR_Unweighted.append("NA")
            MWR_Weighted.append("NA")
        else:

            # appending the column name along with its type
            Predictor.append(P_Columns[i] + '(Continuous)')

            # Calling the plot function of this type
            cat_resp_cont_predictor()

            # p and t value calculation using logistic regression
            predict = statsmodels.api.add_constant(X.iloc[:, i])
            logistic_regression_model = statsmodels.api.Logit(Y, predict)
            logistic_regression_model_fitted = logistic_regression_model.fit()
            t = round(logistic_regression_model_fitted.tvalues[1], 6)
            p = "{:.6e}".format(logistic_regression_model_fitted.pvalues[1])

            # appending the p and t values to the table
            t_score.append(t)
            p_value.append(p)

            # calculatin the random forest variable importance features
            importance = model.feature_importances_
            value = importance[i]

            # appending the random forest variable to the table
            RF_VarImp.append(value)

            Plot.append("NA")
            MWR_Unweighted.append("NA")
            MWR_Weighted.append("NA")

if len(pd.unique(Y1)) > 2:

    for i in range(0, X.shape[1]):
        Response.append(R_columns[0])

        n = len(pd.unique(X.iloc[:, i]))
        if n <= 4:

            # appending the column name along with its type
            Predictor.append(P_Columns[i] + '(Categorical)')

            # calling the plot function of this type
            cont_resp_cat_predictor()

            t_score.append("NA")
            Plot.append("NA")
            p_value.append("NA")
            RF_VarImp.append("NA")
            MWR_Unweighted.append("NA")
            MWR_Weighted.append("NA")

        else:

            # appending the column name along with its type
            Predictor.append(P_Columns[i] + '(Continuous)')

            # calling the plot function
            cont_response_cont_predictor()

            # p and t value calculation using logistic regression method
            predict = statsmodels.api.add_constant(X.iloc[:, i])
            linear_regression_model = statsmodels.api.OLS(Y, predict)
            linear_regression_model_fitted = linear_regression_model.fit()
            t = round(linear_regression_model_fitted.tvalues[1], 6)
            p = "{:.6e}".format(linear_regression_model_fitted.pvalues[1])

            # appending the p and t values to the table
            t_score.append(t)
            p_value.append(p)

            importance = model.feature_importances_
            values = importance[i]
            # appending the random forest variable to the table
            RF_VarImp.append(values)

            Plot.append("NA")
            MWR_Unweighted.append("NA")
            MWR_Weighted.append("NA")

# storing all the results in a data frame and showing

if len(pd.unique(Y1)) < 2:
    Summary_Table['Response(Boolean)'] = Response
else:
    Summary_Table['Response(Continuous)'] = Response

Summary_Table['Predictor'] = Predictor
Summary_Table['Plot'] = Plot
Summary_Table['p-value'] = p_value
Summary_Table['t-score'] = t_score
Summary_Table['RF VarImp'] = RF_VarImp
Summary_Table['MWR Unweighted'] = MWR_Unweighted
Summary_Table['MWR Weighted'] = MWR_Weighted

Summary_Table
