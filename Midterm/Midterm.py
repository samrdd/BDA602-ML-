# importing the necessary pakages
import itertools
import random
import sys
from itertools import combinations
from pathlib import Path
from typing import List

import numpy as np
import pandas
import plotly.express as px
import plotly.graph_objects as go
import seaborn
from scipy.stats import chi2_contingency
from sklearn import datasets
from sklearn.preprocessing import LabelEncoder

TITANIC_PREDICTORS = [
    "pclass",
    "sex",
    "age",
    "sibsp",
    "embarked",
    "parch",
    "fare",
    "who",
    "adult_male",
    "deck",
    "embark_town",
    "alone",
    "class",
]


def get_test_data_set(data_set_name: str = None) -> (pandas.DataFrame, List[str], str):
    """Function to load a few test data sets

    :param:
    data_set_name : string, optional
        Data set to load

    :return:
    data_set : :class:`pandas.DataFrame`
        Tabular data, possibly with some preprocessing applied.
    predictors :list[str]
        List of predictor variables
    response: str
        Response variable
    """
    seaborn_data_sets = ["mpg", "tips", "titanic", "titanic_2"]
    sklearn_data_sets = ["boston", "diabetes", "breast_cancer"]
    all_data_sets = seaborn_data_sets + sklearn_data_sets

    if data_set_name is None:
        data_set_name = random.choice(all_data_sets)
    else:
        if data_set_name not in all_data_sets:
            raise Exception(f"Data set choice not valid: {data_set_name}")

    if data_set_name in seaborn_data_sets:
        if data_set_name == "mpg":
            data_set = seaborn.load_dataset(name="mpg").dropna().reset_index()
            predictors = [
                "cylinders",
                "displacement",
                "horsepower",
                "weight",
                "acceleration",
                "origin",
                "name",
            ]
            response = "mpg"
        elif data_set_name == "tips":
            data_set = seaborn.load_dataset(name="tips").dropna().reset_index()
            predictors = [
                "total_bill",
                "sex",
                "smoker",
                "day",
                "time",
                "size",
            ]
            response = "tip"
        elif data_set_name == "titanic":
            data_set = seaborn.load_dataset(name="titanic").dropna()
            predictors = TITANIC_PREDICTORS
            response = "survived"
        elif data_set_name == "titanic_2":
            data_set = seaborn.load_dataset(name="titanic").dropna()
            predictors = TITANIC_PREDICTORS
            response = "alive"
    elif data_set_name in sklearn_data_sets:
        if data_set_name == "boston":
            data = datasets.load_boston()
            data_set = pandas.DataFrame(data.data, columns=data.feature_names)
            data_set["CHAS"] = data_set["CHAS"].astype(str)
        elif data_set_name == "diabetes":
            data = datasets.load_diabetes()
            data_set = pandas.DataFrame(data.data, columns=data.feature_names)
            data_set["gender"] = ["1" if i > 0 else "0" for i in data_set["sex"]]
        elif data_set_name == "breast_cancer":
            data = datasets.load_breast_cancer()
            data_set = pandas.DataFrame(data.data, columns=data.feature_names)

        data_set["target"] = data.target
        predictors = data.feature_names
        response = "target"

    print(f"Data set selected: {data_set_name}")

    return data_set_name, data_set, predictors, response


def check_dtype(column):

    """Ckecks the datatype of a given column"""

    if (
        column.dtypes == "object"
        or column.dtypes == "category"
        or len(pandas.unique(column)) == 2
    ):

        return 0

    else:

        return 1


def transform(dataframe, string):

    """Label encodes the data if a column is a categorical for a given list of predictors or a response"""

    label_encoder = LabelEncoder()

    if (
        dataframe[string].dtypes == "object"
        or dataframe[string].dtypes == "category"
        or len(pandas.unique(dataframe[string])) == 2
    ):

        dataframe[string] = label_encoder.fit_transform(dataframe[string])

    elif len(pandas.unique(dataframe[string])) > 2:

        pass

    return dataframe


def split_predictors(predictors, df):

    """Splits the given list of predictors to categorical and continuous predictors"""

    cat_pred, cont_pred = [], []

    for column_name in predictors:

        value = check_dtype(df[column_name])

        if value == 0:

            cat_pred.append(column_name)

        else:

            cont_pred.append(column_name)

    return cat_pred, cont_pred


def plot_link(fig, name):

    """Save the given plot as a link"""

    Path("./plots/").mkdir(parents=True, exist_ok=True)
    link = f"./plots/{name}.html"
    fig.write_html(link)

    return link


def make_clickable(val):

    """Make urls in dataframe clickable in html output"""

    if val is not None:
        if "," in val:
            x = val.split(",")
            return f'{x[0]} <a target="_blank" href="{x[1]}">plot</a>'
        else:
            return f'<a target="_blank" href="{val}">plot</a>'
    else:
        return


def cont_cont_plot(df, predictors, i):

    """Returns the scatter plot link of continuous and continuous pairs"""

    comb = list(combinations(predictors, 2))

    fig = px.scatter(df, df[comb[i][0]], df[comb[i][1]], trendline="ols")
    title = f"{comb[i][0]} Vs {comb[i][1]}"
    fig.update_layout(
        title=title,
        xaxis_title=f"{comb[i][0]}",
        yaxis_title=f"{comb[i][1]}",
    )

    # saving the url of the plot
    url = plot_link(fig, title)

    return url


def cat_cat_plot(df, predictors, i):

    """Returns the link of heat map of categorical and categorical pairs"""

    comb = list(combinations(predictors, 2))

    confusion_matrix = pandas.crosstab(df[comb[i][0]], df[comb[i][1]])

    fig = px.imshow(confusion_matrix, text_auto=True, aspect="auto")

    title = f"{comb[i][0]} Vs {comb[i][1]}"
    fig.update_layout(
        title=title,
    )
    url = plot_link(fig, title)

    return url


def cat_cont_plot(df, comb, i):

    """Returns the link of the distribution and violin plot of categorical and continuous pairs"""

    fig1 = px.histogram(
        df,
        x=df[comb[i][1]],
        color=df[comb[i][0]],
        marginal="rug",
        hover_data=df.columns,
    )
    title = f"Distribution Plot of {comb[i][0]} by {comb[i][1]}"
    fig1.update_layout(
        title=title,
        xaxis_title=f"{comb[i][0]}",
        yaxis_title=f"{comb[i][1]}",
    )
    # saving the url of the plot
    url1 = plot_link(fig1, title)

    fig2 = px.violin(
        df,
        y=df[comb[i][1]],
        x=df[comb[i][0]],
        box=True,
        color=df[comb[i][0]],
        hover_data=df.columns,
    )

    title = f"Volin Plot of {comb[i][0]} by {comb[i][1]}"
    fig2.update_layout(
        title=title,
        xaxis_title=f"{comb[i][0]}",
        yaxis_title=f"{comb[i][1]}",
    )
    # saving the url of the plot
    url2 = plot_link(fig2, title)

    return url1, url2


def table_style(table):

    """Applies style to a given table"""

    cell_hover = {  # for row hover use <tr> instead of <td>
        "selector": "td:hover",
        "props": [("background-color", "#D0DFFA")],
    }
    index_names = {
        "selector": ".index_name",
        "props": "font-style: italic; color: darkgrey; font-weight:normal;",
    }
    headers = {
        "selector": "th:not(.index_name)",
        "props": "background-color: #D0DFFA; color: black;",
    }

    table = table.set_table_styles([cell_hover, index_names, headers])

    table = table.set_table_styles(
        [
            {"selector": "th.col_heading", "props": "text-align: center;"},
            {"selector": "th.col_heading.level0", "props": "font-size: 1.5em;"},
            {"selector": "td", "props": "text-align: center; " "font-size: 1.2em;"},
            {"selector": "", "props": [("border", "2px black solid !important")]},
            {"selector": "tbody td", "props": [("border", "1px solid grey")]},
            {"selector": "th", "props": [("border", "1px solid grey")]},
        ],
        overwrite=False,
    )

    return table


def cont_cont_pairs(df, predictors):

    """Returns the correlation table of continuous and continuous pairs along with their plots"""

    table = pandas.DataFrame(
        columns=[
            "Predictors",
            "Correlation Coefficient",
            "Absolute Correlation Coefficient",
            "Plot_link",
        ]
    )

    comb = list(combinations(predictors, 2))

    for i in range(0, len(comb)):

        Predictor = f"{comb[i][0]} and {comb[i][1]}"

        corr_coeff = df[comb[i][0]].corr(df[comb[i][1]])

        Abs_coerr = round(abs(corr_coeff), 4)

        url = cont_cont_plot(df, predictors, i)

        table = table.append(
            {
                "Predictors": Predictor,
                "Correlation Coefficient": corr_coeff,
                "Absolute Correlation Coefficient": Abs_coerr,
                "Plot_link": url,
            },
            ignore_index=True,
        )

    table = table.sort_values(
        by="Absolute Correlation Coefficient", ascending=False
    ).reset_index(drop=True)

    # Make url clickable

    table = table.style.format(
        {
            "Plot_link": make_clickable,
        }
    )

    table = table_style(table)

    return table


def cramers_V(var1, var2):

    """Calculates the correlation of two categorical variables"""

    crosstab = np.array(
        pandas.crosstab(var1, var2, rownames=None, colnames=None)
    )  # Cross table building
    stat = chi2_contingency(crosstab)[
        0
    ]  # Keeping of the test statistic of the Chi2 test
    obs = np.sum(crosstab)  # Number of observations
    mini = (
        min(crosstab.shape) - 1
    )  # Take the minimum value between the columns and the rows of the cross table

    return stat / (obs * mini)


def cat_cat_pairs(df, predictors):

    """Returns the correlation table of categorical and categorical pairs along with their plots"""

    table = pandas.DataFrame(
        columns=["Predictors", "Correlation Coefficient", "Plot_link"]
    )

    comb = list(combinations(predictors, 2))

    for i in range(0, len(comb)):

        Predictor = f"{comb[i][0]} and {comb[i][1]}"

        corr_coeff = cramers_V(df[comb[i][0]], df[comb[i][1]])

        url = cat_cat_plot(df, predictors, i)

        table = table.append(
            {
                "Predictors": Predictor,
                "Correlation Coefficient": corr_coeff,
                "Plot_link": url,
            },
            ignore_index=True,
        )

    table = table.sort_values(
        by="Correlation Coefficient", ascending=False
    ).reset_index(drop=True)

    table = table.style.format(
        {
            "Plot_link": make_clickable,
        }
    )

    table = table_style(table)

    return table


def correlation_ratio(categories, values):

    """Calculates the correlation of a categorical and continuous variable"""

    f_cat, _ = pandas.factorize(categories)
    cat_num = np.max(f_cat) + 1
    y_avg_array = np.zeros(cat_num)
    n_array = np.zeros(cat_num)
    for i in range(0, cat_num):
        cat_measures = values[np.argwhere(f_cat == i).flatten()]
        n_array[i] = len(cat_measures)
        y_avg_array[i] = np.average(cat_measures)
    y_total_avg = np.sum(np.multiply(y_avg_array, n_array)) / np.sum(n_array)
    numerator = np.sum(
        np.multiply(n_array, np.power(np.subtract(y_avg_array, y_total_avg), 2))
    )
    denominator = np.sum(np.power(np.subtract(values, y_total_avg), 2))
    if numerator == 0:
        eta = 0.0
    else:
        eta = np.sqrt(numerator / denominator)
    return eta


def cat_cont_pairs(df, predictors):

    """Returns the correlation table of categorical and continuous pairs along with their plots"""

    cat_predictors, cont_predictors = split_predictors(predictors, df)

    Cat_cont_list = [cat_predictors, cont_predictors]

    comb = [p for p in itertools.product(*Cat_cont_list)]

    table = pandas.DataFrame(
        columns=[
            "Categorical Predictor",
            "Continuous Predictor",
            "Correlation Coefficient",
            "Distribution Plot",
            "Violin Plot",
        ]
    )

    for i in range(0, len(comb)):

        Predictor1 = comb[i][0]

        Predictor2 = comb[i][1]

        corr_coeff = correlation_ratio(
            np.array(df[comb[i][0]]), np.array(df[comb[i][1]])
        )

        url1, url2 = cat_cont_plot(df, comb, i)

        table = table.append(
            {
                "Categorical Predictor": Predictor1,
                "Continuous Predictor": Predictor2,
                "Correlation Coefficient": corr_coeff,
                "Distribution Plot": url1,
                "Violin Plot": url2,
            },
            ignore_index=True,
        )

    table = table.sort_values(
        by="Correlation Coefficient", ascending=False
    ).reset_index(drop=True)

    table = table.style.format(
        {
            "Distribution Plot": make_clickable,
            "Violin Plot": make_clickable,
        }
    )

    table = table_style(table)

    return table


def heatmap(Correlation_matrix, title):

    """Returns the heat map of given confusion matrix"""

    # plotting based on absolute values
    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            x=Correlation_matrix.columns,
            y=Correlation_matrix.index,
            z=np.array(Correlation_matrix),
            text=Correlation_matrix.values,
            texttemplate="%{text}",
            colorscale="Cividis_r",
        )
    )

    url = plot_link(fig, title)

    return url


def Corr_matrix_ContCont(dataframe, list):

    """Returns the url of correlation heat map of Continuous predictor pairs"""

    table = pandas.DataFrame(columns=["HeatMap of Cont_Cont Correlation Matrix"])

    data = pandas.concat(
        [pandas.DataFrame(dataframe[[list[i]]]) for i in range(len(list))], axis=1
    )

    Correlation_matrix = round(abs(data.corr()), 4)

    title = " Heat Map of Continuous Predictor Pairs "

    url = heatmap(Correlation_matrix, title)

    table = table.append(
        {"HeatMap of Cont_Cont Correlation Matrix": url}, ignore_index=True
    )

    table = table.style.format(
        {
            "HeatMap of Cont_Cont Correlation Matrix": make_clickable,
        }
    )

    table = table_style(table)

    return table


def Corr_matrix_CatCat(dataframe, list):

    """Returns the url of correlation heat map of categorical predictor pairs"""

    table = pandas.DataFrame(columns=["HeatMap of Cat_Cat Correlation Matrix"])

    data = pandas.concat(
        [pandas.DataFrame(dataframe[[list[i]]]) for i in range(len(list))], axis=1
    )

    rows = []
    for var1 in data:
        col = []
        for var2 in data:
            cramers = cramers_V(data[var1], data[var2])  # Cramer's V test
            col.append(
                round(cramers, 3)
            )  # Keeping of the rounded value of the Cramer's V
        rows.append(col)

    cramers_results = np.array(rows)
    Correlation_matrix = pandas.DataFrame(
        cramers_results, columns=data.columns, index=data.columns
    )

    title = " Heat Map of Categorical Predictor Pairs "

    url = heatmap(Correlation_matrix, title)

    table = table.append(
        {"HeatMap of Cat_Cat Correlation Matrix": url}, ignore_index=True
    )

    table = table.style.format(
        {
            "HeatMap of Cat_Cat Correlation Matrix": make_clickable,
        }
    )

    table = table_style(table)

    return table


def corr_martix_CatCont(dataframe, list):

    """Returns the url of correlation heat map of categorical and Continuous predictor pairs"""

    table = pandas.DataFrame(columns=["HeatMap of Cat_Cont Correlation Matrix"])

    cat_predictors, cont_predictors = split_predictors(list, dataframe)

    cat_data = pandas.concat(
        [
            pandas.DataFrame(dataframe[[cat_predictors[i]]])
            for i in range(len(cat_predictors))
        ],
        axis=1,
    )
    cont_data = pandas.concat(
        [
            pandas.DataFrame(dataframe[[cont_predictors[i]]])
            for i in range(len(cont_predictors))
        ],
        axis=1,
    )

    rows = []
    for var1 in cat_data:
        col = []
        for var2 in cont_data:
            correlationratio = correlation_ratio(
                np.array(cat_data[var1]), np.array(cont_data[var2])
            )  # Cramer's V test
            col.append(
                round(correlationratio, 3)
            )  # Keeping of the rounded value of the Cramer's V
        rows.append(col)

    cramers_results = np.array(rows)

    Correlation_matrix = pandas.DataFrame(
        cramers_results, columns=cont_data.columns, index=cat_data.columns
    )

    title = " Heat Map of Categorical and Continuous Predictor pairs "

    url = heatmap(Correlation_matrix, title)

    table = table.append(
        {"HeatMap of Cat_Cont Correlation Matrix": url}, ignore_index=True
    )

    table = table.style.format(
        {
            "HeatMap of Cat_Cont Correlation Matrix": make_clickable,
        }
    )

    table = table_style(table)

    return table


def BruteForce_ContCont(df, predictors, response):

    """Returns the Brute Force table of Continuous Predictor Pairs along with residual plots"""

    cat_predictors, cont_predictors = split_predictors(predictors, df)

    comb = list(combinations(cont_predictors, 2))

    table = pandas.DataFrame(
        columns=[
            "Predictor1",
            "Predictor2",
            "Difference with Mean of Response",
            "Difference with Mean of Response(Weighted)",
            "Residual Plot link",
        ]
    )

    for i in range(0, len(comb)):

        Predictor1 = comb[i][0]
        Predictor2 = comb[i][1]

        meanofresponse_table = pandas.DataFrame()

        population_mean = df[response].mean()

        temp_df = df[[comb[i][0], comb[i][1], response]].copy()

        temp_df["Predictor_Bins_1"] = pandas.cut(
            temp_df[comb[i][0]], 10, include_lowest=True, duplicates="drop"
        )

        temp_df["Predictor_Bins_2"] = pandas.cut(
            temp_df[comb[i][1]], 10, include_lowest=True, duplicates="drop"
        )

        temp_df["LowerBin_1"] = (
            temp_df["Predictor_Bins_1"].apply(lambda x: x.left).astype(float)
        )
        temp_df["UpperBin_1"] = (
            temp_df["Predictor_Bins_1"].apply(lambda x: x.right).astype(float)
        )
        temp_df["LowerBin_2"] = (
            temp_df["Predictor_Bins_2"].apply(lambda x: x.left).astype(float)
        )
        temp_df["UpperBin_2"] = (
            temp_df["Predictor_Bins_2"].apply(lambda x: x.right).astype(float)
        )

        temp_df["BinCentre_1"] = (temp_df["LowerBin_1"] + temp_df["UpperBin_1"]) / 2

        temp_df["BinCentre_2"] = (temp_df["LowerBin_2"] + temp_df["UpperBin_2"]) / 2

        bin_mean = temp_df.groupby(by=["BinCentre_1", "BinCentre_2"]).mean()
        bin_count = temp_df.groupby(by=["BinCentre_1", "BinCentre_2"]).count()

        meanofresponse_table["BinCount"] = bin_count[response]
        meanofresponse_table["BinMean"] = bin_mean[response]

        meanofresponse_table["Population_Mean"] = population_mean

        meanofresponse_table["Mean_diff"] = round(
            meanofresponse_table["BinMean"] - meanofresponse_table["Population_Mean"], 6
        )

        meanofresponse_table["mean_squared_diff"] = round(
            (meanofresponse_table["Mean_diff"]) ** 2, 6
        )

        meanofresponse_table["Weight"] = (
            meanofresponse_table["BinCount"] / df[response].count()
        )
        meanofresponse_table["mean_squared_diff_weighted"] = (
            meanofresponse_table["mean_squared_diff"] * meanofresponse_table["Weight"]
        )
        meanofresponse_table = meanofresponse_table.reset_index()

        mean_squared_diff = round(
            (meanofresponse_table["mean_squared_diff"].sum())
            / len(meanofresponse_table.axes[0]),
            6,
        )
        mean_squared_diff_weighted = round(
            meanofresponse_table["mean_squared_diff_weighted"].sum(), 6
        )

        fig = go.Figure()
        fig.add_trace(
            go.Heatmap(
                x=meanofresponse_table["BinCentre_1"].values,
                y=meanofresponse_table["BinCentre_2"].values,
                z=meanofresponse_table["Mean_diff"],
                text=meanofresponse_table["Mean_diff"],
                texttemplate="%{text}",
            )
        )

        title = f"Mean Square of Response of {comb[i][0]} by {comb[i][1]}"

        fig.update_layout(
            title=title,
            xaxis_title=f"{comb[i][0]}",
            yaxis_title=f"{comb[i][1]}",
        )

        url = plot_link(fig, title)

        table = table.append(
            {
                "Predictor1": Predictor1,
                "Predictor2": Predictor2,
                "Difference with Mean of Response": mean_squared_diff,
                "Difference with Mean of Response(Weighted)": mean_squared_diff_weighted,
                "Residual Plot link": url,
            },
            ignore_index=True,
        )

    table = table.sort_values(
        by="Difference with Mean of Response(Weighted)", ascending=False
    ).reset_index(drop=True)

    table = table.style.format(
        {
            "Residual Plot link": make_clickable,
        }
    )

    table = table_style(table)

    return table


def BruteForce_CatCat(df, predictors, response):

    """Returns the Brute Force table of Categorical Predictor Pairs along with Residual Plots"""

    cat_predictors, cont_predictors = split_predictors(predictors, df)

    comb = list(combinations(cat_predictors, 2))

    table = pandas.DataFrame(
        columns=[
            "Predictor1",
            "Predictor2",
            "Difference with Mean of Response",
            "Difference with Mean of Response(Weighted)",
            "Residual Plot link",
        ]
    )

    for i in range(0, len(comb)):

        Predictor1 = comb[i][0]
        Predictor2 = comb[i][1]

        temp_df = df[[comb[i][0], comb[i][1], response]].copy()

        meanofresponse_table = pandas.DataFrame()

        population_mean = df[response].mean()

        bin_count = temp_df.groupby(by=[df[comb[i][0]], df[comb[i][1]]]).count()
        bin_mean = temp_df.groupby(by=[df[comb[i][0]], df[comb[i][1]]]).mean()

        meanofresponse_table["BinCount"] = bin_count[response]
        meanofresponse_table["BinMean"] = bin_mean[response]

        meanofresponse_table["PopulationMean"] = population_mean
        diff_of_mean = (
            meanofresponse_table["BinMean"] - meanofresponse_table["PopulationMean"]
        )

        meanofresponse_table["diff_of_mean"] = round(diff_of_mean, 6)

        mean_squared_difference = (
            meanofresponse_table["BinMean"] - meanofresponse_table["PopulationMean"]
        ) ** 2

        meanofresponse_table["mean_squared_diff"] = round(mean_squared_difference, 6)
        meanofresponse_table["Weight"] = (
            meanofresponse_table["BinCount"] / df[comb[i][0]].count()
        )
        meanofresponse_table["mean_squared_diff_weighted"] = (
            meanofresponse_table["mean_squared_diff"] * meanofresponse_table["Weight"]
        )
        meanofresponse_table = meanofresponse_table.reset_index()

        mean_squared_diff = round(
            (meanofresponse_table["mean_squared_diff"].sum())
            / len(meanofresponse_table.axes[0]),
            6,
        )
        mean_squared_diff_weighted = round(
            meanofresponse_table["mean_squared_diff_weighted"].sum(), 6
        )

        fig = go.Figure()
        fig.add_trace(
            go.Heatmap(
                x=meanofresponse_table[comb[i][0]].values,
                y=meanofresponse_table[comb[i][1]].values,
                z=meanofresponse_table["mean_squared_diff"],
                text=meanofresponse_table["mean_squared_diff"],
                texttemplate="%{text}",
            )
        )

        title = f"Mean Square of Response of {comb[i][0]} by {comb[i][1]}"

        fig.update_layout(
            title=title,
            yaxis_title=f"{comb[i][0]}",
            xaxis_title=f"{comb[i][1]}",
        )

        url = plot_link(fig, title)

        table = table.append(
            {
                "Predictor1": Predictor1,
                "Predictor2": Predictor2,
                "Difference with Mean of Response": mean_squared_diff,
                "Difference with Mean of Response(Weighted)": mean_squared_diff_weighted,
                "Residual Plot link": url,
            },
            ignore_index=True,
        )

    table = table.sort_values(
        by="Difference with Mean of Response(Weighted)", ascending=False
    ).reset_index(drop=True)

    table = table.style.format(
        {
            "Residual Plot link": make_clickable,
        }
    )

    table = table_style(table)

    return table


def BruteForce_CatCont(df, predictors, response):

    """Returns the Brute Force table of Categorical and Continuous Predictor pairs along with residual plots"""

    cat_predictors, cont_predictors = split_predictors(predictors, df)

    Cat_cont_list = [cat_predictors, cont_predictors]

    comb = [p for p in itertools.product(*Cat_cont_list)]

    table = pandas.DataFrame(
        columns=[
            "Categorical Predictor",
            "Continuous Predictor",
            "Difference with Mean of Response",
            "Difference with Mean of Response(Weighted)",
            "Residual Plot link",
        ]
    )

    for i in range(0, len(comb)):

        Predictor1 = comb[i][0]
        Predictor2 = comb[i][1]

        meanofresponse_table = pandas.DataFrame()

        population_mean = df[response].mean()

        temp_df = df[[comb[i][1], comb[i][0], response]].copy()

        temp_df["Bins"] = pandas.cut(
            temp_df[comb[i][1]], 10, include_lowest=True, duplicates="drop"
        )

        temp_df["LowerBin"] = temp_df["Bins"].apply(lambda x: x.left).astype(float)
        temp_df["UpperBin"] = temp_df["Bins"].apply(lambda x: x.right).astype(float)

        temp_df["BinCentre"] = (temp_df["LowerBin"] + temp_df["UpperBin"]) / 2

        bin_count = temp_df.groupby(by=["BinCentre", df[comb[i][0]]]).count()
        bin_mean = temp_df.groupby(by=["BinCentre", df[comb[i][0]]]).mean()

        meanofresponse_table["BinCount"] = bin_count[response]
        meanofresponse_table["BinMean"] = bin_mean[response]

        meanofresponse_table["PopulationMean"] = population_mean

        meanofresponse_table["Mean_diff"] = round(
            meanofresponse_table["BinMean"] - meanofresponse_table["PopulationMean"], 6
        )

        mean_squared_difference = (meanofresponse_table["Mean_diff"]) ** 2

        meanofresponse_table["mean_squared_diff"] = mean_squared_difference

        meanofresponse_table["Weight"] = (
            meanofresponse_table["BinCount"] / df[comb[i][1]].count()
        )
        meanofresponse_table["mean_squared_diff_weighted"] = (
            meanofresponse_table["mean_squared_diff"] * meanofresponse_table["Weight"]
        )
        meanofresponse_table = meanofresponse_table.reset_index()

        mean_squared_diff = round(
            meanofresponse_table["mean_squared_diff"].sum()
            / len(meanofresponse_table.axes[0]),
            6,
        )
        mean_squared_diff_weighted = round(
            meanofresponse_table["mean_squared_diff_weighted"].sum(), 6
        )

        fig = go.Figure()
        fig.add_trace(
            go.Heatmap(
                x=meanofresponse_table["BinCentre"].values,
                y=meanofresponse_table[comb[i][0]].values,
                z=meanofresponse_table["mean_squared_diff"],
                text=meanofresponse_table["Mean_diff"],
                texttemplate="%{text}",
            )
        )

        title = f"Mean Square of Response of {comb[i][0]} by {comb[i][1]}"

        fig.update_layout(
            title=title,
            yaxis_title=f"{comb[i][0]}",
            xaxis_title=f"{comb[i][1]}",
        )

        url = plot_link(fig, title)

        table = table.append(
            {
                "Categorical Predictor": Predictor1,
                "Continuous Predictor": Predictor2,
                "Difference with Mean of Response": mean_squared_diff,
                "Difference with Mean of Response(Weighted)": mean_squared_diff_weighted,
                "Residual Plot link": url,
            },
            ignore_index=True,
        )

    table = table.sort_values(
        by="Difference with Mean of Response(Weighted)", ascending=False
    ).reset_index(drop=True)

    table = table.style.format(
        {
            "Residual Plot link": make_clickable,
        }
    )

    table = table_style(table)

    return table


def main():

    """loading the data"""
    Name, df, predictors, response = get_test_data_set()

    """Checking and transforming the response"""
    df = transform(df, response)

    """Getting the list of cat and cont predictors"""
    cat_predictors, cont_predictors = split_predictors(predictors, df)

    """creating a html file for output"""
    file_html = open("MidTerm.html", "w")

    file_html.write(
        "<h1><center>Mid Term</center></h1>"
        "<h1><center>Data Set: %s</center></h1>"
        "<h1><center>Continuous Vs Continuous Predictor Pairs</center></h1>" % Name
    )

    """Checking if there are continuous predictors in the list of predictors"""
    if len(cont_predictors) <= 1:

        file_html.write("<h2><center>No Continuous pairs</center></h2>")

    elif len(cont_predictors) > 1:

        file_html.write("<h2><center>Correlation Table</center></h2>")

        Cont_cont_CorrTable = cont_cont_pairs(df, cont_predictors).to_html()

        file_html.write("<body><center>%s</center></body>" % Cont_cont_CorrTable)

        file_html.write("<h2><center>Confusion Matrix</center></h2>")

        Corr_ContCont = Corr_matrix_ContCont(df, cont_predictors).to_html()

        file_html.write("<body><center>%s</center></body>" % Corr_ContCont)

        file_html.write("<h2><center>Brute Force Table</center></h2>")

        BruteForce_ContCont_table = BruteForce_ContCont(
            df, predictors, response
        ).to_html()

        file_html.write("<body><center>%s</center></body>" % BruteForce_ContCont_table)

    file_html.write(
        "<h1><center>Categorical Vs Categorical Predictor Pairs </center></h1>"
    )

    """Checking if there are Categorical predictors in the list of predictors"""
    if len(cat_predictors) <= 1:

        file_html.write("<h2><center>No Categorical Pairs</center></h2>")

    elif len(cat_predictors) > 1:

        file_html.write("<h2><center>Correlation Table</center></h2>")

        Cat_Cat_CorrTable = cat_cat_pairs(df, cat_predictors).to_html()

        file_html.write("<body><center>%s</center></body>" % Cat_Cat_CorrTable)

        file_html.write("<h2><center>Confusion Matrix</center></h2>")

        Corr_CatCat = Corr_matrix_CatCat(df, cat_predictors).to_html()

        file_html.write("<body><center>%s</center></body>" % Corr_CatCat)

        file_html.write("<h2><center>Brute Force Table</center></h2>")

        BruteForce_CatCat_table = BruteForce_CatCat(df, predictors, response).to_html()

        file_html.write("<body><center>%s</center></body>" % BruteForce_CatCat_table)

    file_html.write(
        "<h1><center>Categorical Vs Continuous Predictor Pairs</center></h1>"
    )

    """Checking if there are Categorical and Continuous predictor pairs"""

    if len(cat_predictors) == 0 or len(cont_predictors) == 0:

        file_html.write("<h2><center>No Categorical and Continuous Pairs</center></h2>")

    else:

        file_html.write("<h2><center>Correlation Table</center></h2>")

        Cat_Cont_Corrtable = cat_cont_pairs(df, predictors).to_html()

        file_html.write("<body><center>%s</center></body>" % Cat_Cont_Corrtable)

        file_html.write("<h2><center>Confusion Matrix<center></h2>")

        Corr_CatCont = corr_martix_CatCont(df, predictors).to_html()

        file_html.write("<body><center>%s</center></body>" % Corr_CatCont)

        file_html.write("<h2><center>Brute Force Table</center></h2>")

        BruteForce_CatCont_table = BruteForce_CatCont(
            df, predictors, response
        ).to_html()

        file_html.write("<body><center>%s</center></body>" % BruteForce_CatCont_table)

    file_html.close()


if __name__ == "__main__":
    sys.exit(main())
