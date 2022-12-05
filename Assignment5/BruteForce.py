import itertools
from itertools import combinations

import pandas
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from Plots import plot_link
from tablestyle import make_clickable, table_style


def check_datatype(column):
    """Ckecks the datatype of a given column"""

    if (
        column.dtypes == "object"
        or column.dtypes == "category"
        or len(pandas.unique(column)) == 2
    ):

        return 0

    else:

        return 1


def split_predictors(predictors, df):
    """Splits the given list of predictors to categorical and continuous predictors
    :param predictors: the predictor variables in the data set
    :param df: dataset
    :return: categorical and continuous predictors list
    """

    cat_pred, cont_pred = [], []

    for column_name in predictors:

        value = check_datatype(df[column_name])

        if value == 0:

            cat_pred.append(column_name)

        else:

            cont_pred.append(column_name)

    return cat_pred, cont_pred


class MeanSquareDifference:
    @staticmethod
    def table(df, predictors, response):

        """Returns the mean square difference table with plots of all predictors"""

        table = pandas.DataFrame(
            columns=[
                "Predictor",
                "Difference with Mean of Response",
                "Difference with Mean of Response(Weighted)",
                "Plot link",
            ]
        )

        for i in range(0, len(predictors)):

            predictor = predictors[i]

            mean_of_response_table = pandas.DataFrame()

            population_mean = df[response].mean()

            temp_df = df[[predictors[i], response]].copy()

            temp_df["Predictor_Bins"] = pandas.cut(
                temp_df[predictors[i]], 10, include_lowest=True, duplicates="drop"
            )

            temp_df["LowerBin"] = (
                temp_df["Predictor_Bins"].apply(lambda x: x.left).astype(float)
            )
            temp_df["UpperBin"] = (
                temp_df["Predictor_Bins"].apply(lambda x: x.right).astype(float)
            )

            temp_df["BinCentre"] = (temp_df["LowerBin"] + temp_df["UpperBin"]) / 2

            bin_mean = temp_df.groupby(by=["LowerBin", "UpperBin", "BinCentre"]).mean()
            bin_count = temp_df.groupby(
                by=["LowerBin", "UpperBin", "BinCentre"]
            ).count()

            mean_of_response_table["BinCount"] = bin_count[response]
            mean_of_response_table["BinMean"] = bin_mean[response]

            mean_of_response_table["Population_Mean"] = population_mean

            mean_of_response_table["Mean_diff"] = round(
                mean_of_response_table["BinMean"]
                - mean_of_response_table["Population_Mean"],
                6,
            )

            mean_of_response_table["mean_squared_diff"] = round(
                (mean_of_response_table["Mean_diff"]) ** 2, 6
            )

            mean_of_response_table["Weight"] = (
                mean_of_response_table["BinCount"] / df[response].count()
            )
            mean_of_response_table["mean_squared_diff_weighted"] = (
                mean_of_response_table["mean_squared_diff"]
                * mean_of_response_table["Weight"]
            )
            mean_of_response_table = mean_of_response_table.reset_index()

            mean_of_response_table["mean_squared_diff"] = mean_of_response_table[
                "mean_squared_diff"
            ].mean()

            mean_squared_diff = round(
                (mean_of_response_table["mean_squared_diff"].sum()),
                6,
            )
            mean_squared_diff_weighted = round(
                mean_of_response_table["mean_squared_diff_weighted"].sum(), 6
            )

            # adding plots

            x_axis = mean_of_response_table["BinCentre"]

            fig = make_subplots(specs=[[{"secondary_y": True}]])

            fig.add_trace(
                go.Bar(
                    x=x_axis,
                    y=mean_of_response_table["BinCount"],
                    name="Population",
                    opacity=0.5,
                ),
                secondary_y=True,
            )

            fig.add_trace(
                go.Scatter(
                    x=x_axis, y=mean_of_response_table["BinMean"], name="Bin Mean"
                ),
                secondary_y=False,
            )

            fig.add_trace(
                go.Scatter(
                    x=[x_axis.min(), x_axis.max()],
                    y=[
                        mean_of_response_table["Population_Mean"][0],
                        mean_of_response_table["Population_Mean"][0],
                    ],
                    mode="lines",
                    line=dict(color="green", width=2),
                    name="Population Mean",
                )
            )

            fig.add_hline(
                mean_of_response_table["Population_Mean"][0], line_color="green"
            )

            title = f"Bin Difference with Mean of Response vs Bin ({predictors[i]})"
            # Add figure title
            fig.update_layout(
                title_text=f"<b>Bin Difference with Mean of Response of ({predictors[i]})<b>"
            )

            # Set x-axis title
            fig.update_xaxes(title_text=f"<b>Predictor({predictors[i]}) Bin<b>")

            # Set y-axes titles
            fig.update_yaxes(title_text="<b>Population</b>", secondary_y=True)
            fig.update_yaxes(
                title_text=f"<b>Response({response})</b>", secondary_y=False
            )

            url = plot_link(fig, title)

            table = table.append(
                {
                    "Predictor": predictor,
                    "Difference with Mean of Response": mean_squared_diff,
                    "Difference with Mean of Response(Weighted)": mean_squared_diff_weighted,
                    "Plot link": url,
                },
                ignore_index=True,
            )

        table = table.sort_values(
            by="Difference with Mean of Response(Weighted)", ascending=False
        ).reset_index(drop=True)

        table = table.style.format(
            {
                "Plot link": make_clickable,
            }
        )

        table = table_style(table)

        return table


class BruteForce:
    """Returns the brute of table along with the plots of respective tables"""

    @staticmethod
    def cont_cont(df, predictors, response):

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
            predictor_1 = comb[i][0]
            predictor_2 = comb[i][1]

            mean_of_response_table = pandas.DataFrame()

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

            mean_of_response_table["BinCount"] = bin_count[response]
            mean_of_response_table["BinMean"] = bin_mean[response]

            mean_of_response_table["Population_Mean"] = population_mean

            mean_of_response_table["Mean_diff"] = round(
                mean_of_response_table["BinMean"]
                - mean_of_response_table["Population_Mean"],
                6,
            )

            mean_of_response_table["mean_squared_diff"] = round(
                (mean_of_response_table["Mean_diff"]) ** 2, 6
            )

            mean_of_response_table["Weight"] = (
                mean_of_response_table["BinCount"] / df[response].count()
            )
            mean_of_response_table["mean_squared_diff_weighted"] = (
                mean_of_response_table["mean_squared_diff"]
                * mean_of_response_table["Weight"]
            )
            mean_of_response_table = mean_of_response_table.reset_index()

            mean_of_response_table["mean_squared_diff"] = mean_of_response_table[
                "mean_squared_diff"
            ].mean()

            mean_squared_diff = round(
                (mean_of_response_table["mean_squared_diff"].sum()),
                6,
            )
            mean_squared_diff_weighted = round(
                mean_of_response_table["mean_squared_diff_weighted"].sum(), 6
            )

            fig = go.Figure()
            fig.add_trace(
                go.Heatmap(
                    x=mean_of_response_table["BinCentre_1"].values,
                    y=mean_of_response_table["BinCentre_2"].values,
                    z=mean_of_response_table["Mean_diff"],
                    text=mean_of_response_table["Mean_diff"],
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
                    "Predictor1": predictor_1,
                    "Predictor2": predictor_2,
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

    @staticmethod
    def cat_cat(df, predictors, response):

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
            predictor_1 = comb[i][0]
            predictor_2 = comb[i][1]

            temp_df = df[[comb[i][0], comb[i][1], response]].copy()

            mean_of_response_table = pandas.DataFrame()

            population_mean = df[response].mean()

            bin_count = temp_df.groupby(by=[df[comb[i][0]], df[comb[i][1]]]).count()
            bin_mean = temp_df.groupby(by=[df[comb[i][0]], df[comb[i][1]]]).mean()

            mean_of_response_table["BinCount"] = bin_count[response]
            mean_of_response_table["BinMean"] = bin_mean[response]

            mean_of_response_table["PopulationMean"] = population_mean
            diff_of_mean = (
                mean_of_response_table["BinMean"]
                - mean_of_response_table["PopulationMean"]
            )

            mean_of_response_table["diff_of_mean"] = round(diff_of_mean, 6)

            mean_squared_difference = (
                mean_of_response_table["BinMean"]
                - mean_of_response_table["PopulationMean"]
            ) ** 2

            mean_of_response_table["mean_squared_diff"] = round(
                mean_squared_difference, 6
            )
            mean_of_response_table["Weight"] = (
                mean_of_response_table["BinCount"] / df[comb[i][0]].count()
            )
            mean_of_response_table["mean_squared_diff_weighted"] = (
                mean_of_response_table["mean_squared_diff"]
                * mean_of_response_table["Weight"]
            )
            mean_of_response_table = mean_of_response_table.reset_index()

            mean_squared_diff = round(
                (mean_of_response_table["mean_squared_diff"].sum())
                / len(mean_of_response_table.axes[0]),
                6,
            )
            mean_squared_diff_weighted = round(
                mean_of_response_table["mean_squared_diff_weighted"].sum(), 6
            )

            fig = go.Figure()
            fig.add_trace(
                go.Heatmap(
                    x=mean_of_response_table[comb[i][0]].values,
                    y=mean_of_response_table[comb[i][1]].values,
                    z=mean_of_response_table["mean_squared_diff"],
                    text=mean_of_response_table["mean_squared_diff"],
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
                    "Predictor1": predictor_1,
                    "Predictor2": predictor_2,
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

    @staticmethod
    def cat_cont(df, predictors, response):

        """Returns the Brute Force table of Categorical and Continuous Predictor pairs along with residual plots"""

        cat_predictors, cont_predictors = split_predictors(predictors, df)

        cat_cont_list = [cat_predictors, cont_predictors]

        comb = [p for p in itertools.product(*cat_cont_list)]

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
            predictor_1 = comb[i][0]
            predictor_2 = comb[i][1]

            mean_of_response_table = pandas.DataFrame()

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

            mean_of_response_table["BinCount"] = bin_count[response]
            mean_of_response_table["BinMean"] = bin_mean[response]

            mean_of_response_table["PopulationMean"] = population_mean

            mean_of_response_table["Mean_diff"] = round(
                mean_of_response_table["BinMean"]
                - mean_of_response_table["PopulationMean"],
                6,
            )

            mean_squared_difference = (mean_of_response_table["Mean_diff"]) ** 2

            mean_of_response_table["mean_squared_diff"] = mean_squared_difference

            mean_of_response_table["Weight"] = (
                mean_of_response_table["BinCount"] / df[comb[i][1]].count()
            )
            mean_of_response_table["mean_squared_diff_weighted"] = (
                mean_of_response_table["mean_squared_diff"]
                * mean_of_response_table["Weight"]
            )
            mean_of_response_table = mean_of_response_table.reset_index()

            mean_of_response_table["mean_squared_diff"] = mean_of_response_table[
                "mean_squared_diff"
            ].mean()

            mean_squared_diff = round(
                (mean_of_response_table["mean_squared_diff"].sum()),
                6,
            )
            mean_squared_diff_weighted = round(
                mean_of_response_table["mean_squared_diff_weighted"].sum(), 6
            )

            fig = go.Figure()
            fig.add_trace(
                go.Heatmap(
                    x=mean_of_response_table["BinCentre"].values,
                    y=mean_of_response_table[comb[i][0]].values,
                    z=mean_of_response_table["mean_squared_diff"],
                    text=mean_of_response_table["Mean_diff"],
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
                    "Categorical Predictor": predictor_1,
                    "Continuous Predictor": predictor_2,
                    "Difference with Mean of Response": mean_squared_diff / 100,
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
