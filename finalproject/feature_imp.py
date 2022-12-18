import pandas
import plotly.graph_objects as go
import statsmodels.api
from BruteForce import check_datatype
from plotly.subplots import make_subplots
from Plots import Plot, plot_link
from sklearn.ensemble import RandomForestClassifier
from tablestyle import make_clickable, table_style


def msd(df, predictor, response):

    """Returns the mean square difference unweighted and weighted with plot of a given predictor"""

    mean_of_response_table = pandas.DataFrame()
    population_mean = df[response].mean()
    temp_df = df[[predictor, response]].copy()

    if check_datatype(df[predictor]) == 1:

        temp_df["Predictor_Bins"] = pandas.cut(
            temp_df[predictor], 10, include_lowest=True, duplicates="drop"
        )

        temp_df["LowerBin"] = (
            temp_df["Predictor_Bins"].apply(lambda x: x.left).astype(float)
        )
        temp_df["UpperBin"] = (
            temp_df["Predictor_Bins"].apply(lambda x: x.right).astype(float)
        )

        temp_df["BinCentre"] = (temp_df["LowerBin"] + temp_df["UpperBin"]) / 2

        bin_mean = temp_df.groupby(by=["LowerBin", "UpperBin", "BinCentre"]).mean()
        bin_count = temp_df.groupby(by=["LowerBin", "UpperBin", "BinCentre"]).count()

        mean_of_response_table["BinCount"] = bin_count[response]
        mean_of_response_table["BinMean"] = bin_mean[response]

    else:
        bin_count = temp_df.groupby(by=[df[predictor]]).count()
        bin_mean = temp_df.groupby(by=[df[predictor]]).mean()

        mean_of_response_table["BinCount"] = bin_count[response]
        mean_of_response_table["BinMean"] = bin_mean[response]

    mean_of_response_table["Population_Mean"] = population_mean

    mean_of_response_table["Mean_diff"] = round(
        mean_of_response_table["BinMean"] - mean_of_response_table["Population_Mean"],
        6,
    )

    mean_of_response_table["mean_squared_diff"] = round(
        (mean_of_response_table["Mean_diff"]) ** 2, 6
    )

    mean_of_response_table["Weight"] = (
        mean_of_response_table["BinCount"] / df[response].count()
    )
    mean_of_response_table["mean_squared_diff_weighted"] = (
        mean_of_response_table["mean_squared_diff"] * mean_of_response_table["Weight"]
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

    # initiating axis for plots
    if check_datatype(df[predictor]) == 1:
        x_axis = mean_of_response_table["BinCentre"]
    else:
        x_axis = mean_of_response_table[df[predictor].name]

    # adding plots

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
        go.Scatter(x=x_axis, y=mean_of_response_table["BinMean"], name="Bin Mean"),
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

    fig.add_hline(mean_of_response_table["Population_Mean"][0], line_color="green")

    title = f"Bin Difference with Mean of Response vs Bin ({predictor})"
    # Add figure title
    fig.update_layout(
        title_text=f"<b>Bin Difference with Mean of Response of ({predictor})<b>"
    )

    # Set x-axis title
    fig.update_xaxes(title_text=f"<b>Predictor({predictor}) Bin<b>")

    # Set y-axes titles
    fig.update_yaxes(title_text="<b>Population</b>", secondary_y=True)
    fig.update_yaxes(title_text=f"<b>Response({response})</b>", secondary_y=False)

    url = plot_link(fig, title)

    return mean_squared_diff, mean_squared_diff_weighted, url


def feature_imp_ranking(df, predictors, response):

    """Returns the table of feature importance ranking of all predictors"""

    random_forest_classifier = RandomForestClassifier(random_state=42)
    random_forest_classifier.fit(df[predictors], df[response])
    features_importance = random_forest_classifier.feature_importances_

    table_list = []

    for i in range(0, len(predictors)):

        predictor = predictors[i]

        if check_datatype(df[predictor]) == 0:
            t_value = "N/A"
            p_value = "N/A"
            url3 = Plot.heat_map(df, predictor, response)

        elif check_datatype(df[predictor]) == 1:

            pred = statsmodels.api.add_constant(df[predictor])
            linear_regression_model = statsmodels.api.OLS(df[response], pred)
            linear_regression_model_fitted = linear_regression_model.fit()
            t_value = round(linear_regression_model_fitted.tvalues[1], 6)
            p_value = "{:.6e}".format(linear_regression_model_fitted.pvalues[1])
            url3 = Plot.scatter(df, predictor, response)

        url1, url2 = Plot.dist_violin(df, predictor, response)

        rfvimp = features_importance[i]

        mean_squared_diff, mean_squared_diff_weighted, url4 = msd(
            df, predictor, response
        )

        new_list = {
            "Predictor": predictor,
            "Dist Plot": url1,
            "Violin Plot": url2,
            "RF_VIMP": rfvimp,
            "p value": p_value,
            "t score": t_value,
            "Scatter Plot": url3,
            "Difference with Mean of Response": mean_squared_diff,
            "Plot": url4,
            "Difference with Mean of Response(Weighted)": mean_squared_diff_weighted,
        }

        table_list.append(new_list)

    table = pandas.DataFrame(table_list)

    table = table.sort_values(
        by="Difference with Mean of Response(Weighted)", ascending=False
    ).reset_index(drop=True)

    table = table.style.format(
        {
            "Dist Plot": make_clickable,
            "Violin Plot": make_clickable,
            "Scatter Plot": make_clickable,
            "Plot": make_clickable,
        }
    )

    table = table_style(table)

    return table
