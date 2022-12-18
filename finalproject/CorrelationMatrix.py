import itertools

import numpy as np
import pandas
from BruteForce import combinations, split_predictors
from Plots import Plot, heatmap
from scipy.stats import chi2_contingency
from tablestyle import make_clickable, table_style


def cramers_V(var1, var2):

    """Calculates the correlation of two categorical variables"""

    cross_tab = np.array(
        pandas.crosstab(var1, var2, rownames=None, colnames=None)
    )  # Cross table building
    stat = chi2_contingency(cross_tab)[
        0
    ]  # Keeping of the test statistic of the Chi2 test
    obs = np.sum(cross_tab)  # Number of observations
    mini = (
        min(cross_tab.shape) - 1
    )  # Take the minimum value between the columns and the rows of the cross table

    return stat / (obs * mini)


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


class CorrelationMatrix:
    @staticmethod
    def cont_cont(dataframe, predictors):

        """Returns the url of correlation heat map of Continuous predictor pairs"""

        table = pandas.DataFrame(columns=["HeatMap of Cont_Cont Correlation Matrix"])

        data = pandas.concat(
            [
                pandas.DataFrame(dataframe[[predictors[i]]])
                for i in range(len(predictors))
            ],
            axis=1,
        )

        correlation_matrix = round(abs(data.corr()), 4)

        title = " Heat Map of Continuous Predictor Pairs "

        url = heatmap(correlation_matrix, title)

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

    @staticmethod
    def cat_cat(dataframe, predictors):

        """Returns the url of correlation heat map of categorical predictor pairs"""

        table = pandas.DataFrame(columns=["HeatMap of Cat_Cat Correlation Matrix"])

        data = pandas.concat(
            [
                pandas.DataFrame(dataframe[[predictors[i]]])
                for i in range(len(predictors))
            ],
            axis=1,
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
        correlation_matrix = pandas.DataFrame(
            cramers_results, columns=data.columns, index=data.columns
        )

        title = " Heat Map of Categorical Predictor Pairs "

        url = heatmap(correlation_matrix, title)

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

    @staticmethod
    def cat_cont(dataframe, predictors):

        """Returns the url of correlation heat map of categorical and Continuous predictor pairs"""

        table = pandas.DataFrame(columns=["HeatMap of Cat_Cont Correlation Matrix"])

        cat_predictors, cont_predictors = split_predictors(predictors, dataframe)

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

        correlation_matrix = pandas.DataFrame(
            cramers_results, columns=cont_data.columns, index=cat_data.columns
        )

        title = " Heat Map of Categorical and Continuous Predictor pairs "

        url = heatmap(correlation_matrix, title)

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


class CorrelationTables:
    @staticmethod
    def cont_cont(df, predictors):

        """Returns the correlation table of continuous and continuous pairs along with their plots"""

        table_list = []

        comb = list(combinations(predictors, 2))

        for i in range(0, len(comb)):
            predictor = f"{comb[i][0]} and {comb[i][1]}"

            corr_coeff = df[comb[i][0]].corr(df[comb[i][1]])

            abs_corr = round(abs(corr_coeff), 4)

            url = Plot.scatter(df, comb[i][0], comb[i][1])

            new_list = {
                "Predictors": predictor,
                "Correlation Coefficient": corr_coeff,
                "Absolute Correlation Coefficient": abs_corr,
                "Plot_link": url,
            }

            table_list.append(new_list)

        table = pandas.DataFrame(table_list)

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

    @staticmethod
    def cat_cat(df, predictors):

        """Returns the correlation table of categorical and categorical pairs along with their plots"""

        table_list = []

        comb = list(combinations(predictors, 2))

        for i in range(0, len(comb)):
            predictor = f"{comb[i][0]} and {comb[i][1]}"

            corr_coeff = cramers_V(df[comb[i][0]], df[comb[i][1]])

            url = Plot.cat_cat(df, predictors, i)

            new_list = {
                "Predictors": predictor,
                "Correlation Coefficient": corr_coeff,
                "Plot_link": url,
            }

            table_list.append(new_list)

        table = pandas.DataFrame(table_list)

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

    @staticmethod
    def cat_cont(df, predictors):

        """Returns the correlation table of categorical and continuous pairs along with their plots"""

        cat_predictors, cont_predictors = split_predictors(predictors, df)

        Cat_cont_list = [cat_predictors, cont_predictors]

        comb = [p for p in itertools.product(*Cat_cont_list)]

        table_list = []

        for i in range(0, len(comb)):
            Predictor1 = comb[i][0]

            Predictor2 = comb[i][1]

            corr_coeff = correlation_ratio(
                np.array(df[comb[i][0]]), np.array(df[comb[i][1]])
            )

            url1, url2 = Plot.dist_violin(df, Predictor1, Predictor2)

            new_list = {
                "Categorical Predictor": Predictor1,
                "Continuous Predictor": Predictor2,
                "Correlation Coefficient": corr_coeff,
                "Distribution Plot": url1,
                "Violin Plot": url2,
            }

            table_list.append(new_list)

        table = pandas.DataFrame(table_list)

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
