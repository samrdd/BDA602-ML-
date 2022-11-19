from scipy.stats import chi2_contingency
from BruteForce import *
from sklearn.ensemble import RandomForestClassifier


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


class CorrelationTables:

    @staticmethod
    def cont_cont(df, predictors):

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
            predictor = f"{comb[i][0]} and {comb[i][1]}"

            corr_coeff = df[comb[i][0]].corr(df[comb[i][1]])

            abs_corr = round(abs(corr_coeff), 4)

            url = Plot.cont_cont(df, predictors, i)

            table_new_row = pandas.DataFrame(
                {
                    "Predictors": predictor,
                    "Correlation Coefficient": corr_coeff,
                    "Absolute Correlation Coefficient": abs_corr,
                    "Plot_link": url,
                },
                index=[0]
            )

            table = pandas.concat([table, table_new_row])

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

        table = pandas.DataFrame(
            columns=["Predictors", "Correlation Coefficient", "Plot_link"]
        )

        comb = list(combinations(predictors, 2))

        for i in range(0, len(comb)):
            predictor = f"{comb[i][0]} and {comb[i][1]}"

            corr_coeff = cramers_V(df[comb[i][0]], df[comb[i][1]])

            url = Plot.cat_cat(df, predictors, i)

            table_new_row = pandas.DataFrame(
                {
                    "Predictors": predictor,
                    "Correlation Coefficient": corr_coeff,
                    "Plot_link": url,
                }
            )
            table = pandas.concat([table, table_new_row])

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

            url1, url2 = Plot.cat_cont(df, comb, i)

            table_new_row = pandas.DataFrame(
                {
                    "Categorical Predictor": Predictor1,
                    "Continuous Predictor": Predictor2,
                    "Correlation Coefficient": corr_coeff,
                    "Distribution Plot": url1,
                    "Violin Plot": url2,
                }
            )

            table = pandas.concat([table, table_new_row])

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


def random_forest_ranking(df, predictors, response):

    table = pandas.DataFrame(
        columns=[
            "Predictor",
            "RF_VIMP",
            "Violin Plot",
        ]
    )
    random_forest_classifier = RandomForestClassifier(random_state=42)
    random_forest_classifier.fit(df[predictors], df[response])
    features_importance = random_forest_classifier.feature_importances_

    for i in range(0, len(predictors)):

        predictor = predictors[i]

        url= Plot.violin(df, predictor, response)

        score = features_importance[i]

        table_new_row = pandas.DataFrame(
            {
                "Predictor": predictor,
                "RF_VIMP": score,
                "Violin Plot": url,
            },
            index=[0]
        )

        table = pandas.concat([table, table_new_row])
    table = table.sort_values(
        by="RF_VIMP", ascending=False
    ).reset_index(drop=True)

    table = table.style.format(
        {
            "Violin Plot": make_clickable,
        }
    )

    table = table_style(table)

    return table