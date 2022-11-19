from CorrelationTables import *


class CorrelationMatrix:

    @staticmethod
    def cont_cont(dataframe, list):

        """Returns the url of correlation heat map of Continuous predictor pairs"""

        table = pandas.DataFrame(columns=["HeatMap of Cont_Cont Correlation Matrix"])

        data = pandas.concat(
            [pandas.DataFrame(dataframe[[list[i]]]) for i in range(len(list))], axis=1
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
    def cat_cat(dataframe, list):

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
    def cat_cont(dataframe, list):

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
                correlationratio = correlationratio(
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