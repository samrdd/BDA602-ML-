from BruteForce import BruteForce, split_predictors
from CorrelationMatrix import CorrelationMatrix, CorrelationTables
from feature_imp import feature_imp_ranking


class Stats:
    @staticmethod
    def get_all_tables(df, predictors, response):

        """Getting the list of cat and cont predictors"""
        cat_predictors, cont_predictors = split_predictors(predictors, df)

        """creating a html file for output"""
        file_html = open("./results/Baseball_Analytics.html", "w")

        file_html.write("<h1><center>Baseball Analytics</center></h1>")

        file_html.write(
            "<h2><center>Feature Importance Ranking of all Predictors</center></h2>"
        )

        feature_imp_ranking_table = feature_imp_ranking(
            df, predictors, response
        ).to_html()

        file_html.write("<body><center>%s</center></body>" % feature_imp_ranking_table)

        """Checking if there are continuous predictors in the list of predictors"""
        if len(cont_predictors) <= 1:

            pass

        elif len(cont_predictors) > 1:

            file_html.write(
                "<h1><center>Continuous Vs Continuous Predictor Pairs</center></h1>"
            )

            file_html.write("<h2><center>Correlation Table</center></h2>")

            cont_cont_corr_table = CorrelationTables.cont_cont(
                df, cont_predictors
            ).to_html()

            file_html.write("<body><center>%s</center></body>" % cont_cont_corr_table)

            file_html.write("<h2><center>Confusion Matrix</center></h2>")

            corr_cont_cont = CorrelationMatrix.cont_cont(df, cont_predictors).to_html()

            file_html.write("<body><center>%s</center></body>" % corr_cont_cont)

            file_html.write("<h2><center>Brute Force Table</center></h2>")

            cont_cont_bruteforce_table = BruteForce.cont_cont(
                df, predictors, response
            ).to_html()

            file_html.write(
                "<body><center>%s</center></body>" % cont_cont_bruteforce_table
            )

        """Checking if there are Categorical predictors in the list of predictors"""

        if len(cat_predictors) <= 1:

            pass

        elif len(cat_predictors) > 1:

            file_html.write(
                "<h1><center>Categorical Vs Categorical Predictor Pairs </center></h1>"
            )

            file_html.write("<h2><center>Correlation Table</center></h2>")

            cat_cat_corr_table = CorrelationTables.cat_cat(df, cat_predictors).to_html()

            file_html.write("<body><center>%s</center></body>" % cat_cat_corr_table)

            file_html.write("<h2><center>Confusion Matrix</center></h2>")

            corr_cat_cat = CorrelationMatrix.cat_cat(df, cat_predictors).to_html()

            file_html.write("<body><center>%s</center></body>" % corr_cat_cat)

            file_html.write("<h2><center>Brute Force Table</center></h2>")

            cat_cat_bruteforce_table = BruteForce.cat_cat(
                df, predictors, response
            ).to_html()

            file_html.write(
                "<body><center>%s</center></body>" % cat_cat_bruteforce_table
            )

        """Checking if there are Categorical and Continuous predictor pairs"""

        if len(cat_predictors) == 0 or len(cont_predictors) == 0:

            pass

        else:

            file_html.write(
                "<h1><center>Categorical Vs Continuous Predictor Pairs</center></h1>"
            )

            file_html.write("<h2><center>Correlation Table</center></h2>")

            cat_cont_corr_table = CorrelationTables.cat_cont(df, predictors).to_html()

            file_html.write("<body><center>%s</center></body>" % cat_cont_corr_table)

            file_html.write("<h2><center>Confusion Matrix<center></h2>")

            corr_cat_cont = CorrelationMatrix.cat_cont(df, predictors).to_html()

            file_html.write("<body><center>%s</center></body>" % corr_cat_cont)

            file_html.write("<h2><center>Brute Force Table</center></h2>")

            cat_cont_bruteforce_table = BruteForce.cat_cont(
                df, predictors, response
            ).to_html()

            file_html.write(
                "<body><center>%s</center></body>" % cat_cont_bruteforce_table
            )

        file_html.close()

        return
