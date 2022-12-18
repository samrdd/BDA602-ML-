from pathlib import Path

import numpy as np
import pandas
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_auc_score, roc_curve


def plot_link(fig, name):

    """Save the given plot as a link"""
    Path("./results/plots/").mkdir(parents=True, exist_ok=True)
    link = f"./results/plots/{name}.html"
    fig.write_html(link)
    path = f"./plots/{name}.html"

    return path


def heatmap(correlation_matrix, title):

    """Returns the heat map of given confusion matrix"""

    # plotting based on absolute values
    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            x=correlation_matrix.columns,
            y=correlation_matrix.index,
            z=np.array(correlation_matrix),
            text=correlation_matrix.values,
            texttemplate="%{text}",
            colorscale="RdBu",
        )
    )

    url = plot_link(fig, title)

    return url


class Plot:

    """Returns url of the plot of respective pairs"""

    @staticmethod
    def violin(df, predictor, response):

        fig = px.violin(
            df,
            y=df[predictor],
            x=df[response],
            box=True,
            color=df[response],
            hover_data=df.columns,
        )

        title = f"Violin Plot of {predictor} by {response}"
        fig.update_layout(
            title=title,
            xaxis_title=f"{response}",
            yaxis_title=f"{predictor}",
        )
        # saving the url of the plot
        url = plot_link(fig, title)

        return url

    @staticmethod
    def scatter(df, variable1, variable2):

        """Returns the scatter plot link of continuous and continuous pairs"""

        fig = px.scatter(df, df[variable1], df[variable2], trendline="ols")
        title = f"{variable1} Vs {variable2}"
        fig.update_layout(
            title=title,
            xaxis_title=f"{variable1}",
            yaxis_title=f"{variable2}",
        )

        # saving the url of the plot
        url = plot_link(fig, title)

        return url

    @staticmethod
    def heat_map(df, variable1, variable2):

        """Returns the link of heat map given predictors or a predictor and response"""

        confusion_matrix = pandas.crosstab(df[variable1], df[variable2])

        fig = px.imshow(confusion_matrix, text_auto=True, aspect="auto")

        title = f"{variable1} Vs {variable1}"
        fig.update_layout(
            title=title,
        )
        url = plot_link(fig, title)

        return url

    @staticmethod
    def dist_violin(df, variable1, variable2):

        """Returns the link of the distribution and violin plot of categorical and continuous pairs"""

        fig1 = px.histogram(
            df,
            x=df[variable1],
            color=df[variable2],
            marginal="rug",
            hover_data=df.columns,
        )
        title = f"Distribution Plot of {variable2} by {variable1}"
        fig1.update_layout(
            title=title,
            xaxis_title=f"{variable2}",
            yaxis_title=f"{variable1}",
        )
        # saving the url of the plot
        url1 = plot_link(fig1, title)

        fig2 = px.violin(
            df,
            y=df[variable1],
            x=df[variable2],
            box=True,
            color=df[variable2],
            hover_data=df.columns,
        )

        title = f"Violin Plot of {variable1} by {variable1}"
        fig2.update_layout(
            title=title,
            xaxis_title=f"{variable1}",
            yaxis_title=f"{variable1}",
        )
        # saving the url of the plot
        url2 = plot_link(fig2, title)

        return url1, url2

    @staticmethod
    def roc_curve(y_test, y_pred):
        fpr, tpr, _ = roc_curve(y_test, y_pred, drop_intermediate=False)

        # Calculate the AUC
        roc_auc = roc_auc_score(y_test, y_pred)

        # Create the figure
        fig = go.Figure(data=go.Scatter(x=fpr, y=tpr, name="Model"))
        fig = fig.add_trace(
            go.Scatter(
                x=[0.0, 1.0],
                y=[0.0, 1.0],
                line=dict(dash="dash"),
                mode="lines",
                showlegend=False,
            )
        )

        # Label the figure

        title = f"Receiver Operator Characteristic (AUC={round(roc_auc, 6)})"
        fig.update_layout(
            title=title,
            xaxis_title="False Positive Rate (FPR)",
            yaxis_title="True Positive Rate (TPR)",
        )

        url = plot_link(fig, title)

        return url
