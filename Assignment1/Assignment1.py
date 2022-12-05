import sys

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def main():
    # Loading the dataset
    data_df = pd.read_csv("Iris.csv")
    print(data_df.head())
    print(data_df.tail())
    # Getting simple summary stats using pandas
    print(data_df.describe())
    # checking the columns of data
    print(data_df.columns)
    # checking the count of species
    print(data_df["species"].value_counts())
    # converting the data to numpy array and getting summary
    x = data_df.iloc[:, 0:4].values
    # summary of the data
    columns = ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"]
    for i in range(0, 4, 1):
        print(
            columns[i],
            " Summary [",
            "Mean :",
            np.mean(x[:, i]),
            "Median :",
            np.median(x[:, i], axis=None),
            "Minimum:",
            np.min(x[:, i], axis=None),
            "Maximum :",
            np.max(x[:, i], axis=None),
            "1st quartile:",
            np.quantile(x[:, i], 0.25, axis=None),
            "3rd Quartile",
            np.quantile(x[:, i], 0.75, axis=None),
        )
    # scatter plots
    # scatter plot of sepal length and sepal width
    fig = px.scatter(
        data_df,
        x="sepal.width",
        y="sepal.length",
        color="species",
        symbol="species",
        title="Scatter plot of Seoal Length & Sepal Width",
    )
    fig.write_html(file="fig.html", include_plotlyjs="cdn")
    # scatter plot of petal length and petal width
    fig0 = px.scatter(
        data_df,
        x="petal.width",
        y="petal.length",
        color="species",
        symbol="species",
        title="Scatter Plot of Petal Length & Petal Width",
    )
    fig0.write_html(file="fig0.html", include_plotlyjs="cdn")
    # Violin and Box Plots
    fig1 = px.violin(
        data_df, y="sepal.length", box=True, title="Violin & Box Plot of Sepal Length"
    )
    fig2 = px.violin(
        data_df, y="sepal.width", box=True, title="Violin & Box Plot of Sepal Width"
    )
    fig3 = px.violin(
        data_df, y="petal.length", box=True, title="Violin & Box Plot Petal Length"
    )
    fig4 = px.violin(
        data_df, y="petal.width", box=True, title="Violin & Box Plot Petal Width"
    )
    fig1.write_html(file="fig1.html", include_plotlyjs="cdn")
    fig2.write_html(file="fig2.html", include_plotlyjs="cdn")
    fig3.write_html(file="fig3.html", include_plotlyjs="cdn")
    fig4.write_html(file="fig4.html", include_plotlyjs="cdn")
    # Bar Plots
    fig5 = px.bar(
        data_df,
        x="species",
        y="sepal.length",
        color="species",
        title="Bar Plot of Sepal Length",
    )
    fig6 = px.bar(
        data_df,
        x="species",
        y="sepal.width",
        color="species",
        title="Bar Plot of Sepal width",
    )
    fig7 = px.bar(
        data_df,
        x="species",
        y="petal.length",
        color="species",
        title="Bar Plot of Petal Length",
    )
    fig8 = px.bar(
        data_df,
        x="species",
        y="petal.width",
        color="species",
        title="Bar Plot of Petal Width",
    )
    fig5.write_html(file="fig5.html", include_plotlyjs="cdn")
    fig6.write_html(file="fig6.html", include_plotlyjs="cdn")
    fig7.write_html(file="fig7.html", include_plotlyjs="cdn")
    fig8.write_html(file="fig8.html", include_plotlyjs="cdn")
    # Histogram
    fig9 = px.histogram(
        data_df, x="sepal.length", color="species", title="Histogram of Sepal length"
    )
    fig10 = px.histogram(
        data_df, x="sepal.width", color="species", title="Histogram of Sepal width"
    )
    fig11 = px.histogram(
        data_df, x="petal.length", color="species", title="Histogram of Petal length"
    )
    fig12 = px.histogram(
        data_df, x="petal.width", color="species", title="Histogram of Petal width"
    )
    fig9.write_html(file="fig9.html", include_plotlyjs="cdn")
    fig10.write_html(file="fig10.html", include_plotlyjs="cdn")
    fig11.write_html(file="fig11.html", include_plotlyjs="cdn")
    fig12.write_html(file="fig12.html", include_plotlyjs="cdn")
    # Transforming the data using standard scaler
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    # Encoding the output variable
    species = data_df.iloc[:, 4].values  # converting output variable to numpy array
    encoder = LabelEncoder()
    species = encoder.fit_transform(species)  # Encoding the out put values
    # Splitting the data to test and training sets
    x_train, x_test, y_train, y_test = train_test_split(
        x, species, test_size=0.3, random_state=42
    )
    # Fitting Random forest against the data
    classifier = RandomForestClassifier()
    classifier.fit(x_train, y_train)  # training the model
    y_pred = classifier.predict(x_test)  # predicting the output using test data
    print("Predicted output values")
    print(y_pred)
    print("Random forest model Results:")
    print("Confusion Matrix :")
    print(confusion_matrix(y_pred, y_test))
    print("Accuracy Score :")
    print(accuracy_score(y_pred, y_test) * 100)  # calculating the accuracy of the model
    # Fitting different models using a piple line
    randomforestpipeline = Pipeline(
        [("scaler", StandardScaler()), ("Classifier", RandomForestClassifier())]
    )
    decisiontreepipeline = Pipeline(
        [("scaler", StandardScaler()), ("Classifier", DecisionTreeClassifier())]
    )
    logisticregressionpipeline = Pipeline(
        [("scaler", StandardScaler()), ("Classifier", LogisticRegression())]
    )
    svmpipeline = Pipeline([("scaler", StandardScaler()), ("Classifier", SVC())])
    # Defining pipeline in a list
    mypipeline = [
        randomforestpipeline,
        decisiontreepipeline,
        logisticregressionpipeline,
        svmpipeline,
    ]
    # pipe line dictionary
    pipeline_dict = {
        0: "Random Forest",
        1: "Decision Tree",
        2: "Logistic Regression",
        3: "SVM",
    }
    # Fitting the pipeline to data
    for mypipe in mypipeline:
        mypipe.fit(x_train, y_train)
    print("Accuracy of Different Classifiers :")
    for i, model in enumerate(mypipeline):
        print(
            "{} Test Accuracy :{}%".format(
                pipeline_dict[i], model.score(x_test, y_test) * 100
            )
        )
    return


if __name__ == "__main__":
    sys.exit(main())
