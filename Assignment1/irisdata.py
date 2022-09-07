import sys

import numpy as np
import pandas as pd
import plotly.express as px


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
    print(
        "Sepal Length Summary [",
        "Mean :",
        np.mean(x[:, 0]),
        "Median :",
        np.median(x[:, 0], axis=None),
        "Minimum:",
        np.min(x[:, 0], axis=None),
        "Maximum :",
        np.max(x[:, 0], axis=None),
        "1st quartile:",
        np.quantile(x[:, 0], 0.25, axis=None),
        "3rd Quartile",
        np.quantile(x[:, 0], 0.75, axis=None),
    )
    print(
        "Sepal Width Summary [",
        "Mean :",
        np.mean(x[:, 1]),
        "Median :",
        np.median(x[:, 1], axis=None),
        "Minimum:",
        np.min(x[:, 1], axis=None),
        "Maximum :",
        np.max(x[:, 1], axis=None),
        "1st quartile:",
        np.quantile(x[:, 1], 0.25, axis=None),
        "3rd Quartile",
        np.quantile(x[:, 1], 0.75, axis=None),
    )
    print(
        "Petal Length Summary [",
        "Mean :",
        np.mean(x[:, 2]),
        "Median :",
        np.median(x[:, 2], axis=None),
        "Minimum:",
        np.min(x[:, 2], axis=None),
        "Maximum :",
        np.max(x[:, 2], axis=None),
        "1st quartile:",
        np.quantile(x[:, 2], 0.25, axis=None),
        "3rd Quartile",
        np.quantile(x[:, 2], 0.75, axis=None),
    )
    print(
        "Petal Width Summary [",
        "Mean :",
        np.mean(x[:, 3]),
        "Median :",
        np.median(x[:, 3], axis=None),
        "Minimum:",
        np.min(x[:, 3], axis=None),
        "Maximum :",
        np.max(x[:, 3], axis=None),
        "1st quartile:",
        np.quantile(x[:, 3], 0.25, axis=None),
        "3rd Quartile",
        np.quantile(x[:, 3], 0.75, axis=None),
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
    fig.show()
    # scatter plot of petal length and petal width
    fig = px.scatter(
        data_df,
        x="petal.width",
        y="petal.length",
        color="species",
        symbol="species",
        title="Scatter Plot of Petal Length & Petal Width",
    )
    fig.show()
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
    fig1.show()
    fig2.show()
    fig3.show()
    fig4.show()
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
    fig5.show()
    fig6.show()
    fig7.show()
    fig8.show()
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
    fig9.show()
    fig10.show()
    fig11.show()
    fig12.show()
    # Transforming the data using standard scaler
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    # Encoding the output variable
    species = data_df.iloc[:, 4].values  # converting output variable to numpy array
    from sklearn.preprocessing import LabelEncoder

    encoder = LabelEncoder()
    species = encoder.fit_transform(species)  # Encoding the out put values
    # Splitting the data to test and training sets
    from sklearn.model_selection import train_test_split

    x_train, x_test, y_train, y_test = train_test_split(
        x, species, test_size=0.3, random_state=42
    )
    # Fitting Random forest against the data
    from sklearn.ensemble import RandomForestClassifier  # importing the classifier

    classifier = RandomForestClassifier()
    classifier.fit(x_train, y_train)  # training the model
    y_pred = classifier.predict(x_test)  # predicting the output using test data
    print("Predicted output values")
    print(y_pred)
    from sklearn.metrics import accuracy_score, confusion_matrix

    print("Random forest model Results:")
    print("Confusion Matrix :")
    print(confusion_matrix(y_pred, y_test))
    print("Accuracy Score :")
    print(accuracy_score(y_pred, y_test) * 100)  # calculating the accuracy of the model
    # Fitting different models using a piple line
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier

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
