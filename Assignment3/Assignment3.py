# Importing necessary libraries

import sys

from pyspark import StorageLevel, keyword_only
from pyspark.ml import Transformer
from pyspark.sql import SparkSession

# Creating a spark session
spark = (
    SparkSession.builder.config(
        "spark.jars",
        "/Users/nsama6043/Downloads/mysql-connector-java-5.1.46/mysql-connector-java-5.1.46.jar",
    )
    .master("local")
    .appName("PySpark_MySql_test")
    .getOrCreate()
)


# creating a function to load the data from data base
def load_data(query):
    table_data = (
        spark.read.format("jdbc")
        .option("url", "jdbc:mysql://localhost:3306/baseball")
        .option("driver", "com.mysql.jdbc.Driver")
        .option("query", query)
        .option("user", "root")
        .option("password", "root")  # pragma: allowlist secret
        .load()
    )  # pragma: allowlist secret
    return table_data


# Creating a function to join the game and batter_counts table


def batterdata():

    game_query = """SELECT game_id, date(local_date) as local_date From game"""

    battercounts_query = """SELECT game_id, batter,atbat,hit FROM batter_counts"""

    game = load_data(game_query)  # getting the game table data
    batter_counts = load_data(battercounts_query)  # Loading the batter_counts data

    batter_stats = batter_counts.join(game, on="game_id")  # joining the tables

    return batter_stats


# defining a function to calculate rolling average


def rollingavg(spark, data):

    rollingavg_query = """SELECT t1.batter, t1.local_date, (CASE WHEN SUM(t2.atBat)=0 THEN 0
                                  ELSE (SUM(t2.hit)/(SUM(t2.atbat)))END) AS Rolling_battingaverage
                                  FROM batter_stats_temp AS t1 JOIN batter_stats_temp AS t2
                                  ON t1.batter= t2.batter
                                  and t1.local_date > t2.local_date and t2.local_date
                                  between  t1.local_date - INTERVAL 100 DAY and t1.local_date
                                  group by t1.batter, t1.local_date
                                  order by t1.batter, t1.local_date """
    data.createOrReplaceTempView("batter_stats_temp")
    data.persist(StorageLevel.DISK_ONLY)
    rollingaverage = spark.sql(rollingavg_query)

    return rollingaverage


# Constructing a transformer


class rollingaveragetransformer(Transformer):
    @keyword_only
    def __init__(self):
        super(rollingaveragetransformer, self).__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)
        return

    @keyword_only
    def setParams(self):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def _transform(data):
        rolling_avg = rollingavg(spark, data)
        return rolling_avg


def main():

    # Loading the batter data

    batter_stats = batterdata()

    # Using the transformer

    transformer = rollingaveragetransformer

    # Passing the data to transformer

    Rolling_Average = transformer._transform(batter_stats)

    # Displaying the first 10 rows of the result

    Rolling_Average.show(10)


if __name__ == "__main__":
    sys.exit(main())
