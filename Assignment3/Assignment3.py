# Importing necessary libraries
import sys
from pyspark import StorageLevel
from pyspark.sql import SparkSession
from pyspark.ml import Transformer

# Constructing a transformer

class Rollingaverage(Transformer):

    pass



def main():
    # Building the connection
    spark = SparkSession.builder.config("spark.jars",
                                        "/Users/nsama6043/Desktop/NewProj/BDA602-ML-/Assignment3/mysql-connector-java-5.1.46/mysql-connector-java-5.1.46.jar") \
        .master("local").appName("PySpark_MySql_test").config("spark.memory.offHeap.enabled", "true").config(
        "spark.memory.offHeap.size", "10g").getOrCreate()
   # storing the game table in a game data frame

    game_query = """
             SELECT                               
              game_id,
              date(local_date) as local_date
              From game
              """

    game= spark.read.format("jdbc")\
        .option("url","jdbc:mysql://localhost:3306/baseball")\
        .option("driver","com.mysql.jdbc.Driver")\
        .option("query" , game_query)\
        .option("user","root")\
        .option("password","root")\
        .load()

    # storing the batter table in batter_counts data frame
    batter_countsquery = """
                         SELECT 
                             game_id,
                             batter,
                             atbat,
                             hit
                         FROM batter_counts
                         """
    batter_counts=spark.read.format("jdbc")\
        .option("url","jdbc:mysql://localhost:3306/baseball")\
        .option("driver","com.mysql.jdbc.Driver")\
        .option("query" , batter_countsquery)\
        .option("user","root")\
        .option("password","root")\
        .load()

    # joining two dataframes on game_id
    batter_stats = batter_counts.join(game, on="game_id")

    batter_stats.createOrReplaceTempView("batter_stats_temp")
    batter_stats.persist(StorageLevel.DISK_ONLY)


    # calculating the rolling average
    rollingaverage = spark.sql( """
                                          SELECT t1.batter,
                                             t1.local_date,
                                             (CASE WHEN SUM(t2.atBat)=0
                                              THEN 0
                                              ELSE (SUM(t2.hit)/(SUM(t2.atbat)))
                                              END) AS Rolling_battingaverage
                                          FROM batter_stats_temp AS t1 JOIN batter_stats_temp AS t2
                                          ON t1.batter= t2.batter
                                          and t1.local_date > t2.local_date and t2.local_date between  t1.local_date - INTERVAL 100 DAY and t1.local_date
                                          group by t1.batter, t1.local_date
                                          """
                                )

    rollingaverage.show(10)



if __name__ == "__main__":
    sys.exit(main())