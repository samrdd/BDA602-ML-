-- Show all the available databases
SHOW DATABASES;

-- Using the baseball database
USE baseball;

-- Show all the tables in the baseball database
SHOW TABLES;

-- Show game table

SELECT *
     FROM game
     LIMIT 0, 20;

-- Show batter_counts table

SELECT *
     FROM batter_counts
     LIMIT 0,20;

-- Adding index to game table
ALTER TABLE game ADD INDEX `Gameid_date_ind`(`game_id`);

-- Adding index to batter_counts table
ALTER TABLE batter_counts ADD INDEX `Gameid_date_ind`(`game_id`);

-- Creating a table to store batter stats
DROP TABLE IF EXISTS s_batter_stats
CREATE TABLE s_batter_stats
    (INDEX Gameid_ind(game_id), INDEX Localdate(local_date), INDEX year_ind(Year))
             SELECT baseball.game.game_id as game_id,
                    DATE(baseball.game.local_date) as local_date,
                    EXTRACT(YEAR FROM baseball.game.local_date) AS Year ,
                    batter,
                    atBat,
                    Hit
                    FROM batter_counts
             JOIN baseball.game
             ON baseball.game.game_id=batter_counts.game_id;

-- Viewing the generated table

SELECT *
     FROM s_batter_stats
     LIMIT 0,20;

-- Calculating the historic average of single player

SELECT batter as Player ,
       ROUND(SUM(Hit)/SUM(atbat), 3)
           AS Historic_BattingAverage from s_batter_stats WHERE batter=110029  ;

-- Calculating historic average of each player and storing in a table

DROP TABLE IF EXISTS f_historic_battingavg
CREATE TABLE f_historic_battingavg AS
              SELECT batter as Player ,
                      (SELECT CASE
                              WHEN SUM(atBat)=0
                              THEN NULL
                              ELSE (ROUND(SUM(Hit)/(SUM(atBat)), 3))
                              END)
                    AS Historic_BattingAverage
              FROM s_batter_stats
              GROUP BY batter
              ORDER BY batter;

SELECT * FROM f_historic_battingavg LIMIT 0,20;

-- Calculating Annual batting average of players and storing in a table

DROP TABLE IF EXISTS f_annual_battingavg
CREATE TABLE f_annual_battingavg AS
             SELECT batter AS Player,
                    YEAR,
                    (SELECT CASE
                            WHEN SUM(atBat)=0
                            THEN NULL
                            ELSE (ROUND(SUM(Hit)/(SUM(atBat)), 3))
                            END)
                    AS Annual_battingaverage
             FROM s_batterstats_date
             GROUP BY batter, Year
             ORDER BY batter, Year;

SELECT * FROM f_annual_battingavg LIMIT 0,20;

-- calculating the rolling average of single player

SELECT batter,
       local_date,
       atBat,
       Hit,
       (SELECT CASE
               WHEN SUM(atBat)=0
               THEN NULL
               ELSE (ROUND(SUM(Hit)/(SUM(atBat)), 3))
               END)
            AS Rolling_battingaverage
       FROM s_batter_stats WHERE local_date > DATE_ADD(local_date, INTERVAL -100 DAY) AND batter=110029
        GROUP BY local_date;

-- Calculating the rolling batting average of all players and storing in a table

DROP TABLE IF EXISTS f_rolling_battingavg
CREATE TABLE f_rolling_battingavg AS
SELECT batter,
       local_date,
       (SELECT CASE
               WHEN SUM(atBat)=0
               THEN NULL
               ELSE (ROUND(SUM(Hit)/(SUM(atBat)), 3))
               END)
            AS Rolling_battingaverage
       FROM s_batter_stats WHERE local_date > DATE_ADD(local_date, INTERVAL -100 DAY)
        GROUP BY batter, local_date;

SELECT * FROM f_rolling_battingavg LIMIT 0,20;