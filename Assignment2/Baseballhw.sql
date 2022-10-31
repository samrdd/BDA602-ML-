-- Show all the available databases
SHOW DATABASES;

-- Using the baseball database
USE baseball;

-- Show all the tables in the baseball database
SHOW TABLES;


-- Adding index to game table
ALTER TABLE game ADD INDEX `Gameid_Ind`(`game_id`);

-- Adding index to batter_counts table
ALTER TABLE batter_counts ADD INDEX `Gameid_IND`(`game_id`);

-- Creating a table to store batter stats
DROP TABLE IF EXISTS s_batter_stats;
CREATE TABLE s_batter_stats
    (INDEX Gameid_ind(game_id), INDEX Localdate(local_date), INDEX year_ind(Year))
             SELECT game.game_id as game_id,
                    DATE(game.local_date) as local_date,
                    YEAR(game.local_date) AS Year ,
                    batter,
                    atBat,
                    Hit
                    FROM batter_counts
             JOIN baseball.game
             ON game.game_id=batter_counts.game_id;


-- Calculating the historic average of single player

SELECT batter as Player ,
       ROUND(SUM(Hit)/SUM(atbat), 3)
           AS Historic_BattingAverage from s_batter_stats WHERE batter=110029  ;

-- Calculating historic average of each player and storing in a table

DROP TABLE IF EXISTS f_historic_battingavg;
CREATE TABLE f_historic_battingavg AS
              SELECT batter as Player ,
                      (CASE WHEN SUM(atBat)=0
                       THEN 0
                       ELSE (ROUND(SUM(Hit)/(SUM(atBat)), 3))
                       END) AS Historic_BattingAverage
              FROM s_batter_stats
              GROUP BY batter
              ORDER BY batter;

SELECT * FROM f_historic_battingavg LIMIT 0,20;

-- Calculating Annual batting average of players and storing in a table

DROP TABLE IF EXISTS f_annual_battingavg;
CREATE TABLE f_annual_battingavg AS
             SELECT batter AS Player,
                    Year,
                    (CASE WHEN SUM(atBat)=0
                     THEN 0
                     ELSE (ROUND(SUM(Hit)/(SUM(atBat)), 3))
                     END) AS Annual_battingaverage
             FROM s_batter_stats
             GROUP BY batter, Year
             ORDER BY batter, Year;

SELECT * FROM f_annual_battingavg LIMIT 0,20;

-- Creating a table to store the batter data for every date he played

DROP TABLE IF EXISTS s_batter_table;
CREATE TABLE s_batter_table(INDEX Batter_ind(batter), INDEX Localdate_ind(local_date))
       SELECT batter,
              local_date,
              SUM(atBat) AS atbat,
              SUM(Hit) AS hit FROM s_batter_stats
       GROUP BY batter, local_date
       ORDER BY batter;

-- Calculating the rolling batting average of all players and storing in a table

DROP TABLE IF EXISTS f_rolling_battingavg;
CREATE TABLE f_rolling_battingavg AS
SELECT t1.batter,
       t1.local_date,
       (CASE WHEN SUM(t2.atBat)=0 
        THEN 0
        ELSE (ROUND(SUM(t2.hit)/(SUM(t2.atbat)), 3))
        END) AS Rolling_battingaverage
    FROM s_batter_table AS t1 JOIN s_batter_table AS t2
    ON t1.batter= t2.batter
    WHERE t2.local_date >= DATE_ADD(t1.local_date, INTERVAL -100 DAY)
    AND t2.local_date< t1.local_date
 GROUP BY t1.batter, t1.local_date;

SELECT * FROM f_rolling_battingavg LIMIT 0,20;