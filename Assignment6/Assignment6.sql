-- Using the baseball database
USE baseball;

-- Creating a table to store batter stats
CREATE TABLE IF NOT EXISTS batter_stats
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


CREATE TABLE IF NOT EXISTS batter_table(INDEX Batter_ind(batter), INDEX Localdate_ind(local_date))
       SELECT game_id,
             batter,
              local_date,
              SUM(atBat) AS atbat,
              SUM(Hit) AS hit FROM batter_stats
       GROUP BY batter, local_date
       ORDER BY batter;

-- Calculating the rolling batting average of all players and storing in a table

CREATE TABLE IF NOT EXISTS f_rolling_battingavg AS
SELECT t1.batter,
       t1.local_date,
       (CASE WHEN SUM(t2.atBat)=0
        THEN 0
        ELSE (SUM(t2.hit)/(SUM(t2.atbat)))
        END) AS Rolling_battingaverage
    FROM batter_table AS t1 JOIN batter_table AS t2
    ON t1.batter= t2.batter
    WHERE t2.local_date >= DATE_ADD(t1.local_date, INTERVAL -100 DAY)
    AND t2.local_date< t1.local_date AND t1.game_id = 12560
 GROUP BY t1.batter, t1.local_date;

SELECT * FROM f_rolling_battingavg;