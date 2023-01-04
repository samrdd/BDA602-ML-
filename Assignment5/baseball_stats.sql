-- Using the baseball database
USE baseball;

-- appending the local date to the team_batting_stats and team_pitching_stats and storing in a table

DROP TABLE IF EXISTS s_team_batting_stats;
CREATE TABLE s_team_batting_stats
    (INDEX Game_id_Ind(game_id),INDEX Team_id_ind(team_id), INDEX Local_date(local_date))
        SELECT team_batting_counts.game_id as game_id,
               team_batting_counts.team_id as team_id,
               DATE(game.local_date) as local_date,
               team_batting_counts.atBat as b_AtBat,
               team_batting_counts.Hit as b_Hit,
               team_batting_counts.finalScore as Runs,
               team_batting_counts.Home_Run as b_Home_Run,
               team_batting_counts.Sac_Fly as b_Sac_Fly,
               team_batting_counts.Strikeout as b_StrikeOut,
               team_batting_counts.Walk as b_Walk,
               team_batting_counts.Hit_By_Pitch as b_Hit_by_Pitch,
               team_batting_counts.plateApperance as b_plateApperance
               FROM team_batting_counts
        JOIN game
        ON game.game_id = team_batting_counts.game_id;


DROP TABLE IF EXISTS s_team_pitching_stats;
CREATE TABLE s_team_pitching_stats
    (INDEX Game_id_ind(game_id),INDEX Team_id_ind(team_id), INDEX Local_Date(local_date))
    SELECT team_pitching_counts.game_id as game_id,
               team_pitching_counts.team_id as team_id,
               DATE(game.local_date) as local_date,
               team_pitching_counts.atBat as p_AtBat,
               team_pitching_counts.Hit as p_Hit,
               team_pitching_counts.finalScore as Runs,
               team_pitching_counts.Home_Run as p_Home_Run,
               team_pitching_counts.Sac_Fly as p_Sac_Fly,
               team_pitching_counts.Strikeout as p_StrikeOut,
               team_pitching_counts.Walk as p_Walk,
               team_pitching_counts.plateApperance as p_plateApperance
               FROM team_pitching_counts
        JOIN game
        ON game.game_id = team_pitching_counts.game_id;

-- calculating the rolling batting average of each team of 200 days

DROP TABLE IF EXISTS s_team_bat_rolling_avg;
CREATE TABLE s_team_bat_rolling_avg
    (INDEX Game_id_IND(game_id),INDEX Team_id_IND(team_id), INDEX t_rolling_batting_average(t_rolling_battingaverage))
        SELECT t1.team_id,
               t1.game_id,
               t1.local_date,
               IF (SUM(t2.b_AtBat)=0,0,SUM(t2.b_Hit)/(SUM(t2.b_AtBat))) AS t_rolling_battingaverage
            FROM s_team_batting_stats AS t1 JOIN s_team_batting_stats AS t2
            ON t1.team_id= t2.team_id
            WHERE t2.local_date >= DATE_ADD(t1.local_date, INTERVAL -100 DAY)
            AND t2.local_date< t1.local_date
         GROUP BY t1.team_id, t1.local_date,t1.game_id
         ORDER BY t1.team_id, t1.local_date,t1.game_id;


-- calculating the rolling pitching average of each team of 200 days

DROP TABLE IF EXISTS s_team_pitch_rolling_avg;
CREATE TABLE s_team_pitch_rolling_avg
    (INDEX Game_iD_IND(game_id),INDEX Team_iD_IND(team_id), INDEX t_rolling_pitching_average(t_rolling_pitchingaverage))
    SELECT t1.team_id,
           t1.game_id,
           t1.local_date,
           IF (SUM(t2.p_AtBat)=0,0,SUM(t2.p_Hit)/(SUM(t2.p_AtBat))) AS t_rolling_pitchingaverage
        FROM s_team_pitching_stats AS t1 JOIN s_team_pitching_stats AS t2
        ON t1.team_id= t2.team_id
        WHERE t2.local_date >= DATE_ADD(t1.local_date, INTERVAL -200 DAY)
        AND t2.local_date< t1.local_date
     GROUP BY t1.team_id, t1.local_date,t1.game_id;

-- Team rolling home runs scored

DROP TABLE IF EXISTS s_team_home_runs_per_hits;
CREATE TABLE s_team_home_runs_per_hits
     (INDEX Game_ID_IND(game_id),INDEX Team_ID_IND(team_id), INDEX home_runs_per_hits_IND(home_runs_per_hits))
    SELECT t1.team_id,
           t1.game_id,
           t1.local_date,
           IF (SUM(t2.b_Hit)=0,0,SUM(t2.b_Home_Run)/(SUM(t2.b_Hit))) AS home_runs_per_hits
        FROM s_team_batting_stats AS t1 JOIN s_team_batting_stats AS t2
        ON t1.team_id= t2.team_id
        WHERE t2.local_date >= DATE_ADD(t1.local_date, INTERVAL -200 DAY)
        AND t2.local_date< t1.local_date
     GROUP BY t1.team_id, t1.local_date,t1.game_id;

-- calculating BABIP  for teams

DROP TABLE IF EXISTS s_team_rolling_BABIP;
CREATE TABLE s_team_rolling_BABIP
    (INDEX GAME_ID_IND(game_id),INDEX TEAM_ID_IND(team_id), INDEX b_team_BABIP(b_team_BABIP))
    SELECT t1.team_id,
           t1.game_id,
           t1.local_date,
           IF((sum(t2.b_AtBat) - SUM(t2.b_StrikeOut) - SUM(t2.b_Home_Run) + SUM(t2.b_Sac_Fly)) = 0, 0,
                    (SUM(t2.b_Hit) - SUM(t2.b_Home_Run))/(sum(t2.b_AtBat) - SUM(t2.b_StrikeOut) - SUM(t2.b_Home_Run) + SUM(t2.b_Sac_Fly))) as b_team_BABIP
        FROM s_team_batting_stats AS t1 JOIN s_team_batting_stats AS t2
        ON t1.team_id= t2.team_id
        WHERE t2.local_date >= DATE_ADD(t1.local_date, INTERVAL -200 DAY)
        AND t2.local_date< t1.local_date
     GROUP BY t1.team_id, t1.local_date,t1.game_id;

-- calculating rolling BAIP for teams

DROP TABLE IF EXISTS s_team_rolling_BAIP;
CREATE TABLE s_team_rolling_BAIP
    (INDEX GAME_IND(game_id),INDEX TEAM_IND(team_id), INDEX p_team_BAIP(p_team_BAIP))
        SELECT t1.team_id,
               t1.game_id,
               t1.local_date,
               IF((sum(t2.p_AtBat) - SUM(t2.p_StrikeOut) - SUM(t2.p_Home_Run) + SUM(t2.p_Sac_Fly)) = 0, 0,
                            (SUM(t2.p_Hit) - SUM(t2.p_Home_Run))/(sum(t2.p_AtBat) - SUM(t2.p_StrikeOut) - SUM(t2.p_Home_Run) + SUM(t2.p_Sac_Fly))) as p_team_BAIP
            FROM s_team_pitching_stats AS t1 JOIN s_team_pitching_stats AS t2
            ON t1.team_id= t2.team_id
            WHERE t2.local_date >= DATE_ADD(t1.local_date, INTERVAL -200 DAY)
            AND t2.local_date< t1.local_date
         GROUP BY t1.team_id, t1.local_date,t1.game_id;

-- Calculating the rolling BB/k(walk to strike out ratio for batter) for teams

DROP TABLE IF EXISTS s_team_rolling_walk_to_strikeout;
CREATE TABLE s_team_rolling_walk_to_strikeout
    (INDEX GAME_INDEX(game_id),INDEX TEAM_INDEX(team_id), INDEX t_walk_to_Strikeout(t_walk_to_Strikeout))
        SELECT t1.team_id,
               t1.game_id,
               t1.local_date,
               IF (SUM(t2.b_StrikeOut)=0,0,SUM(t2.b_Walk)/(SUM(t2.b_StrikeOut))) AS t_walk_to_Strikeout
            FROM s_team_batting_stats AS t1 JOIN s_team_batting_stats AS t2
            ON t1.team_id= t2.team_id
            WHERE t2.local_date >= DATE_ADD(t1.local_date, INTERVAL -200 DAY)
            AND t2.local_date< t1.local_date
         GROUP BY t1.team_id, t1.local_date,t1.game_id;

-- Calculating the rolling K/BB(Strike out t ratio for batter) for teams

DROP TABLE IF EXISTS s_team_rolling_strikeout_to_walk;
CREATE TABLE s_team_rolling_strikeout_to_walk
    (INDEX GAME_INDEXING(game_id),INDEX TEAM_INDEXING(team_id), INDEX t_Strikeout_to_walk(t_Strikeout_to_walk))
        SELECT t1.team_id,
               t1.game_id,
               t1.local_date,
               IF (SUM(t2.p_Walk)=0,0,SUM(t2.p_StrikeOut)/(SUM(t2.p_Walk))) AS t_Strikeout_to_walk
            FROM s_team_pitching_stats AS t1 JOIN s_team_pitching_stats AS t2
            ON t1.team_id= t2.team_id
            WHERE t2.local_date >= DATE_ADD(t1.local_date, INTERVAL -200 DAY)
            AND t2.local_date< t1.local_date
         GROUP BY t1.team_id, t1.local_date,t1.game_id;

-- Calculating the On BAse Percentage OBP

DROP TABLE IF EXISTS s_team_rolling_OBP;
CREATE TABLE s_team_rolling_OBP
    (INDEX GAME_IN(game_id),INDEX TEAM_IN(team_id), INDEX b_team_OBP(b_team_OBP))
        SELECT t1.team_id,
               t1.game_id,
               t1.local_date,
               IF((sum(t2.b_AtBat) + SUM(t2.b_Walk) + SUM(t2.b_Hit_by_Pitch) + SUM(t2.b_Sac_Fly)) = 0, 0,
                            (SUM(t2.b_Hit) - SUM(t2.b_Walk) + SUM(t2.b_Hit_by_Pitch))/(sum(t2.b_AtBat) + SUM(t2.b_Walk) + SUM(t2.b_Hit_by_Pitch) + SUM(t2.b_Sac_Fly))) as b_team_OBP
            FROM s_team_batting_stats AS t1 JOIN s_team_batting_stats AS t2
            ON t1.team_id= t2.team_id
            WHERE t2.local_date >= DATE_ADD(t1.local_date, INTERVAL -200 DAY)
            AND t2.local_date< t1.local_date
         GROUP BY t1.team_id, t1.local_date,t1.game_id;


-- Calculating the starting pitching stats

DROP TABLE IF EXISTS s_starting_pitcher_table;
CREATE TABLE s_starting_pitcher_table
    (INDEX GAME_ID_IN(game_id),INDEX TEAM_ID_IN(team_id), INDEX PITCHER_IND(pitcher), INDEX Walk_IND(Walk), INDEX LOCAL_DATE_IND(local_date), INDEX INNINGS_PITCHED_IND(innings_pitched))
        SELECT game.game_id as game_id,
               pitcher_counts.team_id as team_id,
               DATE(game.local_date) as local_date,
               pitcher_counts.pitcher as pitcher,
               pitcher_counts.atBat as AtBat,
               pitcher_counts.Hit as Hit,
               pitcher_counts.Walk as Walk,
               pitcher_counts.Strikeout as StrikeOut,
               (pitcher_counts.endingInning - (pitcher_counts.startingInning - 1)) as innings_pitched
               FROM pitcher_counts
               JOIN game
               ON game.game_id = pitcher_counts.game_id
               WHERE startingInning = 1;

-- calculating walks(BB) per 9 innings pitched for the starting pitcher

DROP TABLE IF EXISTS s_starting_pitcher_BB_per_9inn;
CREATE TABLE s_starting_pitcher_BB_per_9inn
    (INDEX GAME_ID_INd(game_id),INDEX TEAM_ID_INd(team_id), INDEX k_per_9_innings_IND(BB_per_9_innings))
        SELECT t1.team_id,
               t1.game_id,
               t1.local_date,
               t1.pitcher,
               (SUM(t2.Walk*9)/SUM(t2.innings_pitched))AS BB_per_9_innings
            FROM s_starting_pitcher_table AS t1 JOIN s_starting_pitcher_table AS t2
            ON t1.team_id= t2.team_id
            WHERE t2.local_date >= DATE_ADD(t1.local_date, INTERVAL -200 DAY)
            AND t2.local_date< t1.local_date
         GROUP BY t1.team_id, t1.local_date,t1.game_id;

-- calculating hits allowed per 9 innings pitched for starting pitcher

DROP TABLE IF EXISTS s_starting_pitcher_H_per_9inn;
CREATE TABLE s_starting_pitcher_H_per_9inn
    (INDEX GAME_ID_INd(game_id),INDEX TEAM_ID_INd(team_id), INDEX H_per_9_innings_IND(H_per_9_innings))
        SELECT t1.team_id,
               t1.game_id,
               t1.local_date,
               t1.pitcher,
               (SUM(t2.Hit*9)/SUM(t2.innings_pitched))AS H_per_9_innings
            FROM s_starting_pitcher_table AS t1 JOIN s_starting_pitcher_table AS t2
            ON t1.team_id= t2.team_id
            WHERE t2.local_date >= DATE_ADD(t1.local_date, INTERVAL -200 DAY)
            AND t2.local_date< t1.local_date
         GROUP BY t1.team_id, t1.local_date,t1.game_id;

-- calculating strikeouts per innings pitched for starting pitcher
DROP TABLE IF EXISTS s_starting_pitcher_K_per_9inn;
CREATE TABLE s_starting_pitcher_K_per_9inn
    (INDEX GAME_ID_INd(game_id),INDEX TEAM_ID_INd(team_id), INDEX k_per_9_innings_IND(k_per_9_innings))
        SELECT t1.team_id,
               t1.game_id,
               t1.local_date,
               t1.pitcher,
               (SUM(t2.StrikeOut * 9)/SUM(t2.innings_pitched))AS k_per_9_innings
            FROM s_starting_pitcher_table AS t1 JOIN s_starting_pitcher_table AS t2
            ON t1.team_id= t2.team_id
            WHERE t2.local_date >= DATE_ADD(t1.local_date, INTERVAL -200 DAY)
            AND t2.local_date< t1.local_date
         GROUP BY t1.team_id, t1.local_date,t1.game_id;

-- calculating rolling WHIP for starting pitcher

DROP TABLE IF EXISTS s_starting_pitcher_rolling_WHIP;
CREATE TABLE s_starting_pitcher_rolling_WHIP
    (INDEX GAME_ID_INd(game_id),INDEX TEAM_ID_INd(team_id), INDEX starting_pitcher_rolling_WHIP_IND(starting_pitcher_rolling_WHIP))
        SELECT t1.team_id,
               t1.game_id,
               t1.local_date,
               t1.pitcher,
               IF (SUM(t2.innings_pitched)=0,0,(SUM(t2.Hit) + SUM(t2.Walk))/(SUM(t2.innings_pitched))) AS starting_pitcher_rolling_WHIP
            FROM s_starting_pitcher_table AS t1 JOIN s_starting_pitcher_table AS t2
            ON t1.team_id= t2.team_id
            WHERE t2.local_date >= DATE_ADD(t1.local_date, INTERVAL -200 DAY)
            AND t2.local_date< t1.local_date
         GROUP BY t1.team_id, t1.local_date,t1.game_id;

-- appending game and boxscore table

DROP TABLE IF EXISTS s_baseball_game_stats;
CREATE TABLE s_baseball_game_stats
       (INDEX GAME_ID_IN(game_id),INDEX home_team_id_IND(home_team_id),INDEX away_team_id_IND(away_team_id))
        SELECT boxscore.game_id as game_id,
               game.home_team_id as home_team_id,
               game.away_team_id as away_team_id,
               game.stadium_id as stadium_id,
               DATE(game.local_date) as local_date,
               boxscore.home_runs as home_team_runs,
               boxscore.home_hits as home_team_hits,
               boxscore.away_runs as away_team_runs,
               boxscore.away_hits as away_team_hits,
               boxscore.temp as temperature,
               boxscore.wind as wind,
               boxscore.winddir as wind_direction,
               boxscore.winner_home_or_away as winner
               FROM boxscore
        JOIN game
        ON boxscore.game_id = game.game_id;


-- Calculating Hits per run (HCP) for home team

DROP TABLE IF EXISTS s_home_team_HCP;
CREATE TABLE s_home_team_HCP
    (INDEX GAME_ID_INdex(game_id),INDEX HOME_TEAM_INDEX(home_team_id), INDEX home_team_HCP_IND(home_team_HCP))
        SELECT t1.home_team_id,
               t1.game_id,
               t1.local_date,
               IF (SUM(t2.home_team_runs)=0,0,(SUM(t2.home_team_hits))/(SUM(t2.home_team_runs))) AS home_team_HCP
            FROM s_baseball_game_stats AS t1 JOIN s_baseball_game_stats AS t2
            ON t1.home_team_id= t2.home_team_id
            WHERE t2.local_date >= DATE_ADD(t1.local_date, INTERVAL -200 DAY)
            AND t2.local_date< t1.local_date
         GROUP BY t1.home_team_id, t1.local_date,t1.game_id;

-- Calculating Hits per run (HCP) for Away team

DROP TABLE IF EXISTS s_away_team_HCP;
CREATE TABLE s_away_team_HCP
    (INDEX GAME_ID_INdex(game_id),INDEX AWAY_TEAM_INDEX(away_team_id), INDEX away_team_HCP_IND(away_team_HCP))
        SELECT t1.away_team_id,
               t1.game_id,
               t1.local_date,
               IF (SUM(t2.away_team_runs)=0,0,(SUM(t2.away_team_hits))/(SUM(t2.away_team_runs))) AS away_team_HCP
            FROM s_baseball_game_stats AS t1 JOIN s_baseball_game_stats AS t2
            ON t1.away_team_id= t2.away_team_id
            WHERE t2.local_date >= DATE_ADD(t1.local_date, INTERVAL -200 DAY)
            AND t2.local_date< t1.local_date
         GROUP BY t1.away_team_id, t1.local_date,t1.game_id;

-- combining all home team statistics in a table

DROP TABLE IF EXISTS baseball_home_game_stats;
CREATE TABLE baseball_home_game_stats
        (INDEX INDEX_GAME_ID(game_id))
            SELECT s_baseball_game_stats.game_id,
                   s_baseball_game_stats.local_date,
                   s_team_bat_rolling_avg.t_rolling_battingaverage               as home_team_batting_avg,
                   s_team_home_runs_per_hits.home_runs_per_hits                  as home_team_home_runs_per_hit,
                   s_team_pitch_rolling_avg.t_rolling_pitchingaverage            as home_team_pitching_avg,
                   s_team_rolling_BABIP.b_team_BABIP                             as home_team_BABIP,
                   s_team_rolling_BAIP.p_team_BAIP                               as home_team_BAIP,
                   s_team_rolling_walk_to_strikeout.t_walk_to_Strikeout          as home_team_striketowalk,
                   s_team_rolling_strikeout_to_walk.t_Strikeout_to_walk          as home_team_walktostrike,
                   s_team_rolling_OBP.b_team_OBP                                 as home_team_OBP,
                   s_home_team_HCP.home_team_HCP                                 as home_team_HCP,
                   s_starting_pitcher_rolling_WHIP.starting_pitcher_rolling_WHIP as home_team_startingpitcher_WHIP,
                   s_starting_pitcher_K_per_9inn.k_per_9_innings                 as home_team_startingpitcher_K_per_9_inn,
                   s_starting_pitcher_BB_per_9inn.BB_per_9_innings               as home_team_startingpitcher_BB_per_9_inn,
                   s_starting_pitcher_H_per_9inn.H_per_9_innings                 as home_team_startingpitcher_H_per_9_inn,
                   s_baseball_game_stats.winner
            FROM s_baseball_game_stats
            JOIN s_team_bat_rolling_avg
            ON s_baseball_game_stats.home_team_id = s_team_bat_rolling_avg.team_id and s_baseball_game_stats.game_id = s_team_bat_rolling_avg.game_id
            JOIN s_team_home_runs_per_hits
            ON s_baseball_game_stats.home_team_id = s_team_home_runs_per_hits.team_id and s_baseball_game_stats.game_id = s_team_home_runs_per_hits.game_id
            JOIN s_team_pitch_rolling_avg
            ON s_baseball_game_stats.home_team_id = s_team_pitch_rolling_avg.team_id and s_baseball_game_stats.game_id = s_team_pitch_rolling_avg.game_id
            JOIN s_team_rolling_BABIP
            ON s_baseball_game_stats.home_team_id = s_team_rolling_BABIP.team_id and s_baseball_game_stats.game_id = s_team_rolling_BABIP.game_id
            JOIN s_team_rolling_BAIP
            ON s_baseball_game_stats.home_team_id = s_team_rolling_BAIP.team_id and s_baseball_game_stats.game_id = s_team_rolling_BAIP.game_id
            JOIN s_team_rolling_walk_to_strikeout
            ON s_baseball_game_stats.home_team_id = s_team_rolling_walk_to_strikeout.team_id and s_baseball_game_stats.game_id = s_team_rolling_walk_to_strikeout.game_id
            JOIN s_team_rolling_strikeout_to_walk
            ON s_baseball_game_stats.home_team_id = s_team_rolling_strikeout_to_walk.team_id and s_baseball_game_stats.game_id = s_team_rolling_strikeout_to_walk.game_id
            JOIN s_team_rolling_OBP
            ON s_baseball_game_stats.home_team_id = s_team_rolling_OBP.team_id and s_baseball_game_stats.game_id = s_team_rolling_OBP.game_id
            JOIN s_starting_pitcher_rolling_WHIP
            ON s_baseball_game_stats.home_team_id = s_starting_pitcher_rolling_WHIP.team_id and s_baseball_game_stats.game_id = s_starting_pitcher_rolling_WHIP.game_id
            JOIN s_home_team_HCP
            ON s_baseball_game_stats.home_team_id = s_home_team_HCP.home_team_id and s_baseball_game_stats.game_id = s_home_team_HCP.game_id
            JOIN s_starting_pitcher_K_per_9inn
            ON s_baseball_game_stats.home_team_id = s_starting_pitcher_K_per_9inn.team_id and s_baseball_game_stats.game_id = s_starting_pitcher_K_per_9inn.game_id
            JOIN s_starting_pitcher_BB_per_9inn
            ON s_baseball_game_stats.home_team_id = s_starting_pitcher_BB_per_9inn.team_id and s_baseball_game_stats.game_id = s_starting_pitcher_BB_per_9inn.game_id
            JOIN s_starting_pitcher_H_per_9inn
            ON s_baseball_game_stats.home_team_id = s_starting_pitcher_H_per_9inn.team_id and s_baseball_game_stats.game_id = s_starting_pitcher_H_per_9inn.game_id;


-- combining all away team statistics in a table

DROP TABLE IF EXISTS baseball_away_game_stats;
CREATE TABLE baseball_away_game_stats
    (INDEX INDEXING_GAME_ID(game_id))
         SELECT s_baseball_game_stats.game_id,
               s_team_bat_rolling_avg.t_rolling_battingaverage               as away_team_batting_avg,
               s_team_home_runs_per_hits.home_runs_per_hits                  as away_team_home_runs_per_hit,
               s_team_pitch_rolling_avg.t_rolling_pitchingaverage            as away_team_pitching_avg,
               s_team_rolling_BABIP.b_team_BABIP                             as away_team_BABIP,
               s_team_rolling_BAIP.p_team_BAIP                               as away_team_BAIP,
               s_team_rolling_walk_to_strikeout.t_walk_to_Strikeout          as away_team_striketowalk,
               s_team_rolling_strikeout_to_walk.t_Strikeout_to_walk          as away_team_walktostrike,
               s_team_rolling_OBP.b_team_OBP                                 as away_team_OBP,
               s_away_team_HCP.away_team_HCP                                 as away_team_HCP,
               s_starting_pitcher_rolling_WHIP.starting_pitcher_rolling_WHIP as away_team_startingpitcher_WHIP,
               s_starting_pitcher_K_per_9inn.k_per_9_innings                 as away_team_startingpitcher_K_per_9_inn,
               s_starting_pitcher_BB_per_9inn.BB_per_9_innings               as away_team_startingpitcher_BB_per_9_inn,
               s_starting_pitcher_H_per_9inn.H_per_9_innings as away_team_startingpitcher_H_per_9_inn,
               s_baseball_game_stats.winner
        FROM s_baseball_game_stats
        JOIN s_team_bat_rolling_avg
        ON s_baseball_game_stats.away_team_id = s_team_bat_rolling_avg.team_id and s_baseball_game_stats.game_id = s_team_bat_rolling_avg.game_id
        JOIN s_team_home_runs_per_hits
        ON s_baseball_game_stats.away_team_id = s_team_home_runs_per_hits.team_id and s_baseball_game_stats.game_id = s_team_home_runs_per_hits.game_id
        JOIN s_team_pitch_rolling_avg
        ON s_baseball_game_stats.away_team_id = s_team_pitch_rolling_avg.team_id and s_baseball_game_stats.game_id = s_team_pitch_rolling_avg.game_id
        JOIN s_team_rolling_BABIP
        ON s_baseball_game_stats.away_team_id = s_team_rolling_BABIP.team_id and s_baseball_game_stats.game_id = s_team_rolling_BABIP.game_id
        JOIN s_team_rolling_BAIP
        ON s_baseball_game_stats.away_team_id = s_team_rolling_BAIP.team_id and s_baseball_game_stats.game_id = s_team_rolling_BAIP.game_id
        JOIN s_team_rolling_walk_to_strikeout
        ON s_baseball_game_stats.away_team_id = s_team_rolling_walk_to_strikeout.team_id and s_baseball_game_stats.game_id = s_team_rolling_walk_to_strikeout.game_id
        JOIN s_team_rolling_strikeout_to_walk
        ON s_baseball_game_stats.away_team_id = s_team_rolling_strikeout_to_walk.team_id and s_baseball_game_stats.game_id = s_team_rolling_strikeout_to_walk.game_id
        JOIN s_team_rolling_OBP
        ON s_baseball_game_stats.away_team_id = s_team_rolling_OBP.team_id and s_baseball_game_stats.game_id = s_team_rolling_OBP.game_id
        JOIN s_starting_pitcher_rolling_WHIP
        ON s_baseball_game_stats.away_team_id = s_starting_pitcher_rolling_WHIP.team_id and s_baseball_game_stats.game_id = s_starting_pitcher_rolling_WHIP.game_id
        JOIN s_away_team_HCP
        ON s_baseball_game_stats.away_team_id = s_away_team_HCP.away_team_id and s_baseball_game_stats.game_id = s_away_team_HCP.game_id
        JOIN s_starting_pitcher_K_per_9inn
        ON s_baseball_game_stats.away_team_id = s_starting_pitcher_K_per_9inn.team_id and s_baseball_game_stats.game_id = s_starting_pitcher_K_per_9inn.game_id
        JOIN s_starting_pitcher_BB_per_9inn
        ON s_baseball_game_stats.away_team_id = s_starting_pitcher_BB_per_9inn.team_id and s_baseball_game_stats.game_id = s_starting_pitcher_BB_per_9inn.game_id
        JOIN s_starting_pitcher_H_per_9inn
        ON s_baseball_game_stats.away_team_id = s_starting_pitcher_H_per_9inn.team_id and s_baseball_game_stats.game_id = s_starting_pitcher_H_per_9inn.game_id;


-- combining all team features to a final table

DROP TABLE IF EXISTS baseball_game_stats;
CREATE TABLE baseball_game_stats

    SELECT baseball_home_game_stats.game_id,
           local_date,
           home_team_batting_avg as HT_BA,
           away_team_batting_avg as AT_BA,
           home_team_home_runs_per_hit as HT_home_runs_per_hit,
           away_team_home_runs_per_hit as AT_home_runs_per_hit,
           home_team_pitching_avg as HT_PA,
           away_team_pitching_avg as AT_PA,
           home_team_BABIP as HT_BABIP,
           away_team_BABIP as AT_BABIP,
           home_team_BAIP as HT_BAIP,
           away_team_BAIP as AT_BAIP,
           home_team_striketowalk as HT_K_to_BB,
           away_team_striketowalk as AT_K_to_BB,
           home_team_walktostrike as HT_BB_to_K,
           away_team_walktostrike as AT_BB_to_K,
           home_team_OBP as HT_OBP,
           away_team_OBP as AT_OBP,
           home_team_HCP as HT_HCP,
           away_team_HCP as AT_HCP,
           home_team_startingpitcher_WHIP as HT_SP_WHIP,
           away_team_startingpitcher_WHIP as AT_SP_WHIP,
           home_team_startingpitcher_K_per_9_inn as HT_SP_K_per_9inn,
           away_team_startingpitcher_K_per_9_inn as AT_SP_K_per_9inn,
           home_team_startingpitcher_BB_per_9_inn as HT_SP_BB_per_9_inn,
           away_team_startingpitcher_BB_per_9_inn as AT_SP_BB_per_9_inn,
           home_team_startingpitcher_H_per_9_inn HT_SP_H_per_9_inn,
           away_team_startingpitcher_H_per_9_inn AT_SP_H_per_9_inn,
           baseball_home_game_stats.winner as HT_Wins
           FROM baseball_home_game_stats
    JOIN baseball_away_game_stats
    ON baseball_home_game_stats.game_id = baseball_away_game_stats.game_id
    ORDER BY local_date;

-- Updating the winner column by replacing H and A with 1 and 0

UPDATE baseball_game_stats
SET HT_Wins= '0' WHERE HT_Wins = '';
UPDATE baseball_game_stats
SET HT_Wins = REPLACE(HT_Wins,'A','0');
UPDATE baseball_game_stats
SET HT_Wins = REPLACE(HT_Wins,'H','1');


SELECT * FROM baseball_game_stats LIMIT 0,20;
