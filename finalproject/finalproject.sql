-- Using the baseball database
USE baseball;
-- appending the local date to the team_batting_stats and team_pitching_stats and storing in a table

DROP TEMPORARY TABLE IF EXISTS s_team_batting_stats;
CREATE TEMPORARY TABLE s_team_batting_stats
    (INDEX Game_id_Ind(game_id),INDEX Team_id_ind(team_id), INDEX Local_date(local_date), INDEX IND_away_team_id(away_team_id))
        SELECT team_batting_counts.game_id as game_id,
               team_batting_counts.team_id as team_id,
               team_batting_counts.homeTeam as HT,
               UNIX_TIMESTAMP(game.local_date) as local_date,
               team_batting_counts.atBat as b_AtBat,
               team_batting_counts.Hit as b_Hit,
               team_batting_counts.finalScore as Runs,
               team_batting_counts.Home_Run as b_Home_Run,
               team_batting_counts.Sac_Fly as b_Sac_Fly,
               team_batting_counts.Strikeout as b_StrikeOut,
               team_batting_counts.Walk as b_Walk,
               team_batting_counts.Hit_By_Pitch as b_Hit_by_Pitch,
               ((1*team_batting_counts.Single)+(2*team_batting_counts.`Double`)+(3*team_batting_counts.Triple)+(4*team_batting_counts.Home_Run)) as b_total_bases,
               (team_batting_counts.stolenBase2B+team_batting_counts.stolenBase3B+team_batting_counts.stolenBaseHome) as b_stolen_bases,
               (team_batting_counts.caughtStealing2B+team_batting_counts.caughtStealing3B+team_batting_counts.caughtStealingHome) as b_caught_stealing,
               team_batting_counts.Grounded_Into_DP as b_GIDP,
               team_batting_counts.finalScore as t_runs_scored,
               team_batting_counts.opponent_finalScore as t_runs_allowed,
               team_batting_counts.opponent_team_id as away_team_id,
               team_batting_counts.inning as innings,
               team_batting_counts.plateApperance as b_plateApperance
               FROM team_batting_counts
        JOIN game
        ON game.game_id = team_batting_counts.game_id;

DROP TEMPORARY TABLE IF EXISTS s_team_bat_50_day_rolling_stats;
CREATE TEMPORARY TABLE s_team_bat_50_day_rolling_stats
(INDEX Game_id_IND(game_id),INDEX Team_id_IND(team_id))
SELECT t1.team_id,
       t1.game_id,
       t1.local_date,
       t1.HT as HT,
       IF (SUM(t2.b_AtBat)=0,0,SUM(t2.b_Hit)/(SUM(t2.b_AtBat))) AS b_t_50_batting_avg,
       IF (SUM(t2.b_Hit)=0,0,SUM(t2.b_Home_Run)/(SUM(t2.b_Hit))) AS b_t_50_home_runs_per_hits,
       IF (SUM(t2.b_AtBat)=0,0,SUM(t2.b_Home_Run)/(SUM(t2.b_AtBat))) AS b_t_50_HRR,
       IF((sum(t2.b_AtBat) - SUM(t2.b_Hit) + SUM(t2.b_caught_stealing) + SUM(t2.b_GIDP)) = 0, 0,
            (SUM(t2.b_total_bases) + SUM(t2.b_Hit_by_Pitch) + SUM(t2.b_Walk) + SUM(t2.b_stolen_bases))/(sum(t2.b_AtBat) - SUM(t2.b_Hit) + SUM(t2.b_caught_stealing) + SUM(t2.b_GIDP))) as b_t_50_total_avg,
       IF((sum(t2.b_AtBat) - SUM(t2.b_StrikeOut) - SUM(t2.b_Home_Run) + SUM(t2.b_Sac_Fly)) = 0, 0,
            (SUM(t2.b_Hit) - SUM(t2.b_Home_Run))/(sum(t2.b_AtBat) - SUM(t2.b_StrikeOut) - SUM(t2.b_Home_Run) + SUM(t2.b_Sac_Fly))) as b_t_50_BABIP,
       IF (SUM(t2.b_StrikeOut)=0,0,SUM(t2.b_Walk)/(SUM(t2.b_StrikeOut))) AS b_t_50_walk_to_Strikeout,
       IF ((SUM(t2.t_runs_allowed)+SUM(t2.t_runs_scored))=0,0,SUM(POWER(t2.t_runs_scored,2))/(SUM(POWER(t2.t_runs_allowed,2))+SUM(POWER(t2.t_runs_scored,2)))) AS t_b_50_PE,
       IF((sum(t2.b_AtBat) + SUM(t2.b_Walk) + SUM(t2.b_Hit_by_Pitch) + SUM(t2.b_Sac_Fly)) = 0, 0,
                    (SUM(t2.b_Hit) - SUM(t2.b_Walk) + SUM(t2.b_Hit_by_Pitch))/(sum(t2.b_AtBat) + SUM(t2.b_Walk) + SUM(t2.b_Hit_by_Pitch) + SUM(t2.b_Sac_Fly))) as b_t_50_OBP,
       IF (SUM(t2.b_plateApperance)=0,0,SUM(t2.b_Walk)/(SUM(t2.b_plateApperance))) AS b_t_50_BBP,
       IF (SUM(t2.b_AtBat)=0,0,(SUM(t2.b_total_bases) - SUM(t2.b_Hit))/(SUM(t2.b_AtBat))) AS b_t_50_Isolated_power
    FROM s_team_batting_stats AS t1 JOIN s_team_batting_stats AS t2
    ON t1.team_id= t2.team_id
    WHERE t2.local_date >= (t1.local_date - 4320000)
    AND t2.local_date< t1.local_date
    GROUP BY t1.team_id, t1.local_date,t1.game_id
    ORDER BY t1.team_id, t1.local_date,t1.game_id;

DROP TEMPORARY TABLE IF EXISTS s_team_bat_100_day_rolling_stats;
CREATE TEMPORARY TABLE s_team_bat_100_day_rolling_stats
(INDEX Game_id_IND(game_id),INDEX Team_id_IND(team_id))
SELECT t1.team_id,
       t1.game_id,
       t1.local_date,
       t1.HT as HT,
       IF (SUM(t2.b_AtBat)=0,0,SUM(t2.b_Hit)/(SUM(t2.b_AtBat))) AS b_t_100_batting_avg,
       IF (SUM(t2.b_Hit)=0,0,SUM(t2.b_Home_Run)/(SUM(t2.b_Hit))) AS b_t_100_home_runs_per_hits,
       IF (SUM(t2.b_AtBat)=0,0,SUM(t2.b_Home_Run)/(SUM(t2.b_AtBat))) AS b_t_100_HRR,
       IF((sum(t2.b_AtBat) - SUM(t2.b_Hit) + SUM(t2.b_caught_stealing) + SUM(t2.b_GIDP)) = 0, 0,
            (SUM(t2.b_total_bases) + SUM(t2.b_Hit_by_Pitch) + SUM(t2.b_Walk) + SUM(t2.b_stolen_bases))/(sum(t2.b_AtBat) - SUM(t2.b_Hit) + SUM(t2.b_caught_stealing) + SUM(t2.b_GIDP))) as b_t_100_total_avg,
       IF((sum(t2.b_AtBat) - SUM(t2.b_StrikeOut) - SUM(t2.b_Home_Run) + SUM(t2.b_Sac_Fly)) = 0, 0,
            (SUM(t2.b_Hit) - SUM(t2.b_Home_Run))/(sum(t2.b_AtBat) - SUM(t2.b_StrikeOut) - SUM(t2.b_Home_Run) + SUM(t2.b_Sac_Fly))) as b_t_100_BABIP,
       IF (SUM(t2.b_StrikeOut)=0,0,SUM(t2.b_Walk)/(SUM(t2.b_StrikeOut))) AS b_t_100_walk_to_Strikeout,
       IF ((SUM(t2.t_runs_allowed)+SUM(t2.t_runs_scored))=0,0,SUM(POWER(t2.t_runs_scored,2))/(SUM(POWER(t2.t_runs_allowed,2))+SUM(POWER(t2.t_runs_scored,2)))) AS t_b_100_PE,
       IF((sum(t2.b_AtBat) + SUM(t2.b_Walk) + SUM(t2.b_Hit_by_Pitch) + SUM(t2.b_Sac_Fly)) = 0, 0,
                    (SUM(t2.b_Hit) - SUM(t2.b_Walk) + SUM(t2.b_Hit_by_Pitch))/(sum(t2.b_AtBat) + SUM(t2.b_Walk) + SUM(t2.b_Hit_by_Pitch) + SUM(t2.b_Sac_Fly))) as b_t_100_OBP,
       IF ((SUM(t2.b_stolen_bases)+SUM(t2.b_caught_stealing))=0,0,SUM(t2.b_stolen_bases)/(SUM(t2.b_stolen_bases)+SUM(t2.b_caught_stealing))) AS b_t_100_SBP,
       IF (SUM(t2.b_plateApperance)=0,0,SUM(t2.b_Walk)/(SUM(t2.b_plateApperance))) AS b_t_100_BBP,
       (0.3* (SUM(t2.b_stolen_bases)) - 0.6 * (SUM(t2.b_stolen_bases))) as b_t_100_SBR ,
       IF (SUM(t2.b_AtBat)=0,0,(SUM(t2.b_total_bases) - SUM(t2.b_Hit))/(SUM(t2.b_AtBat))) AS b_t_100_Isolated_power
    FROM s_team_batting_stats AS t1 JOIN s_team_batting_stats AS t2
    ON t1.team_id= t2.team_id
    WHERE t2.local_date >= (t1.local_date - 8640000)
    AND t2.local_date< t1.local_date
    GROUP BY t1.team_id, t1.local_date,t1.game_id
    ORDER BY t1.team_id, t1.local_date,t1.game_id;

DROP TEMPORARY TABLE IF EXISTS s_team_pitching_stats;
CREATE TEMPORARY TABLE s_team_pitching_stats
    (INDEX Game_id_ind(game_id),INDEX Team_id_ind(team_id), INDEX Local_Date(local_date))
    SELECT team_pitching_counts.game_id as game_id,
           team_pitching_counts.team_id as team_id,
           team_pitching_counts.homeTeam as HT,
           UNIX_TIMESTAMP(game.local_date) as local_date,
           team_pitching_counts.atBat as p_AtBat,
           team_pitching_counts.Hit as p_Hit,
           team_pitching_counts.finalScore as Runs,
           team_pitching_counts.Home_Run as p_Home_Run,
           team_pitching_counts.Sac_Fly as p_Sac_Fly,
           team_pitching_counts.Strikeout as p_StrikeOut,
           team_pitching_counts.Walk as p_Walk,
           team_pitching_counts.Hit_By_Pitch as p_Hit_By_Pitch,
           team_pitching_counts.plateApperance as p_plateApperance,
           s_team_batting_stats.innings as innings,
           team_batting_counts.opponent_finalScore as runs_allowed
           FROM team_pitching_counts
        JOIN game
        ON game.game_id = team_pitching_counts.game_id
        JOIN team_batting_counts
        ON team_pitching_counts.game_id = team_batting_counts.game_id AND team_pitching_counts.team_id = team_batting_counts.team_id
        JOIN s_team_batting_stats
        ON team_pitching_counts.game_id = s_team_batting_stats.game_id AND team_pitching_counts.team_id = s_team_batting_stats.away_team_id;

DROP TEMPORARY TABLE IF EXISTS s_team_pitch_50_day_rolling_stats;
CREATE TEMPORARY TABLE s_team_pitch_50_day_rolling_stats
(INDEX Game_iD_IND(game_id),INDEX Team_iD_IND(team_id))
SELECT t1.team_id,
       t1.game_id,
       t1.local_date,
       t1.HT as HT,
       9 * (SUM(t2.p_Walk)/SUM(t2.innings))AS t_p_50_BB_per_9_innings,
       9 * (SUM(t2.p_Hit)/SUM(t2.innings))AS t_p_50_H_per_9_innings,
       9 * (SUM(t2.p_StrikeOut)/SUM(t2.innings))AS t_p_50_k_per_9_innings,
       9 * (SUM(t2.runs_allowed)/SUM(t2.innings))AS t_p_50_ERA,
       IF (SUM(t2.innings)=0,0,(SUM(t2.p_Hit) + SUM(t2.p_Walk))/(SUM(t2.innings))) AS t_p_50_WHIP,
       IF (SUM(t2.innings)=0,0, 3 + ((SUM(13* t2.p_Home_Run) + SUM(3* (t2.p_Walk + t2.p_Hit_by_pitch)) - SUM(2* t2.p_StrikeOut))/(SUM(t2.innings)))) AS t_p_50_DICE,
       IF (SUM(t2.p_AtBat)=0,0,SUM(t2.p_Hit)/(SUM(t2.p_AtBat))) AS t_p_50_OBA,
       IF((sum(t2.p_AtBat) - SUM(t2.p_StrikeOut) - SUM(t2.p_Home_Run) + SUM(t2.p_Sac_Fly)) = 0, 0,
                        (SUM(t2.p_Hit) - SUM(t2.p_Home_Run))/(sum(t2.p_AtBat) - SUM(t2.p_StrikeOut) - SUM(t2.p_Home_Run) + SUM(t2.p_Sac_Fly))) as t_p_50_BAIP,
       IF (SUM(t2.p_Walk)=0,0,SUM(t2.p_StrikeOut)/(SUM(t2.p_Walk))) AS t_p_50_Strikeout_to_walk
    FROM s_team_pitching_stats AS t1 JOIN s_team_pitching_stats AS t2
    ON t1.team_id= t2.team_id
    WHERE t2.local_date >= (t1.local_date - 4320000)
    AND t2.local_date< t1.local_date
    GROUP BY t1.team_id, t1.local_date,t1.game_id;

DROP TEMPORARY TABLE IF EXISTS s_team_pitch_100_day_rolling_stats;
CREATE TEMPORARY TABLE s_team_pitch_100_day_rolling_stats
(INDEX Game_iD_IND(game_id),INDEX Team_iD_IND(team_id))
SELECT t1.team_id,
       t1.game_id,
       t1.local_date,
       t1.HT as HT,
       9 * (SUM(t2.p_Walk)/SUM(t2.innings))AS t_p_100_BB_per_9_innings,
       9 * (SUM(t2.p_Hit)/SUM(t2.innings))AS t_p_100_H_per_9_innings,
       9 * (SUM(t2.p_StrikeOut)/SUM(t2.innings))AS t_p_100_k_per_9_innings,
       9 * (SUM(t2.runs_allowed)/SUM(t2.innings))AS t_p_100_ERA,
       IF (SUM(t2.innings)=0,0,(SUM(t2.p_Hit) + SUM(t2.p_Walk))/(SUM(t2.innings))) AS t_p_100_WHIP,
       IF (SUM(t2.innings)=0,0, 3 + ((SUM(13* t2.p_Home_Run) + SUM(3* (t2.p_Walk + t2.p_Hit_by_pitch)) - SUM(2* t2.p_StrikeOut))/(SUM(t2.innings)))) AS t_p_100_DICE,
       IF (SUM(t2.p_AtBat)=0,0,SUM(t2.p_Hit)/(SUM(t2.p_AtBat))) AS t_p_100_OBA,
       IF((sum(t2.p_AtBat) - SUM(t2.p_StrikeOut) - SUM(t2.p_Home_Run) + SUM(t2.p_Sac_Fly)) = 0, 0,
                        (SUM(t2.p_Hit) - SUM(t2.p_Home_Run))/(sum(t2.p_AtBat) - SUM(t2.p_StrikeOut) - SUM(t2.p_Home_Run) + SUM(t2.p_Sac_Fly))) as t_p_100_BAIP,
       IF (SUM(t2.p_Walk)=0,0,SUM(t2.p_StrikeOut)/(SUM(t2.p_Walk))) AS t_p_100_Strikeout_to_walk
    FROM s_team_pitching_stats AS t1 JOIN s_team_pitching_stats AS t2
    ON t1.team_id= t2.team_id
    WHERE t2.local_date >= (t1.local_date - 8640000)
    AND t2.local_date< t1.local_date
    GROUP BY t1.team_id, t1.local_date,t1.game_id;

DROP TEMPORARY TABLE IF EXISTS s_starting_pitcher_table;
CREATE TEMPORARY TABLE s_starting_pitcher_table
(INDEX GAME_ID_IN(game_id),INDEX TEAM_ID_IN(team_id), INDEX PITCHER_IND(pitcher), INDEX Walk_IND(Walk), INDEX LOCAL_DATE_IND(local_date), INDEX INNINGS_PITCHED_IND(innings_pitched))
SELECT game.game_id as game_id,
       pitcher_counts.team_id as team_id,
       pitcher_counts.homeTeam as HT,
       UNIX_TIMESTAMP(game.local_date) as local_date,
       pitcher_counts.pitcher as pitcher,
       pitcher_counts.atBat as AtBat,
       pitcher_counts.Hit as Hit,
       pitcher_counts.Walk as Walk,
       pitcher_counts.Home_Run as Home_Run,
       pitcher_counts.Hit_By_Pitch as Hit_by_pitch,
       pitcher_counts.Strikeout as StrikeOut,
       pitcher_counts.pitchesThrown as pitches_thrown,
       pitcher_counts.DaysSinceLastPitch as Days_got_rest,
       (pitcher_counts.endingInning - (pitcher_counts.startingInning - 1)) as innings_pitched,
       CASE WHEN (pitcher_counts.endingInning - (pitcher_counts.startingInning - 1)) = (game.inning_loaded)/2 THEN 1 else 0 END AS finished_innings
       FROM pitcher_counts
       JOIN game
       ON game.game_id = pitcher_counts.game_id
       WHERE startingInning = 1;

DROP TEMPORARY TABLE IF EXISTS s_starting_pitcher_50_rolling_stats;
CREATE TEMPORARY TABLE s_starting_pitcher_50_rolling_stats
(INDEX GAME_ID_INd(game_id),INDEX TEAM_ID_INd(team_id))
    SELECT t1.team_id,
           t1.game_id,
           t1.local_date,
           t1.pitcher,
           t1.HT as HT,
           (SUM(t2.Walk*9)/SUM(t2.innings_pitched))AS sp_50_BB_per_9_innings,
           (SUM(t2.Hit*9)/SUM(t2.innings_pitched))AS sp_50_H_per_9_innings,
           (SUM(t2.StrikeOut * 9)/SUM(t2.innings_pitched))AS sp_50_k_per_9_innings,
           IF (SUM(t2.innings_pitched)=0,0,(SUM(t2.Hit) + SUM(t2.Walk))/(SUM(t2.innings_pitched))) AS sp_50_WHIP,
           IF (SUM(t2.innings_pitched)=0,0, 3 + ((SUM(13* t2.Home_Run) + SUM(3* (t2.Walk + t2.Hit_by_pitch)) - SUM(2* t2.StrikeOut))/(SUM(t2.innings_pitched)))) AS sp_50_DICE,
           SUM(t2.finished_innings) as sp_50_inings_finished,
           (SUM(t2.AtBat)/SUM(t2.pitches_thrown)) as sp_50_atBat_per_pt,
           t2.Days_got_rest as sp_Days_got_rest
        FROM s_starting_pitcher_table AS t1 JOIN s_starting_pitcher_table AS t2
        ON t1.team_id= t2.team_id
        WHERE t2.local_date >= (t1.local_date - 4320000)
        AND t2.local_date< t1.local_date
        GROUP BY t1.team_id, t1.local_date,t1.game_id;

DROP TEMPORARY TABLE IF EXISTS s_starting_pitcher_100_rolling_stats;
CREATE TEMPORARY TABLE s_starting_pitcher_100_rolling_stats
(INDEX GAME_ID_INd(game_id),INDEX TEAM_ID_INd(team_id))
    SELECT t1.team_id,
           t1.game_id,
           t1.local_date,
           t1.pitcher,
           t1.HT as HT,
           (SUM(t2.Walk*9)/SUM(t2.innings_pitched))AS sp_100_BB_per_9_innings,
           (SUM(t2.Hit*9)/SUM(t2.innings_pitched))AS sp_100_H_per_9_innings,
           (SUM(t2.StrikeOut * 9)/SUM(t2.innings_pitched))AS sp_100_k_per_9_innings,
           IF (SUM(t2.innings_pitched)=0,0,(SUM(t2.Hit) + SUM(t2.Walk))/(SUM(t2.innings_pitched))) AS sp_100_WHIP,
           IF (SUM(t2.innings_pitched)=0,0, 3 + ((SUM(13* t2.Home_Run) + SUM(3* (t2.Walk + t2.Hit_by_pitch)) - SUM(2* t2.StrikeOut))/(SUM(t2.innings_pitched)))) AS sp_100_DICE,
           SUM(t2.finished_innings) as sp_100_inings_finished,
           (SUM(t2.AtBat)/SUM(t2.pitches_thrown)) as sp_100_atBat_per_pt,
           t2.Days_got_rest as sp_Days_got_rest
        FROM s_starting_pitcher_table AS t1 JOIN s_starting_pitcher_table AS t2
        ON t1.team_id= t2.team_id
        WHERE t2.local_date >= (t1.local_date - 8640000)
        AND t2.local_date< t1.local_date
        GROUP BY t1.team_id, t1.local_date,t1.game_id;


DROP TEMPORARY TABLE IF EXISTS s_team_stats;
CREATE TEMPORARY TABLE s_team_stats
SELECT boxscore.game_id,
       UNIX_TIMESTAMP(game.local_date) as local_date,
       REPLACE(boxscore.temp, ' degrees', '' ) as temperature,
       REPLACE(boxscore.wind,' mph','') as wind_speed,
       boxscore.overcast as overcast,
       boxscore.winddir as wind_direction,
       game.home_team_id as HT_id,
       game.away_team_id as AT_id,
       game.stadium_id as stadium_id,
       boxscore.home_runs as home_team_runs,
       boxscore.home_hits as home_team_hits,
       boxscore.away_runs as away_team_runs,
       boxscore.away_hits as away_team_hits,
       CASE WHEN ((boxscore.home_runs) - (boxscore.away_runs)) >= 0 THEN 1 else 0 END AS HT_Wins
       FROM boxscore
       JOIN game
       ON game.game_id = boxscore.game_id;



-- Calculating Hits per run (HCP) for home team

DROP TEMPORARY TABLE IF EXISTS s_home_team_HCP;
CREATE TEMPORARY TABLE s_home_team_HCP
    (INDEX GAME_ID_INdex(game_id),INDEX HOME_TEAM_INDEX(HT_id), INDEX home_team_HCP_IND(home_team_HCP))
        SELECT t1.HT_id,
               t1.game_id,
               t1.local_date,
               IF (SUM(t2.home_team_runs)=0,0,(SUM(t2.home_team_hits))/(SUM(t2.home_team_runs))) AS home_team_HCP
            FROM s_team_stats AS t1 JOIN s_team_stats AS t2
            ON t1.HT_id= t2.HT_id
            WHERE t2.local_date >= (t1.local_date - 4320000)
            AND t2.local_date< t1.local_date
         GROUP BY t1.HT_id, t1.local_date,t1.game_id;

-- Calculating Hits per run (HCP) for Away team

DROP TEMPORARY TABLE IF EXISTS s_away_team_HCP;
CREATE TEMPORARY TABLE s_away_team_HCP
    (INDEX GAME_ID_INdex(game_id),INDEX AWAY_TEAM_INDEX(AT_id), INDEX away_team_HCP_IND(away_team_HCP))
        SELECT t1.AT_id,
               t1.game_id,
               t1.local_date,
               IF (SUM(t2.away_team_runs)=0,0,(SUM(t2.away_team_hits))/(SUM(t2.away_team_runs))) AS away_team_HCP
            FROM s_team_stats AS t1 JOIN s_team_stats AS t2
            ON t1.AT_id= t2.AT_id
            WHERE t2.local_date >= (t1.local_date - 4320000)
            AND t2.local_date< t1.local_date
         GROUP BY t1.AT_id, t1.local_date,t1.game_id;

DROP TEMPORARY TABLE IF EXISTS s_home_team_stats;
CREATE TEMPORARY TABLE s_home_team_stats
(INDEX GAME_ID_INdexing(game_id),INDEX HOME_TEAM_INDEXing(HT_id), INDEX HT_id_IND(HT_id), INDEX AT_id_IND(AT_id))
SELECT s_team_stats.game_id as game_id,
       (s_team_stats.local_date) AS game_date,
       s_team_stats.HT_id,
       s_team_stats.AT_id,
       s_team_stats.temperature,
       REPLACE(s_team_stats.wind_speed,'Indoors','0') as wind_speed,
       s_team_stats.wind_direction,
       s_team_stats.stadium_id,
       b.b_t_50_BABIP AS HT_50_BABIP,
       b.b_t_50_batting_avg AS HT_50_BA,
       b.b_t_50_home_runs_per_hits AS HT_50_HR_per_H,
       b.b_t_50_HRR AS HT_50_HRR,
       b.b_t_50_Isolated_power AS HT_50_IP,
       b.b_t_50_OBP AS HT_50_OBP,
       b.b_t_50_total_avg AS HT_50_TA,
       b.t_b_50_PE AS HT_50_PE,
       b.b_t_50_BBP AS HT_50_BBP,
       b.b_t_50_walk_to_Strikeout AS HT_50_K_to_BB,
       b1.b_t_100_BABIP AS HT_100_BABIP,
       b1.b_t_100_batting_avg AS HT_100_BA,
       b1.b_t_100_home_runs_per_hits AS HT_100_HR_per_H,
       b1.b_t_100_HRR AS HT_100_HRR,
       b1.b_t_100_Isolated_power AS HT_100_IP,
       b1.b_t_100_OBP AS HT_100_OBP,
       b1.b_t_100_total_avg AS HT_100_TA,
       b1.t_b_100_PE AS HT_100_PE,
       b1.b_t_100_BBP AS HT_100_BBP,
       b1.b_t_100_walk_to_Strikeout AS HT_100_K_to_BB,
       b1.b_t_100_SBP AS HT_100_SBP,
       b1.b_t_100_SBR AS HT_100_SBR,
       p.t_p_50_H_per_9_innings AS HT_50_H_9IP,
       p.t_p_50_BAIP AS HT_50_BAIP,
       p.t_p_50_BB_per_9_innings AS HT_50_BB_9IP,
       p.t_p_50_DICE AS HT_50_DICE,
       p.t_p_50_ERA AS HT_50_ERA,
       p.t_p_50_k_per_9_innings AS HT_50_K_9IP,
       p.t_p_50_WHIP AS HT_50_WHIP,
       p.t_p_50_OBA AS HT_50_OBA,
       p.t_p_50_Strikeout_to_walk AS HT_50_K_BB,
       p1.t_p_100_H_per_9_innings AS HT_100_H_9IP,
       p1.t_p_100_BAIP AS HT_100_BAIP,
       p1.t_p_100_BB_per_9_innings AS HT_100_BB_9IP,
       p1.t_p_100_DICE AS HT_100_DICE,
       p1.t_p_100_ERA AS HT_100_ERA,
       p1.t_p_100_k_per_9_innings AS HT_100_K_9IP,
       p1.t_p_100_WHIP AS HT_100_WHIP,
       p1.t_p_100_OBA AS HT_100_OBA,
       p1.t_p_100_Strikeout_to_walk AS HT_100_K_BB,
       sp.sp_50_atBat_per_pt AS HT_SP_50_AB_PT,
       sp.sp_50_BB_per_9_innings AS HT_SP_50_BB_9IP,
       sp.sp_Days_got_rest AS HT_SP_50_DGR,
       sp.sp_50_inings_finished AS HT_SP_50_IF,
       sp.sp_50_WHIP AS HT_SP_50_WHIP,
       sp.sp_50_DICE AS HT_SP_50_DICE,
       sp.sp_50_H_per_9_innings AS HT_SP_50_H_9IP,
       sp.sp_50_k_per_9_innings AS HT_SP_50_K_9IP,
       sp1.sp_100_atBat_per_pt AS HT_SP_100_AB_PT,
       sp1.sp_100_BB_per_9_innings AS HT_SP_100_BB_9IP,
       sp1.sp_Days_got_rest AS HT_SP_100_DGR,
       sp1.sp_100_inings_finished AS HT_SP_100_IF,
       sp1.sp_100_WHIP AS HT_SP_100_WHIP,
       sp1.sp_100_DICE AS HT_SP_100_DICE,
       sp1.sp_100_H_per_9_innings AS HT_SP_100_H_9IP,
       sp1.sp_100_k_per_9_innings AS HT_SP_100_K_9IP
       FROM s_team_stats
       JOIN s_team_bat_50_day_rolling_stats b
       ON s_team_stats.HT_id = b.team_id  AND s_team_stats.game_id = b.game_id
       JOIN s_team_bat_100_day_rolling_stats b1
       ON s_team_stats.HT_id = b1.team_id  AND s_team_stats.game_id = b1.game_id
       JOIN s_team_pitch_50_day_rolling_stats p
       ON s_team_stats.HT_id = p.team_id AND s_team_stats.game_id = p.game_id
       JOIN s_team_pitch_100_day_rolling_stats p1
       ON s_team_stats.HT_id = p1.team_id AND s_team_stats.game_id = p1.game_id
       JOIN s_starting_pitcher_100_rolling_stats sp1
       ON s_team_stats.HT_id = sp1.team_id AND s_team_stats.game_id = sp1.game_id
       JOIN s_starting_pitcher_50_rolling_stats sp
       ON s_team_stats.HT_id = sp.team_id AND s_team_stats.game_id = SP.game_id;


DROP TABLE IF EXISTS baseball_team_stats;
CREATE TABLE baseball_team_stats
SELECT s_home_team_stats.game_id,
       s_home_team_stats.game_date,
       s_home_team_stats.HT_id,
       s_home_team_stats.AT_id,
       s_home_team_stats.temperature,
       s_home_team_stats.wind_speed,
       s_home_team_stats.wind_direction,
       s_home_team_stats.stadium_id,
       HT_50_BABIP,
       HT_50_BA,
       HT_50_HR_per_H, HT_50_HRR, HT_50_IP,
       HT_50_OBP, HT_50_TA, HT_50_PE, HT_50_BBP,
       HT_50_K_to_BB, HT_100_BABIP, HT_100_BA,
       HT_100_HR_per_H, HT_100_HRR, HT_100_IP,
       HT_100_OBP, HT_100_TA, HT_100_PE, HT_100_BBP, HT_100_K_to_BB,
       HT_100_SBP, HT_100_SBR, HT_50_H_9IP, HT_50_BAIP, HT_50_BB_9IP,
       HT_50_DICE, HT_50_ERA, HT_50_K_9IP, HT_50_WHIP, HT_50_OBA, HT_50_K_BB,
       HT_100_H_9IP, HT_100_BAIP, HT_100_BB_9IP, HT_100_DICE, HT_100_ERA,
       HT_100_K_9IP, HT_100_WHIP, HT_100_OBA, HT_100_K_BB, HT_SP_50_AB_PT,
       HT_SP_50_BB_9IP, HT_SP_50_DGR, HT_SP_50_IF, HT_SP_50_WHIP, HT_SP_50_DICE,
       HT_SP_50_H_9IP, HT_SP_50_K_9IP, HT_SP_100_AB_PT, HT_SP_100_BB_9IP,
       HT_SP_100_DGR, HT_SP_100_IF, HT_SP_100_WHIP, HT_SP_100_DICE,
       HT_SP_100_H_9IP, HT_SP_100_K_9IP,
       b.b_t_50_BABIP AS AT_50_BABIP,
       b.b_t_50_batting_avg AS AT_50_BA,
       b.b_t_50_home_runs_per_hits AS AT_50_HR_per_H,
       b.b_t_50_HRR AS AT_50_HRR,
       b.b_t_50_Isolated_power AS AT_50_IP,
       b.b_t_50_OBP AS AT_50_OBP,
       b.b_t_50_total_avg AS AT_50_TA,
       b.t_b_50_PE AS AT_50_PE,
       b.b_t_50_BBP AS AT_50_BBP,
       b.b_t_50_walk_to_Strikeout AS AT_50_K_to_BB,
       b1.b_t_100_BABIP AS AT_100_BABIP,
       b1.b_t_100_batting_avg AS AT_100_BA,
       b1.b_t_100_home_runs_per_hits AS AT_100_HR_per_H,
       b1.b_t_100_HRR AS AT_100_HRR,
       b1.b_t_100_Isolated_power AS AT_100_IP,
       b1.b_t_100_OBP AS AT_100_OBP,
       b1.b_t_100_total_avg AT_100_TA,
       b1.t_b_100_PE AS AT_100_PE,
       b1.b_t_100_BBP AS AT_100_BBP,
       b1.b_t_100_walk_to_Strikeout AS AT_100_K_to_BB,
       b1.b_t_100_SBP AS AT_100_SBP,
       b1.b_t_100_SBR AS AT_100_SBR,
       p.t_p_50_H_per_9_innings AS AT_50_H_9IP,
       p.t_p_50_BAIP AS AT_50_BAIP,
       p.t_p_50_BB_per_9_innings AS AT_50_BB_9IP,
       p.t_p_50_DICE AS AT_50_DICE,
       p.t_p_50_ERA AS AT_50_ERA,
       p.t_p_50_k_per_9_innings AS AT_50_K_9IP,
       p.t_p_50_WHIP AS AT_50_WHIP,
       p.t_p_50_OBA AS AT_50_OBA,
       p.t_p_50_Strikeout_to_walk AS AT_50_K_BB,
       p1.t_p_100_H_per_9_innings AS AT_100_H_9IP,
       p1.t_p_100_BAIP AS AT_100_BAIP,
       p1.t_p_100_BB_per_9_innings AS AT_100_BB_9IP,
       p1.t_p_100_DICE AS AT_100_DICE,
       p1.t_p_100_ERA AS AT_100_ERA,
       p1.t_p_100_k_per_9_innings AS AT_100_K_9IP,
       p1.t_p_100_WHIP AS AT_100_WHIP,
       p1.t_p_100_OBA AS AT_100_OBA,
       p1.t_p_100_Strikeout_to_walk AS AT_100_K_BB,
       sp.sp_50_atBat_per_pt AS AT_SP_50_AB_PT,
       sp.sp_50_BB_per_9_innings AS AT_SP_50_BB_9IP,
       sp.sp_Days_got_rest AS AT_SP_50_DGR,
       sp.sp_50_inings_finished AS AT_SP_50_IF,
       sp.sp_50_WHIP AS AT_SP_50_WHIP,
       sp.sp_50_DICE AS AT_SP_50_DICE,
       sp.sp_50_H_per_9_innings AS AT_SP_50_H_9IP,
       sp.sp_50_k_per_9_innings AS AT_SP_50_K_9IP,
       sp1.sp_100_atBat_per_pt AS AT_SP_100_AB_PT,
       sp1.sp_100_BB_per_9_innings AS AT_SP_100_BB_9IP,
       sp1.sp_Days_got_rest AS AT_SP_100_DGR,
       sp1.sp_100_inings_finished AS AT_SP_100_IF,
       sp1.sp_100_WHIP AS AT_SP_100_WHIP,
       sp1.sp_100_DICE AS AT_SP_100_DICE,
       sp1.sp_100_H_per_9_innings AS AT_SP_100_H_9IP,
       sp1.sp_100_k_per_9_innings AS AT_SP_100_K_9IP,
       s_home_team_HCP.home_team_HCP AS HT_HCP,
       s_away_team_HCP.away_team_HCP AS AT_HCP,
       (HT_50_BA - b.b_t_50_batting_avg) as HT_AT_50_diff_BA,
       (HT_100_BA - b1.b_t_100_batting_avg) as HT_AT_100_diff_BA,
       (HT_50_PE - b.t_b_50_PE) as 50_PE_diff,
       (HT_100_PE - b1.t_b_100_PE) as 100_PE_diff,
       (HT_50_PE - p.t_p_50_k_per_9_innings) as PE_K_diff,
       (HT_100_PE - p.t_p_50_k_per_9_innings) as PE_K_50_diff,
       (HT_100_PE - p1.t_p_100_WHIP) as PE_WHIP_100_diff,
       (HT_50_PE - p1.t_p_100_k_per_9_innings) as PE_K_100_diff,
       s_team_stats.HT_Wins as HT_Wins
       FROM s_home_team_stats
       JOIN s_team_bat_50_day_rolling_stats b
       ON s_home_team_stats.AT_id = b.team_id  AND s_home_team_stats.game_id = b.game_id
       JOIN s_team_bat_100_day_rolling_stats b1
       ON s_home_team_stats.AT_id = b1.team_id  AND s_home_team_stats.game_id = b1.game_id
       JOIN s_team_pitch_50_day_rolling_stats p
       ON s_home_team_stats.AT_id = p.team_id AND s_home_team_stats.game_id = p.game_id
       JOIN s_team_pitch_100_day_rolling_stats p1
       ON s_home_team_stats.AT_id = p1.team_id AND s_home_team_stats.game_id = p1.game_id
       JOIN s_starting_pitcher_100_rolling_stats sp1
       ON s_home_team_stats.AT_id = sp1.team_id AND s_home_team_stats.game_id = sp1.game_id
       JOIN s_starting_pitcher_50_rolling_stats sp
       ON s_home_team_stats.AT_id = sp.team_id AND s_home_team_stats.game_id = SP.game_id
       JOIN s_home_team_HCP
       ON s_home_team_HCP.HT_id = s_home_team_stats.HT_id AND s_home_team_stats.game_id = s_home_team_HCP.game_id
       JOIN s_away_team_HCP
       ON s_away_team_HCP.AT_id = s_home_team_stats.AT_id AND s_home_team_stats.game_id = s_away_team_HCP.game_id
       JOIN s_team_stats
       ON s_team_stats.game_id = s_home_team_stats.game_id;



