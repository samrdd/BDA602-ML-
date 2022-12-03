#!/bin/bash

sleep 10 # giving some time for mariadb setup

RESULT=`mariadb -u root -ppassword -h mariadb --skip-column-names -e "SHOW DATABASES like 'baseball'"`

if [ "$RESULT" == "baseball" ];
then
    echo "Database exist"
    echo "100 days rolling average of batters in game 12560"
    mariadb -u root -ppassword -h mariadb baseball < Assignment6.sql
    mariadb -u root -ppassword -h mariadb baseball < Assignment6.sql > mystuff/file_output.csv

else
    echo "Database does not exist, Creating database"
    mariadb -u root -ppassword -h mariadb -e "CREATE DATABASE baseball;"
    echo "Created a database baseball"
    mariadb -u root -ppassword -h mariadb baseball < baseball.sql
    echo "Dumped baseball file to database baseball"
    echo "100 days rolling average of batters in game 12560"
    mariadb -u root -ppassword -h mariadb baseball < Assignment6.sql
    mariadb -u root -ppassword -h mariadb baseball < Assignment6.sql > mystuff/file_output.csv

fi

