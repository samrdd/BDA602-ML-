#!/bin/bash

sleep 10 # giving some time for mariadb setup

RESULT=`mariadb -u root -ppassword -h mariadb1 --skip-column-names -e "SHOW DATABASES like 'baseball'"`

if [ "$RESULT" == "baseball" ];
then
    echo "Database exist"
    echo "features created with baseball stats are loading, it may take upto 4 min "
    mariadb -u root -ppassword -h mariadb1 baseball < finalproject.sql
    echo "Data is Ready, Time for some Analytics"

else
    echo "Database does not exist, Creating database"
    mariadb -u root -ppassword -h mariadb1 -e "CREATE DATABASE baseball;"
    echo "Created a database baseball"
    mariadb -u root -ppassword -h mariadb1 baseball < baseball.sql
    echo "Dumped baseball file to database baseball"
    echo "features created with baseball stats are loading, it may take upto 4 min "
    mariadb -u root -ppassword -h mariadb1 baseball < finalproject.sql
    echo "Data is Ready, Time for some Analytics"

fi

sleep 5

python3 finalproject.py
