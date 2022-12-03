#!/bin/bash


if mariadb -u root -ppassword -h mariadb1 -e "USE baseball;"
then
    echo "Database exist"
    mariadb -u root -ppassword -h mariadb1 baseball < Assignment6.sql
    mariadb -u root -ppassword -h mariadb1 baseball < Assignment6.sql > mystuff/file_output.csv

else
    echo "Database does not exist, Creating database"
    mariadb -u root -ppassword -h mariadb1 -e "CREATE DATABASE baseball;"
    mariadb -u root -ppassword -h mariadb1 baseball < baseball.sql
    mariadb -u root -ppassword -h mariadb1 baseball < Assignment6.sql
    mariadb -u root -ppassword -h mariadb1 baseball < Assignment6.sql > mystuff/file_output.csv

fi

