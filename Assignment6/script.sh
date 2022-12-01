#!/bin/bash

RESULT=`mariadb -u root -ppassword --skip-column-names -e "SHOW DATABASES LIKE 'baseball'"`

if [ "$RESULT" == "baseball" ]; then
    echo "Database exist"
else
    echo "Database does not exist, loading database.."
mariadb -u root -ppassword -h mariadb3 -e "CREATE DATABASE baseball;"
mariadb -u root -ppassword -h mariadb3 -e baseball < baseball.sql

mariadb -u root -ppassword -h mariadb3 -e baseball < Assignment.sql

mariadb -u root -ppassword -h mariadb1 -e baseball < Assignment.sql > mystuff/file_output

fi