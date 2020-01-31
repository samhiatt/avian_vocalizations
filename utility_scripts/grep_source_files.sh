#!/bin/sh

USAGE="Searches for PATTERN in the files found by find_source_files.sh.\nUSAGE: $0 [-i] <PATTERN>"

if [ -z $1 ]; then
    echo $USAGE
    exit 1
fi
if [ $1=='-i' ]; then
    if [ -z $2 ]; then 
        echo $USAGE
        exit 1
    fi
    IGNORE='-i '
    PATTERN=$2
else
    IGNORE=''
    PATTERN=$1
fi
grep $IGNORE $PATTERN `./find_source_files.sh` 2>/dev/null | cut -c1-`tput cols`
