#!/bin/sh

ps uax | grep hyperopt-mongo-worker | grep -v grep | awk '{print $2}' | xargs kill -9
