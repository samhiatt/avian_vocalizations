#!/bin/bash

if [ "$(which inotifywait)" == "" ]; then 
  echo "inotifywait not installed. Enter admin password to install with 'sudo apt-get install'";
  sudo apt-get install inotify-tools; 
else
  echo "inotifywait is installed.";
fi

while inotifywait -e modify $1
  do ./create_pdf.sh $1
done
