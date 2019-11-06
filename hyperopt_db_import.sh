#!/bin/sh

mongoimport -d avian_vocalizations -c jobs < db_backup/jobs.json 
mongoimport -d avian_vocalizations -c fs.chunks < db_backup/fs.chunks.json 
mongoimport -d avian_vocalizations -c fs.files < db_backup/fs.files.json 
mongoimport -d avian_vocalizations -c job_ids < db_backup/job_ids.json 
