#!/bin/sh

mongoexport --port 27018 -d avian_vocalizations -c jobs -o db_backup/jobs.json 
mongoexport --port 27018 -d avian_vocalizations -c fs.chunks -o db_backup/fs.chunks.json 
mongoexport --port 27018 -d avian_vocalizations -c fs.files -o db_backup/fs.files.json 
mongoexport --port 27018 -d avian_vocalizations -c job_ids -o db_backup/job_ids.json 
