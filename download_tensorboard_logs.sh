#!bin/sh

scp -r -P 15882 -i /home/sam/.ssh/vastai root@ssh5.vast.ai:/root/avian_vocalizations/tensorboard .
#rsync -a -e "ssh -p 15882 -i /home/sam/.ssh/vastai" root@ssh5.vast.ai:/root/avian_vocalizations/tensorboard .
