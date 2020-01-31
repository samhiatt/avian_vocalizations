#!/usr/bin/env python
from vastai.api import VastClient
from argparse import ArgumentParser
import sys
from glob import glob

def main(args):
    print(args)
    if not args.all and not args.instances:
        print("Specify instance(s) with -i or use -a for all running instances.")
        sys.exit(1)
    
    vast = VastClient().authenticate()
    running_instances = vast.get_running_instances()
    if args.all: 
        instances = running_instances
    else:
        instances = [ i for i in running_instances if i.id in args.instances ]
        
    if len(instances)==0:
        print("No instance found. Is it running?")
        sys.exit(1)
        
    for instance in instances:
        print("Hot-updating instance %s"%instance.id)
        for file in args.files:
            for fn in glob(file):
                print("Pushing "+fn)
                instance.pb_remote.sftp.put(fn, '/root/avian_vocalizations/'+fn)
    

if __name__=='__main__':
    parser = ArgumentParser(description="Hot update running instance with working tree.")
    parser.add_argument('files',nargs='*',help="Files to hot-update.")
    parser.add_argument('-a','--all',dest='all',action='store_true',help="Hot update all running machines.")
    parser.add_argument('-i','--instance',dest='instances',action='append',help="Hot update instances indicated by id.")
    args = parser.parse_args()
    main(args)