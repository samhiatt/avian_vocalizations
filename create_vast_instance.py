#!/usr/bin/env python
from argparse import ArgumentParser
import sys
from vastai.api import VastClient
# from io import StringIO

def create_instance(vast, offer_id, storage=15):
    onstart = """#!/bin/sh
    touch ~/.no_auto_tmux
    echo onstart.sh Starting `date` >> /root/startup.log 2>&1
    apt-get install -y mongodb-server git vim >> /root/startup.log 2>&1
    service mongodb start >> /root/startup.log 2>&1
    pip install --upgrade pip >> /root/startup.log 2>&1
    echo onstart.sh Completed at `date` >> /root/startup.log
    """
    resp=vast.create_instance(offer_id, disk=storage, onstart_cmd=onstart, python_utf8=True, lang_utf8=True,
                              image="tensorflow/tensorflow:1.15.2-gpu-py3-jupyter")
    instance_id = resp['new_contract']

    return vast.get_instance(instance_id, retries=4)

def filter_offers(vast, args):
    filters = 'dph<%f inet_down>%f'%(args.max_dph, args.min_inet_down)
    return vast.search_offers('dph', filters, instance_type='on-demand')

def main(args):
    print(args)

    vast = VastClient().authenticate()
    
    attrs = ['id','dph_total','gpu_name','inet_up','inet_down']
    line_format = "%10s  %-10s %15s %10s %10s"
    
    def _get_filtered_offers():
        filtered_offers = filter_offers(vast, args)
        print(line_format%(*attrs,))
        for offer in filtered_offers:
            print(line_format%(*[offer[a] for a in attrs],))
        return filtered_offers
        
    if args.offer_id:
        offer_id = args.offer_id
    elif args.first_offer:
        offer_id = _get_filtered_offers()[0]['id']
    else:
        filtered_offers = _get_filtered_offers()
        answer = input("Which offer? (default %s) "%filtered_offers[0]['id']).strip()
        if answer == '':
            offer_id = filtered_offers[0]['id']
        else:
            #offers = list(filter(lambda x: str(x['id'])==answer, filtered_offers))
            #if len(offers)<1:
            #    print("Offer %s not found."%answer)
            #    sys.exit(1)
            #offer_id = offers[0]['id']
            offer_id = int(answer)
    print("Creating instance from offer %s"%offer_id)
    create_instance(vast, offer_id, storage=args.storage)

if __name__=='__main__':
    parser = ArgumentParser(description="Create a new vast.ai instance.")
    parser.add_argument('-o','--offer', dest='offer_id', type=int, default=None)
    parser.add_argument('-p','--dph', dest='max_dph', type=float, default=.2, 
                        help="Maximum $/hr")
    parser.add_argument('-d','--inet_down', dest='min_inet_down', type=float, default=100, 
                        help="Minimum inet down speed.")
    parser.add_argument('-u','--inet_up', dest='min_inet_up', type=float, default=25,
                        help="Minimum inet up speed.")
    parser.add_argument('-s','--storage', dest='storage', type=float, default=15,
                        help="Storage space, in GB (default: 15)")
    parser.add_argument('-y', dest='first_offer', action='store_true', default=False,
                        help="Automatically select the first offer.")
    args=parser.parse_args()
    main(args)