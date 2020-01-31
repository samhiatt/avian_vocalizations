from vastai.api import VastClient
from io import StringIO
import sys
from IPython.display import clear_output

vast = VastClient().authenticate()

def show_running_instances():
    for instance in vast.get_running_instances():
        print()
        print("Instance %i:"%instance.id)
        print(instance.ssh_connection_command)
        
def create_instance():
    if len(sys.argv)<2:
        print("Expected offer id argument.")
        sys.exit(1)
    offer_id = int(sys.argv[1])
    onstart = """#!/bin/sh
    touch ~/.no_auto_tmux
    echo onstart.sh Starting `date` >> /root/startup.log 2>&1
    apt-get install -y mongodb-server git vim >> /root/startup.log 2>&1
    service mongodb start >> /root/startup.log 2>&1
    pip install --upgrade pip >> /root/startup.log 2>&1
    echo onstart.sh Completed at `date` >> /root/startup.log
    """
    resp=vast.create_instance(offer_id, disk=15, onstart_cmd=onstart, python_utf8=True, lang_utf8=True,
                              image="tensorflow/tensorflow:1.15.2-gpu-py3-jupyter")
    instance_id = resp['new_contract']
    print("Instance %i created."%instance_id) 
        
#def _get_instances():
#    if len(sys.argv)<2:
#        print("Expected instance id argument or '-a' for all running instances.")
#        sys.exit(1)
#    if sys.argv[1].strip()=='-a':
#        instances = vast.get_running_instances()
#    else:
#        instances = [ vast.get_instance(int(sys.argv[1])) ]
#    return instances

def _get_instance():
    if len(sys.argv)<2:
        print("Expected instance id argument.")
        sys.exit(1)
    inst_id = int(sys.argv[1])
    return vast.get_instance(inst_id)

def _monitor_file(remote, log_file, popen=None, n_lines=5, update_freq_s=1):
    while True:
        try:
            clear_output(wait=True)
            print(remote['tail'](log_file,'-n%i'%n_lines))
            if popen is not None:
                ret = popen.proc.poll()
                if ret is not None:
                    print("Process has exited with code:",popen.proc.poll())
                    break
            time.sleep(update_freq_s)
        except KeyboardInterrupt:
            break
        except NameError:
            break
            
def monitor_hyperopt_worker():
    instance = _get_instance()
    _monitor_file(instance.pb_remote, '/root/hyperopt_worker.log')

def monitor_install_script():
    instance = _get_instance()
    _monitor_file(instance.pb_remote, '/root/install.log')
    

def start_remote_hyperopt_worker():
    hyperopt_worker_log="/root/hyperopt_worker.log"
    start_hyperopt_worker_target='/root/avian_vocalizations/start_hyperopt_worker.sh'
    start_worker_script_str="#!/bin/sh\ncd /root/avian_vocalizations\n"\
        "export LC_ALL=C.UTF-8\n"\
        "export LANG=C.UTF-8\n"\
        "hyperopt-mongo-worker --mongo=localhost:27017/avian_vocalizations "\
        " --max-consecutive-failures=1 --reserve-timeout=999999 --workdir=."\
        " >> %s 2>&1\n"%hyperopt_worker_log
    instance = _get_instance().wait_until_running()
    remote = instance.pb_remote
    remote.sftp.putfo(StringIO(start_worker_script_str),start_hyperopt_worker_target)
    remote['chmod']('+x',start_hyperopt_worker_target)
    pworker = remote.session().popen(start_hyperopt_worker_target)
    _monitor_file(remote, hyperopt_worker_log, pworker)
    
        
install_script_str = StringIO("""#!/bin/sh
INSTALL_LOG=/root/install.log
echo avian_vocalizations INSTALL BEGIN `date` >> $INSTALL_LOG 2>&1
cd ~
[ ! -d avian_vocalizations ] && git clone https://github.com/samhiatt/avian_vocalizations.git  >> $INSTALL_LOG 2>&1
cd avian_vocalizations
export LC_ALL=C.UTF-8
export LANG=C.UTF-8
#pip install pipenv  >> $INSTALL_LOG 2>&1
#pipenv lock -r > requirements.txt
pip install -r requirements.txt >> $INSTALL_LOG 2>&1
pip install -e . >> $INSTALL_LOG 2>&1
download_data >> $INSTALL_LOG 2>&1
echo avian_vocalizations INSTALL COMPLETE `date` >> $INSTALL_LOG 2>&1
""")
# ("""pipenv install >> $INSTALL_LOG 2>&1
# pipenv run python setup.py develop >> $INSTALL_LOG 2>&1
# pipenv run download_data >> $INSTALL_LOG 2>&1
# echo avian_vocalizations INSTALL COMPLETE `date` >> INSTALL_LOG 2>&1
# """)
# monitor_output(mongo_instance, install_script)
        
def run_install_script():
    #for instance in _get_instances():
    instance = _get_instance().wait_until_running()
    remote = instance.pb_remote
    install_script_target='/root/install_script.sh'
    remote.sftp.putfo(install_script_str,install_script_target)
    remote['chmod']('+x',install_script_target)
    p=remote.session().popen(install_script_target)
