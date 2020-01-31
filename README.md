# avian_vocalizations

Machine learning model for classifying audio recordings of avian vocalizations by species.

## Installation

`pip install git+github.com/samhiatt/avian_vocalizations.git`

or from source:

```bash
git clone https://github.com/samhiatt/avian_vocalizations.git
cd avian_vocalizations
python setup.py develop
```


## Vast.ai remote machine configuration

Running `python setup.py develop` installs vast_* console scripts used for creating and configuring new vast.ai instances.

Use the [vast.ai web console](https://vast.ai/console/create/) to select an offer, then create a new instance referencing its offer ID.
```sh
vast_create_instance <OFFER_ID>
```
Should respond with something like:
```
{"success": true, "new_contract": 465119}
Instance 465119 created.
```
In this case 465119 is the new instance id. Use it for the instance configuration commands. 

Wait for this instance to start. Once this new instance shows up in the list displayed by running `vast_running_instances`, configure the instance by running `vast_run_install_script <INSTANCE_ID>`.

```sh
vast_run_install_script 465119
```

You may monitor the progress of the install script by running `vast_monitor_install_script <INSTANCE_ID>` which will show the last 5 lines of the install log, or by logging into the remote machine over ssh and observing the contents of `/root/install.log`.
```sh
vast_monitor_install_script 465119
```
When install is complete, `vast_monitor_install_script` should show `INSTALL COMPLETE` on the last line.
```
avian_vocalizations INSTALL COMPLETE Thu Jan 30 20:42:26 UTC 2020
```

Now you can start a hyperopt worker on the remote machine by running:
```sh
vast_start_hyperopt_worker 465119
```

And the hyperopt worker may be monitored by running 
```sh
vast_monitor_hyperopt_worker 465113
```

If successful it should show something like:
```
INFO:hyperopt.mongoexp:no job found, sleeping for 4.7s
INFO:hyperopt.mongoexp:no job found, sleeping for 3.7s
INFO:hyperopt.mongoexp:no job found, sleeping for 2.8s
INFO:hyperopt.mongoexp:no job found, sleeping for 2.7s
INFO:hyperopt.mongoexp:no job found, sleeping for 4.3s
```

