# avian_vocalizations

Machine learning model for classifying audio recordings of avian vocalizations by species.

## Setup

### Installation

Make a new venv, then install requirements with pip.

```bash
git clone https://github.com/samhiatt/avian_vocalizations.git
cd avian_vocalizations
python3.6 -m venv .env
source .env/bin/activate
pip install -r requirements.txt
```

### Download training/test data

```python
from avian_vocalizations import data

data_dir = 'data'
data.load_data(data_dir, download_data=True)
```
or, alternatively, in a shell run:

```bash
python setup.py develop
download_data
```
This will download and extract the following datasets (from a mirror) to the path specified by `data_dir`:

* [Xeno-Canto Avian Vocalizations CA/NV, USA](https://www.kaggle.com/samhiatt/xenocanto-avian-vocalizations-canv-usa)
* [Avian Vocalizations: Partitioned Data](https://www.kaggle.com/samhiatt/avian-vocalizations-partitioned-data)
* [Avian Vocalizations: Spectrograms and MFCCs](https://www.kaggle.com/samhiatt/avian-vocalizations-spectrograms-and-mfccs)

## Files
### Library
`/avian_vocalizations/data.py` - The AudioFeatureGenerator and `download_data` method.  
`/avian_vocalizations/model.py` - The model definition.  
`/avian_vocalizations/visualization.py` - Helper methods for drawing visualizations.  
### Notebooks 
`notebooks/Data Exploration.ipynb`  
`notebooks/Data Generator and Training Example.ipynb`  
`notebooks/Benchmark Model.ipynb`  
`notebooks/Model Inference and Testing.ipynb`  
### Report
`/report/proposal.pdf` - Original project proposal.  
`/report/avian-vocalizations-report.pdf` - The final pdf report.  
