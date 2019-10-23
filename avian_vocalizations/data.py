import os, wget, re
from zipfile import ZipFile
import pandas as pd
import numpy as np

data_urls=[
    "https://xeno-canto-ca-nv.s3.amazonaws.com/avian-vocalizations-partitioned-data.zip",
    "https://xeno-canto-ca-nv.s3.amazonaws.com/xenocanto-avian-vocalizations-canv-usa.zip",
    "https://xeno-canto-ca-nv.s3.amazonaws.com/avian-vocalizations-spectrograms-and-mfccs.zip",
]
train_index_filename = 'train_file_ids.csv'

class ExceptionDataDirNotFound(Exception):
    def __init__(self, data_dir, errors=None):
        message = os.path.abspath(data_dir)+" not found. \n"+\
                  "Download data by calling load_data with download_data=True"
        super().__init__(message)
        
class ExceptionDataFileNotFound(Exception):
    def __init__(self, file_path, errors=None):
        message = file_path+" not found."
        super().__init__(message)

def load_data(data_dir='data', download_data=False ):
    """
    Args:
        data_dir (string): Relative path pointing to data dir. 
        download (bool):   Download data if not found. (Default: False)
    Returns: index_df, with cols augmented to include:
                train/test flag,
                nFrames,
                melspectrogram filename,
                mfcc filename
    Raises: 
        ExceptionDataDirNotFound: if data_dir cannot be found.
        ExceptionDataFileNotFound: if any of the required data files dannot be found. 
    """
    if not os.path.exists(data_dir):
        if download_data:
            download_data(data_dir)
        else:
            raise(ExceptionDataDirNotFound(data_dir))
            
    data_files=[
        "xeno-canto_ca-nv_index.csv",
        "feature_shapes.csv",
        "train_file_ids.csv",
        "test_file_ids.csv",
    ]
    # Make sure we have all the expected data files
    for filename in data_files:
        path = os.path.join(data_dir,filename)
        if not os.path.exists(path): 
            raise(ExceptionDataFileNotFound(path))
            
            
    index_df = pd.read_csv("data/xeno-canto_ca-nv_index.csv", 
                           index_col='file_id')
    shapes_df = pd.read_csv(os.path.join(data_dir,"feature_shapes.csv"),
                            index_col=0 )
    train_df = pd.read_csv(os.path.join(data_dir,"train_file_ids.csv"),
                           index_col=0)
    test_df = pd.read_csv(os.path.join(data_dir,"test_file_ids.csv"),
                          index_col=0)
    
    # Parse shapes (which were originally inadvertently saved as a string).
    shapes_df['n_frames'] = [_parse_shape(row['mfcc_shapes'])[1] for i,row in shapes_df.iterrows() ]
    # Add n_frames to index_df too
    index_df['n_frames'] = shapes_df.loc[index_df.index==shapes_df['file_id'],'n_frames'].values
    
    # Indicate which files belong to the test dataset. 
    index_df['test']=False
    index_df.loc[test_df.index,'test']=True
    
    # Add paths to mfcc and melspectrogram mmapped files 
    index_df['melspectrogram_path'] = [os.path.join(data_dir,'features','XC%s_melspectrogram.dat'%(file_id)) 
                                       for file_id in index_df.index]
    index_df['mfcc_path'] = [os.path.join(data_dir,'features','XC%s_mfcc.dat'%(file_id)) 
                             for file_id in index_df.index]
    
    return index_df, shapes_df, train_df, test_df

def get_mfcc_array(df, file_id):
    rec = df.loc[file_id]
    return np.memmap(rec.get('mfcc_path'), dtype='float32', mode='readonly', 
                     shape=(20,rec.get('n_frames')))

def get_melsg_array(df, file_id):
    rec = df.loc[file_id]
    return np.memmap(rec.get('melspectrogram_path'), dtype='float32', mode='readonly', 
                     shape=(128,rec.get('n_frames')))

def download_data(data_dir='data'):
    if not os.path.exists(data_dir):
        print("Creating data dir "+os.path.abspath(data_dir))
        os.mkdir(data_dir)
    for url in data_urls: _download_and_extract(url, data_dir)
    # These files should have been included in the zip archives extracted above.
    zip_file_destinations = {
        "xeno-canto-ca-nv.zip": "audio",
        "melspectrograms.zip": "features",
        "mfccs.zip": "features",
    }
    for zip_filename in zip_file_destinations:
        zip_filepath=os.path.join(data_dir, zip_filename)
        with ZipFile(zip_filepath) as zf:
            destination_dir=os.path.join(data_dir,zip_file_destinations[zip_filename])
            if not os.path.exists(destination_dir): os.mkdir(destination_dir)
            zf.extractall(destination_dir)
            print("Extracted contents of %s to %s."%(zip_filename, destination_dir))
        os.remove(zip_filepath)
        
        
def _parse_shape(shape_str):
    """Shape was saved in feature_shapes as a string. Woops.
       Convenience funtion to parse out the values. """
    if type(shape_str) is pd.Series:
        return shape_str.map(_parse_shape)
    a,b = re.search(r'\((\d+), (\d+)\)',shape_str).groups()
    return int(a), int(b)
        
def _download_and_extract(url, data_dir, keep_zip=False):
    print("Downloading and extracting "+url)
    filename = url[url.rindex('/')+1:]
    if not os.path.exists(filename): 
        wget.download(url, out=filename)
        print("Downloaded "+filename)
    else:
        print("Using existing file "+filename)
        keep_zip=True # Don't delete a pre-existing file
    with ZipFile(filename) as archive:
        for file in archive.infolist():
            archive.extract(file, data_dir)
            print("Extracted %s."%(os.path.abspath(os.path.join(data_dir,file.filename))))
    if not keep_zip: os.remove(xc_index_zip_filename)

def log_clipped(a):
    """Convenience function to clip the input to positive values then return the log.""" 
    return np.log(np.clip(a,.0000001,a.max()))
