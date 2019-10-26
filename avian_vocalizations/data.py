import os, wget, re
from zipfile import ZipFile
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import keras
from keras.utils import to_categorical

data_urls=[
    "https://xeno-canto-ca-nv.s3.amazonaws.com/avian-vocalizations-partitioned-data.zip",
    "https://xeno-canto-ca-nv.s3.amazonaws.com/xenocanto-avian-vocalizations-canv-usa.zip",
    "https://xeno-canto-ca-nv.s3.amazonaws.com/avian-vocalizations-spectrograms-and-mfccs.zip",
]
train_index_filename = 'train_file_ids.csv'


# https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
class AudioFeatureGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_file_ids, labels, batch_size, n_frames=128, n_channels=1, 
                 data_dir='data', shuffle=False, seed=None, n_classes=None, verbose=False):
        """ Initialize a data generator with the list of labeled `file_id`s.
        Args
            list_file_ids (list[int]): A list of `file_id`s to be included in this generator.
            labels (list[int]): A list of integer-based labels corresponding to `list_file_ids`.
            batch_size (int): Number of samples per batch.
            n_frames (int): Number of audio frames per sample (sample length). (default: 128)
            n_channels (int): Number of channels. (for compatibility with image generators) (default: 1)
            data_dir (str): Path to directory containing data downloaded from `_download_data`.
            shuffle (bool): Whether or not to shuffle the data index between batches.
            seed (int): Seed for `numpy.random.RandomState`.
            n_classes (int): Number of classes (distinct labels) in dataset. (default: `labels.max()-labels.min()+1`)
            verbose (bool): Print to stdout after generating each batch. (default: False)
        """
        self.n_frames = n_frames
        self.batch_size = batch_size
        self.n_batches = np.ceil(len(list_file_ids)/batch_size)
        self.n_classes = n_classes if n_classes else max(labels)-min(labels)+1
        self.labels_by_id = {list_file_ids[i]:l for i,l in enumerate(labels)}
        self.list_file_ids = list_file_ids
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.seed = seed
        self.verbose = verbose
        self.on_epoch_end()
        
        self.index_df, self.shapes_df, self.train_df, self.test_df = load_data(data_dir)
        self.melsg_scaler, self.melsg_log_scaler, self.mfcc_scaler = \
                                        get_scalers(self.index_df.loc[self.train_df.index])

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_file_ids) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_file_ids_temp = [self.list_file_ids[k] for k in indexes]
        X, y = self.__data_generation(list_file_ids_temp, index)
        return X, y

    def on_epoch_end(self):
        'Update indexes, to be called after each epoch'
        self.indexes = np.arange(len(self.list_file_ids))
        if self.shuffle == True:
            np.random.seed(self.seed)
            self.seed = self.seed+1 # increment the seed so we get a different batch.
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_file_ids_temp, batch_index):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        melsg_arr = np.empty((self.batch_size, 128, self.n_frames, self.n_channels))
        mfcc_arr = np.empty((self.batch_size, 20, self.n_frames, self.n_channels))
        #X = np.empty((self.batch_size, 128+20, self.n_frames, self.n_channels))
        y = np.empty((self.batch_size, self.n_classes), dtype=int) # one-hot encoded labels
        offsets = np.empty(self.batch_size)

        for i, file_id in enumerate(list_file_ids_temp):
            melsg = get_melsg_array(self.index_df, file_id)
            melsg_lognorm = self.melsg_log_scaler.transform(np.log(melsg, where=melsg>0 ))
        
            mfcc = get_mfcc_array(self.index_df, file_id)
            mfcc = self.mfcc_scaler.transform(mfcc)
            
            # Pick a random window from the sound file
            d_len = mfcc.shape[1] - self.n_frames
            if d_len<0: # Clip is shorter than window, so pad with mean value.
                n = int(np.random.uniform(0, -d_len))
                pad_range = (n, -d_len-n) # pad with n values on the left, clip_length - n values on the right 
#                 melsg_cropped = np.pad(melsg, ((0,0), pad_range), 'constant', constant_values=melsg.mean())
                melsg_lognorm_cropped = np.pad(melsg_lognorm, ((0,0), pad_range), 'constant', constant_values=0)
                mfcc_cropped = np.pad(mfcc, ((0,0), pad_range), 'constant', constant_values=0)
            else: # Clip is longer than window, so slice it up
                n = int(np.random.uniform(0, d_len))
#                 melsg_cropped = melsg[:, n:(n+self.n_frames)]
                melsg_lognorm_cropped = melsg_lognorm[:, n:(n+self.n_frames)]
                mfcc_cropped = mfcc[:, n:(n+self.n_frames)]
            offsets[i,] = n
            melsg_arr[i,] = melsg_lognorm_cropped.reshape(1,128,self.n_frames,1)
            mfcc_arr[i,] = mfcc_cropped.reshape(1,20,self.n_frames,1)
            y[i,] = to_categorical(self.labels_by_id[file_id], num_classes=self.n_classes)

#         print("Generated batch with input shapes ",(melsg_arr.shape, mfcc_arr.shape))
        if self.verbose:
            print("Generated batch #%i/%i."%(batch_index+1,self.n_batches))
            sys.stdout.flush()
        return {'melsg':melsg_arr, 
                'mfcc':mfcc_arr, 
                'id':list_file_ids_temp,
                'offset':offsets,
               }, y

class DataDirNotFound(Exception):
    """ Raised when data dir is not found. Suggests downloading by calling `load_data(download_data=True)`.
    """
    def __init__(self, data_dir, errors=None):
        message = os.path.abspath(data_dir)+" not found. \n"+\
                  "Download data by calling load_data with download_data=True"
        super().__init__(message)
        
class DataFileNotFound(Exception):
    """ Raised when expected data file could not be found. 
    """
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
        DataDirNotFound: if data_dir cannot be found.
        DataFileNotFound: if any of the required data files dannot be found. 
    """
    if not os.path.exists(data_dir):
        if download_data:
            _download_data(data_dir)
        else:
            raise(DataDirNotFound(data_dir))
            
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
            raise(DataFileNotFound(path))
            
            
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

def _download_data(data_dir='data', keep_zip=False):
    if not os.path.exists(data_dir):
        print("Creating data dir "+os.path.abspath(data_dir))
        os.mkdir(data_dir)
    for url in data_urls: _download_and_extract(url, data_dir, keep_zip=keep_zip)
    # These files should have been included in the zip archives extracted above.
    zip_file_destinations = {
        "xeno-canto-ca-nv.zip": "audio",
        "melspectrograms.zip": "",
        "mfccs.zip": "",
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
    if not keep_zip: os.remove(filename)

# def log_clipped(a):
#     """Convenience function to clip the input to positive values then return the log.""" 
#     return np.log(np.clip(a,.0000001,a.max()))

def get_scalers(index_df, data_dir='data', recalc=False):
    """
    Returns:
        tuple of `sklearn.preprocessing.StandardScaler`: (melsg_scaler, melsg_log_scaler, mfcc_scaler)
    """
    melsg_scaler = StandardScaler()
    melsg_log_scaler = StandardScaler()
    mfcc_scaler = StandardScaler()
    scaler_params=os.path.join(data_dir,'scaler_params.csv')

    if recalc or not os.path.exists(scaler_params):
        print("%s not found. Calculating scaler statistics...")
        for i, file_id in enumerate(index_df.index):
            #print("\rReading melsg %i/%i (%.1f%%)"%(i+1,len(index_df),100*(i+1)/len(index_df)), end="")
            melsg = get_melsg_array(index_df, file_id).flatten()
            melsg_scaler.partial_fit(melsg.reshape(-1,1))
            melsg_log_scaler = melsg_log_scaler.partial_fit(np.log(melsg, where=melsg>0 ).reshape(-1,1))

        for i, file_id in enumerate(index_df.index):
            #print("\rReading mfcc %i/%i (%.1f%%)"%(i+1,len(index_df),100*(i+1)/len(index_df)), end="")
            mfcc = get_mfcc_array(index_df, file_id).flatten()
            mfcc_scaler.partial_fit(mfcc.reshape(-1,1)) 

        with open(scaler_params,'w') as f:
            def write_scaler_params(name,scaler):
                f.write("%s,%i,%f,%f\n"%(name, scaler.n_samples_seen_, scaler.mean_, scaler.var_))
            f.write("dataset_name,total_pixels,mean,variance\n")
            write_scaler_params("melsg",melsg_scaler)
            write_scaler_params("melsg_log",melsg_log_scaler)
            write_scaler_params("mfcc",mfcc_scaler)
        #print("\nMean pixel data saved to %s."%scaler_params)

    else: # Load cached scaler params
        #print("Loading scaler params from %s."%scaler_params)
        scalers_df = pd.read_csv(scaler_params, index_col=0)
        def load_scaler_params(name, scaler):
            scaler.n_samples_seen_ = scalers_df.loc[name, 'total_pixels']
            scaler.mean_ = scalers_df.loc[name, 'mean']
            scaler.var_ = scalers_df.loc[name, 'variance']
            scaler.scale_ = np.sqrt(scalers_df.loc[name, 'variance'])
        load_scaler_params("melsg",melsg_scaler)
        load_scaler_params("melsg_log",melsg_log_scaler)
        load_scaler_params("mfcc",mfcc_scaler)
        
    return melsg_scaler, melsg_log_scaler, mfcc_scaler
