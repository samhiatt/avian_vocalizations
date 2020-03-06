import os
import re
import sys
import warnings
import wget
import json
from zipfile import ZipFile
import keras
import librosa
import numpy as np
import pandas as pd
from keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

warnings.simplefilter('ignore', FutureWarning)

data_urls = [
    "https://xeno-canto-ca-nv.s3.amazonaws.com/avian-vocalizations-partitioned-data.zip",
    "https://xeno-canto-ca-nv.s3.amazonaws.com/xenocanto-avian-vocalizations-canv-usa.zip",
    "https://xeno-canto-ca-nv.s3.amazonaws.com/avian-vocalizations-spectrograms-and-mfccs.zip",
]
train_index_filename = 'train_file_ids.csv'


def get_file_path(filename, data_dir=None):
    if data_dir is None:
        data_dir = '%s/../data' % os.path.dirname(__file__)
    return "%s/%s" % (data_dir, filename)


def get_stats(data_dir=None, filename=None):
    if filename is None:
        filename = 'training_dataset_statistics.json'
    fn = get_file_path(filename, data_dir)
    with open(fn) as f:
        return json.load(f)


def get_index_df(data_dir=None, filename=None):
    if filename is None:
        filename = 'xeno-canto_ca-nv_stats.csv'
    return pd.read_csv(get_file_path(filename, data_dir), index_col='file_id')


def get_training_df(data_dir=None, train_filename='train_file_ids.csv',
                    index_filename=None):
    index_df = get_index_df(data_dir, index_filename)
    train_df = pd.read_csv(get_file_path(train_filename, data_dir))
    return index_df.loc[train_df['file_id']]


def get_label_encoder(data_dir=None, index_filename=None):
    df = get_index_df(data_dir, index_filename)
    return LabelEncoder().fit(df['english_cname'])


def scale_features(meldb, mfcc, stats_filename):
    stats = get_stats()
    mfcc_mean = np.array(stats['mfcc_mean'])
    mfcc_std = np.array(stats['mfcc_std'])
    # print(mfcc_mean.shape, mfcc.shape)
    mfcc -= mfcc_mean.reshape(-1, 1)
    # print(mfcc.shape, mfcc_mean.shape, mfcc_std.shape)
    mfcc /= mfcc_std.reshape(-1, 1)
    meldb_mean = np.array(stats['meldb_mean'])
    meldb_std = np.array(stats['meldb_std'])
    meldb -= meldb_mean.reshape(-1, 1)
    meldb /= meldb_std.reshape(-1, 1)


def preprocess(file_path, sr=44100, fmin=500, fmax=15000, hop_length=256, n_fft=2048, scale=False,
               include_meldb=True, include_mfcc=True):
    """Read mp3 file and compute audio features
    Args:
        file_path (str): path to mp3 file to load
        sr (int): sample rate
        fmin (int): minimum frequency
        fmax (int): maximum frequency
        hop_length (int): number of audio samples between adjacent STFT columns
        n_fft (int): length of the windowed signal after padding with zeros, passed to `librosa.feature.melspectrogram`
                     and `librosa.feature.mfcc`.
        scale (bool): Whether or not to scale data using dataset statistics. (default: False)
        include_meldb (bool): Whether or not to include mel-spectrogram in output. (default: True)
        include_mfcc (bool): Whether or not to include mfcc in output. (default: True)
    Returns:
        Mel-frequency spectrogram and MFCC arrays
    """
    import warnings; warnings.simplefilter('ignore')
    data, sr = librosa.load(file_path, sr=sr)
    msg_args = dict(y=data, sr=sr, n_fft=n_fft, hop_length=hop_length, fmin=fmin, fmax=fmax)
    # print("msg args:", msg_args, msg_args['y'].shape)
    msg = librosa.feature.melspectrogram(**msg_args)
    meldb = librosa.power_to_db(msg, ref=np.max)
    # print("Spectrogram shape:", msg.shape)

    mfcc = librosa.feature.mfcc(**msg_args)
    # print("MFCC shape:", mfcc.shape)

    if scale:  # Do band-wise feature scaling
        scale_features(meldb, mfcc)

    res = []
    if include_meldb:
        res.append(meldb)
    if include_mfcc:
        res.append(mfcc)
    return tuple(res)


# https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
class AudioFeatureGenerator(keras.utils.Sequence):
    """Generates data for Keras"""
    def __init__(self, list_file_ids, labels, batch_size, n_frames=128, n_channels=1,
                 data_dir='data', scale=False, index_df_filename=None, stats_filename=None,
                 shuffle=False, seed=0, n_classes=None, verbose=False,
                 include_mfcc=True, include_melsg=True):
        """ Initialize a data generator with the list of labeled `file_id`s.
        Args
            list_file_ids (list[int]): A list of `file_id`s to be included in this generator.
            labels (list[int]): A list of integer-based labels corresponding to `list_file_ids`.
            batch_size (int): Number of samples per batch.
            n_frames (int): Number of audio frames per sample (sample length). (default: 128)
            n_channels (int): Number of channels. (for compatibility with image generators) (default: 1)
            data_dir (str): Path to directory containing data downloaded from `_download_data`.
            scale (bool): Whether or not to scale the data. (default: False)
            index_df_filename (str): Csv file in `data_dir` with dataset index. If `None` then uses
                                     default defined in `get_index_df`.
            stats_filename (str): Json file in `data_dir` with training dataset statistics. If `None`
                                     then uses default defined in `get_stats`.
            shuffle (bool): Whether or not to shuffle the data index between batches.
            seed (int): Seed for `numpy.random.RandomState`.
            n_classes (int): Number of classes (distinct labels) in dataset. (default: `labels.max()-labels.min()+1`)
            verbose (bool): Print to stdout after generating each batch. (default: False)
            include_mfcc (bool): Whether or not to include mfcc in output. (default: True)
            include_melsg (bool): Whether or not to include mel-spectrogram in output. (default: True)
        """
        self.n_frames = n_frames
        self.batch_size = batch_size
        self.n_classes = n_classes if n_classes else max(labels)-min(labels)+1
        self.labels_by_id = {list_file_ids[i]: l for i, l in enumerate(labels)}
        self.list_file_ids = list_file_ids
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.seed = seed
        self.indexes = np.arange(len(self.list_file_ids))
        self.verbose = verbose
        self.on_epoch_end()
        self.data_dir = data_dir
        self.stats = get_stats(data_dir, stats_filename)
        self.scale = scale
        self.index_df = get_index_df(self.data_dir, index_df_filename)
        self.include_mfcc = include_mfcc
        self.include_melsg = include_melsg
        # self.index_df, self.shapes_df, self.train_df, self.test_df = load_data(data_dir)
        # self.melsg_scaler, self.melsg_log_scaler, self.mfcc_scaler = get_scalers(
        #     self.index_df.loc[self.train_df.index], data_dir)

    @property
    def n_batches(self):
        return len(self)

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.ceil(len(self.list_file_ids) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        if index >= len(self):
            raise IndexError("Requested batch index %i on generator with only %i batches." % (index, len(self)))
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # indexes = np.arange(self.batch_size)[index*self.batch_size:(index+1)*self.batch_size]
        list_file_ids_temp = [self.list_file_ids[k] for k in indexes]
        return self.__data_generation(list_file_ids_temp, index)

    def on_epoch_end(self):
        """Update indexes, to be called after each epoch"""
        self.seed = self.seed+self.batch_size  # increment the seed so we get a different batch.
        if self.shuffle:
            np.random.seed(self.seed)
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_file_ids_temp, batch_index):
        """Generates data containing batch_size samples"""
        x = {}
        if self.include_melsg:
            x['melsg'] = np.empty((len(list_file_ids_temp), 128, self.n_frames, self.n_channels))
        if self.include_mfcc:
            x['mfcc'] = np.empty((len(list_file_ids_temp), 20, self.n_frames, self.n_channels))
        y = np.empty((len(list_file_ids_temp), self.n_classes), dtype=int)  # one-hot encoded labels
        offsets = np.empty(len(list_file_ids_temp))

        for i, file_id in enumerate(list_file_ids_temp):
            data_length = int(self.index_df.loc[file_id]['data_length'])
            # Pick a random window from the sound file
            np.random.seed(self.seed + i)
            offset = int(np.random.uniform(0, data_length))
            offsets[i, ] = offset
            y[i,] = to_categorical(self.labels_by_id[file_id], num_classes=self.n_classes)

            if self.include_melsg:
                meldb = get_melsg_array(self.index_df, file_id)
                meldb_cropped = meldb[:, offset:offset + self.n_frames]
            if self.include_mfcc:
                mfcc = get_mfcc_array(self.index_df, file_id)
                mfcc_cropped = mfcc[:, offset:offset + self.n_frames]

            for j in range(int(np.ceil(self.n_frames / data_length ))):
                if self.include_melsg:
                    meldb_cropped = np.concatenate([meldb_cropped, meldb], axis=1)[:, :self.n_frames]
                if self.include_mfcc:
                    mfcc_cropped = np.concatenate([mfcc_cropped, mfcc], axis=1)[:, :self.n_frames]

            if self.include_melsg:
                x['melsg'][i, ] = meldb_cropped.reshape((1, 128, self.n_frames, 1))
            if self.include_mfcc:
                x['mfcc'][i, ] = mfcc_cropped.reshape((1, 20, self.n_frames, 1))

        if self.verbose:
            print("Generated batch #%i/%i." % (batch_index+1, self.n_batches))
            sys.stdout.flush()
        return {**x,
                'id': list_file_ids_temp,
                'offset': offsets,
                }, y


class DataDirNotFound(Exception):
    """ Raised when data dir is not found. Suggests downloading by calling `load_data(download_data=True)`.
    """
    def __init__(self, data_dir, errors=None):
        message = os.path.abspath(data_dir)+" not found. \n" + str(errors) +\
                  "Download data by calling load_data with download_data=True"
        super().__init__(message)


class DataFileNotFound(Exception):
    """ Raised when expected data file could not be found. 
    """
    def __init__(self, file_path, errors=None):
        message = file_path+" not found."+str(errors)
        super().__init__(message)


def load_data(data_dir='data', download_data=False):
    """
    Args:
        data_dir (string): Relative path pointing to data dir. 
        download_data (bool):   Download data if not found. (Default: False)
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

    data_files = [
        "xeno-canto_ca-nv_index.csv",
        "feature_shapes.csv",
        "train_file_ids.csv",
        "test_file_ids.csv",
    ]
    # Make sure we have all the expected data files
    for filename in data_files:
        path = os.path.join(data_dir, filename)
        if not os.path.exists(path):
            raise(DataFileNotFound(path))

    index_df = pd.read_csv(os.path.join(data_dir, "xeno-canto_ca-nv_index.csv"), index_col='file_id')
    shapes_df = pd.read_csv(os.path.join(data_dir, "feature_shapes.csv"), index_col=0)
    train_df = pd.read_csv(os.path.join(data_dir, "train_file_ids.csv"), index_col=0)
    test_df = pd.read_csv(os.path.join(data_dir, "test_file_ids.csv"), index_col=0)

    # Parse shapes (which were originally inadvertently saved as a string).
    shapes_df['n_frames'] = [_parse_shape(row['mfcc_shapes'])[1] for i, row in shapes_df.iterrows()]
    # Add n_frames to index_df too
    index_df['n_frames'] = shapes_df.loc[index_df.index == shapes_df['file_id'], 'n_frames'].values

    # Indicate which files belong to the test dataset. 
    index_df['test'] = False
    index_df.loc[test_df.index, 'test'] = True

    # Add paths to mfcc and melspectrogram mmapped files 
    index_df['melspectrogram_path'] = [os.path.join(data_dir, 'features', 'XC%s_melspectrogram.dat' % file_id)
                                       for file_id in index_df.index]
    index_df['mfcc_path'] = [os.path.join(data_dir, 'features', 'XC%s_mfcc.dat' % file_id)
                             for file_id in index_df.index]

    return index_df, shapes_df, train_df, test_df


def get_mfcc_array(df, file_id):
    rec = df.loc[file_id]
    shape = (20, int(rec['feature_length']))
    return np.memmap('../data/%s' % (rec['mfcc_path']), dtype='float32', mode='readonly',
                     shape=shape)


def get_melsg_array(df, file_id):
    rec = df.loc[file_id]
    shape = (128, int(rec['feature_length']))
    return np.memmap('../data/%s' % (rec['melsg_path']), dtype='float32', mode='readonly',
                     shape=shape)


def _download_data(data_dir='data', keep_zip=True):
    if not os.path.exists(data_dir):
        print("Creating data dir "+os.path.abspath(data_dir))
        os.mkdir(data_dir)
    for url in data_urls:
        _download_and_extract(url, data_dir, keep_zip=keep_zip)
    # These files should have been included in the zip archives extracted above.
    zip_file_destinations = {
        "xeno-canto-ca-nv.zip": "audio",
        "melspectrograms.zip": "",
        "mfccs.zip": "",
    }
    for zip_filename in zip_file_destinations:
        zip_filepath = os.path.join(data_dir, zip_filename)
        with ZipFile(zip_filepath) as zf:
            destination_dir = os.path.join(data_dir, zip_file_destinations[zip_filename])
            if not os.path.exists(destination_dir):
                os.mkdir(destination_dir)
            zf.extractall(destination_dir)
            print("Extracted contents of %s to %s." % (zip_filename, destination_dir))
        os.remove(zip_filepath)


def _parse_shape(shape_str):
    """Shape was saved in feature_shapes as a string. Woops.
       Convenience funtion to parse out the values. """
    if type(shape_str) is pd.Series:
        return shape_str.map(_parse_shape)
    a, b = re.search(r'\((\d+), (\d+)\)', shape_str).groups()
    return int(a), int(b)


def _download_and_extract(url, data_dir, keep_zip=False):
    print("Downloading and extracting "+url)
    filename = url[url.rindex('/')+1:]
    if not os.path.exists(filename):
        wget.download(url, out=filename)
        print("Downloaded "+filename)
    else:
        print("Using existing file "+filename)
        keep_zip = True  # Don't delete a pre-existing file
    with ZipFile(filename) as archive:
        for file in archive.infolist():
            archive.extract(file, data_dir)
            print("Extracted %s." % (os.path.abspath(os.path.join(data_dir, file.filename))))
    if not keep_zip:
        os.remove(filename)

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
    scaler_params = os.path.join(data_dir, 'scaler_params.csv')

    if recalc or not os.path.exists(scaler_params):
        print("%s not found. Calculating scaler statistics..." % scaler_params)
        for i, file_id in enumerate(index_df.index):
            # print("\rReading melsg %i/%i (%.1f%%)"%(i+1,len(index_df),100*(i+1)/len(index_df)), end="")
            melsg = get_melsg_array(index_df, file_id).flatten()
            melsg_scaler.partial_fit(melsg.reshape(-1, 1))
            melsg_log_scaler = melsg_log_scaler.partial_fit(np.log(melsg,
                                                                   out=np.zeros(melsg.shape),
                                                                   where=melsg > 0).reshape(-1, 1))

        for i, file_id in enumerate(index_df.index):
            # print("\rReading mfcc %i/%i (%.1f%%)"%(i+1,len(index_df),100*(i+1)/len(index_df)), end="")
            mfcc = get_mfcc_array(index_df, file_id).flatten()
            mfcc_scaler.partial_fit(mfcc.reshape(-1, 1))

        with open(scaler_params, 'w') as f:
            def write_scaler_params(name, scaler):
                f.write("%s,%i,%f,%f\n" % (name, scaler.n_samples_seen_, scaler.mean_, scaler.var_))
            f.write("dataset_name,total_pixels,mean,variance\n")
            write_scaler_params("melsg", melsg_scaler)
            write_scaler_params("melsg_log", melsg_log_scaler)
            write_scaler_params("mfcc", mfcc_scaler)
        # print("\nMean pixel data saved to %s."%scaler_params)

    else:  # Load cached scaler params
        # print("Loading scaler params from %s."%scaler_params)
        scalers_df = pd.read_csv(scaler_params, index_col=0)

        def load_scaler_params(name, scaler):
            scaler.n_samples_seen_ = scalers_df.loc[name, 'total_pixels']
            scaler.mean_ = scalers_df.loc[name, 'mean']
            scaler.var_ = scalers_df.loc[name, 'variance']
            scaler.scale_ = np.sqrt(scalers_df.loc[name, 'variance'])
        load_scaler_params("melsg", melsg_scaler)
        load_scaler_params("melsg_log", melsg_log_scaler)
        load_scaler_params("mfcc", mfcc_scaler)

    return melsg_scaler, melsg_log_scaler, mfcc_scaler
