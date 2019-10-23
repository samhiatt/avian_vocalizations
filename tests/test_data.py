import pytest
from avian_vocalizations import data #.data import load_data, DataDirNotFound, get_melsg_array, get_scaler_data
from pandas import DataFrame
import numpy as np

def test_load_data(fs):
    assert pytest.raises(data.DataDirNotFound, data.load_data), "Shouldn't find data dir since pyfakefs fixture is loaded."
    
@pytest.fixture
def data_frames():
    index_df, shapes_df, train_df, test_df = data.load_data()
    assert type(index_df) is DataFrame
    assert type(shapes_df) is DataFrame
    assert type(train_df) is DataFrame
    assert type(test_df) is DataFrame
    return (index_df, shapes_df, train_df, test_df)

@pytest.fixture
def scalers(data_frames):
    index_df, shapes_df, train_df, test_df = data_frames
    return data.get_scalers(train_df)

def test_scalers(scalers):
    melsg_scaler, melsg_log_scaler, mfcc_scaler = scalers
    assert type(melsg_scaler.mean_) is np.float64
    assert type(melsg_scaler.scale_) is np.float64
    assert type(melsg_log_scaler.mean_) is np.float64
    assert type(melsg_log_scaler.scale_) is np.float64
    assert type(mfcc_scaler.mean_) is np.float64
    assert type(mfcc_scaler.scale_) is np.float64
    
def test_get_melsg_array(data_frames, scalers):
    index_df, shapes_df, train_df, test_df = data_frames
    melsg_scaler, melsg_log_scaler, mfcc_scaler = scalers
    file_id = index_df.head(1).index[0]
    melsg = data.get_melsg_array(index_df, file_id)
    assert melsg.shape[0]==128, "Should be 128 px tall."
    print(melsg.mean())
    assert np.abs(melsg_scaler.mean_ - melsg.mean()) < np.abs(melsg_scaler.scale_),\
        "melsg.mean() should be within one standard deviation of the dataset mean pixel."
    
def test_get_mfcc_array(data_frames, scalers):
    index_df, _, _, _ = data_frames
    _, _, mfcc_scaler = scalers
    file_id = index_df.head(1).index[0]
    mfcc = data.get_mfcc_array(index_df, file_id)
    assert mfcc.shape[0]==20, "Should be 20 px tall."
    print(mfcc.mean())
    assert np.abs(mfcc_scaler.mean_ - mfcc.mean()) < np.abs(mfcc_scaler.scale_),\
        "mfcc.mean() should be within one standard deviation of the dataset mean pixel."