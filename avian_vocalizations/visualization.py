import os
import librosa
from librosa.display import specshow
from IPython.display import Audio
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import matplotlib.pyplot as plt
import numpy as np

def show_sample(melsg, mfcc, file_id=None, label="", offset=0, data_dir='data', load_clip=False):
    fig = plt.figure(figsize=(7,5))
    if file_id or label!="":
        fig.suptitle(' '.join([ ("XC%s"%file_id) if file_id else "", 
                         label
                       ]))
    gs = GridSpec(4, 1, fig, hspace=.1, wspace=0, top=.93)
    melsg_ax = fig.add_subplot(gs[0:3])
    specshow(melsg.squeeze(), y_axis='mel', vmin=-3, vmax=3, ax=melsg_ax)
    plt.colorbar(melsg_ax.collections[0], ax=melsg_ax, pad=.01)
    mfcc_ax = fig.add_subplot(gs[3])
    specshow(mfcc.squeeze(), ax=mfcc_ax, x_axis='s')
    mfcc_ax.set_ylabel("MFCC")
    mfcc_ax.set_yticks([0,5,10,15])
    # TODO: Ensure 22050 is correct frame rate
    mfcc_ax.set_xticklabels(["%0.1f"%(t+offset/(22050/512)) 
                             for t in mfcc_ax.get_xticks()])
    plt.colorbar(mfcc_ax.collections[0], ax=mfcc_ax, aspect=7, pad=.01)
    plt.show()
    if file_id and load_clip:
        file_path = os.path.join(data_dir,'audio',"XC%s.mp3"%file_id)
        print(file_path)
        import warnings; warnings.simplefilter('ignore')
        data, samplerate = librosa.load(file_path)
        display(Audio(data, rate=samplerate))
    
    
def vis_sample_with_histograms(sample, data_dir='data'):
    data, sr = librosa.load(os.path.join(data_dir,'audio',sample.file_name))
    Audio(data, rate = sr)
    hop_length = 512
    fmin = 0
    melsg = librosa.feature.melspectrogram(data, sr=sr, hop_length=hop_length, n_fft=2048, fmin=fmin)
    mfcc = librosa.feature.mfcc(data, sr=sr, hop_length=hop_length, n_fft=2048, fmin=fmin)
    fig = plt.figure(figsize=(15,5)); #plt.subplots_adjust(hspace=4)
    cols = GridSpec(1,2, fig) # get a grid of 1 row x 2 cols
    left_col = GridSpecFromSubplotSpec(3,1,subplot_spec=cols[0], hspace=1)
    right_col = GridSpecFromSubplotSpec(5,1,subplot_spec=cols[1], hspace=3)
    fig.suptitle("%s: %s"%(sample.file_name, sample.full_name))
    hist_ax1 = fig.add_subplot(left_col[0], xlabel="Power",
                               title="Histogram of Melspectrogram Pixels")
    hist_ax2 = fig.add_subplot(left_col[1], xlabel="log(Power)",
                               title="Histogram of log(Melspectrogram)")
    mfcc_hist_ax = fig.add_subplot(left_col[2], xlabel="MFCC",
                               title="Histogram of MFCC Pixels")
    melsg_ax = fig.add_subplot(right_col[:3])
    mfcc_ax = fig.add_subplot(right_col[3:])
    hist_ax1.hist(melsg.flatten(), bins=100)
    hist_ax2.hist(np.log(melsg.flatten()), bins=100)
    mfcc_hist_ax.hist(mfcc.flatten(), bins=100)
    specshow(np.log(melsg), y_axis='mel', x_axis='s', ax=melsg_ax, fmin=fmin,
             hop_length=hop_length, sr=sr)
#     specshow(librosa.amplitude_to_db(melsg[50:,:], ref=np.max), y_axis='mel', x_axis='s', ax=melsg_ax, 
#              hop_length=hop_length, sr=sr, fmin=1376)
    melsg_ax.set_title("log(Melspectrogram)")
    plt.colorbar(melsg_ax.get_children()[0], ax=melsg_ax)
    specshow(mfcc, x_axis='s', ax=mfcc_ax, fmin=fmin, 
             hop_length=hop_length, sr=sr)
    mfcc_ax.set_title("Mel-frequency Cepstral Coefficients")
    plt.colorbar(mfcc_ax.get_children()[0], ax=mfcc_ax);
    display(fig)
    plt.close('all')
