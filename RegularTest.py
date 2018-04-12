import numpy as np
import utils
import csv
import os
import librosa
import argparse
from scipy.io import wavfile
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import autoencoder

parser = argparse.ArgumentParser()
parser.add_argument('--test_directory', type=str,
						default='spectograms/noise/test/', help="path for the data")
parser.add_argument('--noise_type', type=str,
						default='babble/', help="noise type") 
parser.add_argument('--noise_snr', type=str,
						default='-6', help="noise snr to test at") 
parser.add_argument('--model_directory', type=str,
						default='models/normal_train/', help="path where model weight lies")
parser.add_argument('--model_name', type=str,
						default='noise_-6db/', help="name of actual model weights")
parser.add_argument('--save_audio', type=bool,
						default=False, help="if u want to save the audio files")
parser.add_argument('--window_size', type=int,
						default=5, help="size of window on each side")
parser.add_argument('--exp_name', type=str,
						default='test', help="name of your experiment")

 
args = parser.parse_args()
test_directory = args.test_directory
noise_type = args.noise_type
noise_snr = args.noise_snr
model_directory = args.model_directory
model_name = args.model_name
save_audio = args.save_audio
window_size = args.window_size
exp_name = args.exp_name

model_load_path = model_directory + model_name + 'model_auto.h5'
test_load_path = test_directory + noise_type
print('Model Name ')
print(model_load_path + '\n')
print('Test Directory ')
print(test_load_path + '\n')
print('noise type test')
print(noise_type + '\n')
print('noise snr test')
print(noise_snr + '\n')
print('experiment name')
print(exp_name + '\n')

total_test = 1680
noise_spect_name = 'spect_' + noise_snr + '.npy'
print('noise spectogram name')
print(noise_spect_name + '\n')
frame_size = 11
clean_spect_name = 'spect_clean.npy'
print('clean spectogram name')
print(clean_spect_name + '\n')
audio_clean_name = 'audio_clean.WAV'
audio_noise_name = 'audio_' + noise_snr + '.WAV'


model = autoencoder.Auto(1771, 161)
state_dict = torch.load(model_load_path, map_location=lambda storage, loc: storage)
model.load_state_dict(state_dict['state_dict'])

if torch.cuda.is_available():
    print('cuda available....')
    model.cuda()
    
model.eval()

MSE = []
PESQ = []
STOI = []
with open(test_load_path + noise_snr + '_' + exp_name + '.txt','a') as f:
    f.write(model_load_path + '\n' + test_load_path + '\n')
   


for idx in range(total_test):
    print('Testing File: %d' % idx)
   
   #path where the folder is
    test_file_path = test_load_path + str(idx) + '/' 

    #get the clean magnitudes and the noise magnitude at the specific SNR
    clean_mag = np.load(test_file_path + clean_spect_name)
    noise_mag = np.load(test_file_path + noise_spect_name)

    #Get the actual clean audio and noise audio
    full_audio_clean_name = test_file_path + audio_clean_name
    full_audio_noise_name = test_file_path + audio_noise_name
    print(full_audio_clean_name)
    print(full_audio_noise_name)
    [clean_audio,fs] = librosa.load(full_audio_clean_name,16000)
    [noise_audio,fs] = librosa.load(full_audio_noise_name,16000)

    #PASS NOISE_MAG into function that returns APPROX_CLEAN_MAG and MMSE

    ###some function ri2ght here
  

    ### WOULD PASS THROUGH approx_clean_mag into reconstruct.
    ### RETURNS Approx_clean_audio
    ### this is just for testing
    approx_clean_audio = utils.reconstruct_clean(noise_audio, clean_mag[:,window_size:-1 * window_size])
    approx_clean_name =test_file_path + 'approx_clean_' + noise_snr + '_' + exp_name + '.WAV'
    if save_audio:
        print('saving audio...')
        wavfile.write(approx_clean_name, 16000,approx_clean_audio)
        #np.save(test_file_path + 'approx_clean_mag_' + noise_snr + '_' + exp_name + '.npy', approx_clean_mag)
        with open(test_load_path + noise_snr + '_' + exp_name + '.txt','a') as f:
            f.write(full_audio_clean_name + '\t' + full_audio_noise_name + '\n')
            f.write(full_audio_clean_name + '\t' + approx_clean_name + '\n' )


    if idx == 3:
        break

    ###pass approx_clean_audio,clean_audio,noise_audio,clean_mag,noise_mag,approx_clean_mag into function
    ### returns all the scores: PESQ,STOI,SDR anything
    

  





