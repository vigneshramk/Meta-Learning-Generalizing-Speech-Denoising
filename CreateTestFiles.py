import numpy as np
import utils
import csv
import os
import librosa
import argparse
from scipy.io import wavfile


parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str,
						default='TIMIT/TEST', help="path for the data")

 
args = parser.parse_args()
data_path = args.data_path
noise = 'babble'
noise_snr = [-15,-10,-6,-2,0,2,4,8]
fs=16000
n_fft = 320
hop_size = 160


#Load file names

total = []
save_directory = 'spectograms/noise/test/'
with open('dataset/meta_data/test/test.txt') as tsv:
    for line in csv.reader(tsv, delimiter = '\t'):
        total.append(line)

for idx, line in enumerate(total):
    print( )
    file_name = line[0]
    file_path = os.path.join(data_path, file_name)

    #get clean spectogram
    [clean_audio,fs] = librosa.load(file_path, fs)
    clean_spect = librosa.stft(clean_audio, n_fft=n_fft, hop_length=hop_size)

    #load the noise audio
    if noise == 'babble':
        [noise_add, noise_fs] = librosa.load('noise/babble_test.wav', fs)
    
    #get clean magnitude
    magC, phaseC = librosa.magphase(clean_spect)

    #make directory and save clean magnitude
    save_directory_noise =  save_directory + noise + '/' + str(idx) + '/'
    if not os.path.exists(save_directory_noise):
        os.makedirs(save_directory_noise)
    np.save(save_directory_noise + 'spect_clean.npy', magC)

    for snr in noise_snr:


        #get the snr, add to audio and get the noise magnitude
        noise_audio = utils.add_noise(clean_audio, noise_add, snr)
        noise_spect = librosa.stft(noise_audio, n_fft=n_fft, hop_length=hop_size)
        magN, phaseN = librosa.magphase(noise_spect)

        #save the noise audio and the magnitude
        np.save(save_directory_noise + 'spect_' + str(snr) + '.npy', magN)
        wavfile.write(save_directory_noise + 'audio_' + str(snr) + '.WAV',fs,noise_audio)



        

 

    





