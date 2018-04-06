import numpy as np
import utils
import torch
from torch.utils.data import Dataset
import csv
import os
import librosa
import librosa.display
import librosa.core



class LoadData(Dataset):
    def __init__(self, tsv_file, clean_dir, features=None,num_spectograms=3,
                    hop_size=160, n_fft=320, fs=16000,frame_size=21, SNR=None, noise=None):

        #Add each line to a list
        #tsv_file is the dataset which has names of files 
        #clean directory is where the data lies
        total = []
        with open(tsv_file) as tsv:
            for line in csv.reader(tsv, delimiter = '\t'):
                total.append(line)

        self.features = features
        self.wav = total
        self.hop_size = hop_size
        self.n_fft = n_fft
        self.fs = fs
        self.snr = SNR
        self.noise = noise
        self.clean_dir = clean_dir
        self.frame_size = frame_size
        self.num_spectograms = num_spectograms

    def __len__(self):
        return len(self.wav)
    
    
    def __getitem__(self, idx):

        #Read Audio File
        
        file_name = self.wav[idx][0]
        file_path = os.path.join(self.clean_dir, file_name)
        [clean_audio,fs] = librosa.load(file_path,self.fs)

        #Get Clean Spectogram
        clean_spect = librosa.stft(clean_audio,n_fft=self.n_fft, hop_length=self.hop_size)

        #If snr is None, read from the file
        #if noise is None, read from the file
        #else it is a parameter to dataloader
        if self.snr is None:
            self.snr = [float(self.wav[idx][1])]
        if self.noise is None:
            self.noise = self.wav[idx][2]

        #adding different noise types
        if self.noise == 'babble':
            [noise_add, noise_fs] = librosa.load('noise/babble_train.wav', self.fs)
            
        elif self.noise == 'engine':
            [noise_add, noise_fs] = librosa.load('noise/engine_train.wav', self.fs)
            
        elif self.noise == 'factory1':
            [noise_add, noise_fs] = librosa.load('noise/factory1_train.wav', self.fs)

        #creating the spectogram tensor that depends on how many SNR levels to add 
        flatten_length = (self.n_fft/2 + 1) * self.frame_size
        flatten_noise_spectograms = np.zeros((flatten_length, len(self.snr)))
        flatten_clean_spectograms = np.zeros((flatten_length, len(self.snr)))

        #loop through SNR array 
        for s in range(len(self.snr)):
            snr_level = self.snr[s]
            noise_audio = utils.add_noise(clean_audio, noise_add, snr_level)

            #Get Noise Spectogram
            noise_spect = librosa.stft(noise_audio,n_fft=self.n_fft, hop_length=self.hop_size)
        
            #Get only the magnitudes for clean and noise spectograms
            magC, phaseC = librosa.magphase(clean_spect)
            magN, phaseN = librosa.magphase(noise_spect)

            #Getting equal size spectograms that is frame size long. Start at random spot
            spect_shape = magC.shape
            width = spect_shape[1]

            start = np.random.randint(0,width-self.frame_size+1)
            magC = magC[:,start:start + self.frame_size]
            magN = magN[:,start:start + self.frame_size]

            #flatten spectogram 
            flatten_magC = magC.flatten()
            flatten_magN = magN.flatten()

            #adding it to the spectograms. will be size (flatten_length,# of SNR)
            flatten_clean_spectograms[:,s] = flatten_magC
            flatten_noise_spectograms[:,s] = flatten_magN



        #Return shortened clean and noise spectogram pairs 
        sample = {'clean_mag': flatten_clean_spectograms, 'noise_mag': flatten_noise_spectograms}
        return sample