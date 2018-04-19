import numpy as np
import utils 
import torch
from torch.utils.data import Dataset
import csv
import os
import librosa
import librosa.display
import librosa.core



class TestSpect(Dataset):
    def __init__(self,tsv_file, val_dir, features=None, num_features=None,
                    hop_size=160, n_fft=320, fs=16000,SNR=10, noise='white'):

        total = []
        with open(tsv_file) as tsv:
            for line in csv.reader(tsv, delimiter = '\t'):
                total.append(line)

        self.features = features
        self.wav = total
        self.hop_size = hop_size
        self.num_features = num_features
        self.n_fft = n_fft
        self.fs = fs
        self.snr = int(SNR)
        self.noise = noise
        self.val_dir = val_dir

    def __len__(self):
        return len(self.wav)
    
    
    def __getitem__(self, idx):

        #train file will have audio, type noise, SNR

        wav_files = self.wav
        file_name = wav_files[idx][0]
        file_path = os.path.join(self.val_dir, file_name)
        [audio, fs] = librosa.load(file_path,self.fs)

        clean_spect = librosa.stft(audio,n_fft=self.n_fft, hop_length=self.hop_size)
            
        if self.noise == 'babble':
            [sub_noise, sub_fs] = librosa.load('noise/babble_test.wav',self.fs) 
        elif self.noise == 'factory1':
            [sub_noise, sub_fs] = librosa.load('noise/factory1_test.wav',self.fs) 
        elif self.noise == 'engine':
            [sub_noise, sub_fs] = librosa.load('noise/engine_test.wav',self.fs)
        elif self.noise =='ops':
            [sub_noise, sub_fs] = librosa.load('noise/ops.wav',self.fs)
        elif self.noise == 'bucc':
            [sub_noise, sub_fs] = librosa.load('noise/bucc.wav',self.fs)


        elif self.noise =='white':
            sub_noise = np.random.normal(0,1,audio.shape)
        
        noise_audio = utils.add_noise(audio,sub_noise, self.snr)    
                
        noise_spect = librosa.stft(noise_audio,n_fft=self.n_fft, hop_length=self.hop_size)
        
        magC, phaseC = librosa.magphase(clean_spect)
        magN, phaseN = librosa.magphase(noise_spect)

        magClean = np.transpose(magC)
        magNoise = np.transpose(magN)

        #make this a function later on

        sample = {'clean_mag': magClean, 'noise_mag': magNoise, 'noise_audio' :noise_audio,'clean_audio': audio }
        return sample 