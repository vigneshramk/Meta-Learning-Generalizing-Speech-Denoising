import numpy as np
import utils
import torch
from torch.utils.data import Dataset
import csv
import os
import librosa
import librosa.display
import librosa.core
import random

"""

Gated recurrent network mask estiamtion

class GRU(nn.Module):
	def __init__(self, input_channels, hidden_units, hidden_layers, output_channels,bi):
		super(GRU, self).__init__()
		self.hidden_units = hidden_units
		self.hidden_layers = hidden_layers
		self.input_channels = input_channels
		self.output_channels = output_channels
		if bi:
			print('Model is Bi Directional')
			self.mult = 2
		else:
			self.mult = 1

		self.gru = nn.GRU(input_channels, hidden_units, hidden_layers, batch_first=True, bidirectional=bi)
		self.fc = nn.Linear(hidden_units * self.mult, output_channels)


	def forward(self, x):

		h0 = Variable(torch.zeros(self.hidden_layers * 2, x.size(0), self.hidden_units).type(Tensor))
		self.gru.flatten_parameters()

		#test = self.conv1(x)
		#print(test.size())
		
		out, _ = self.gru(x, h0)
		#print(out.size())
		output = F.sigmoid(self.fc(out))
		output = output.squeeze(0)

		return output
"""

class LoadData(Dataset):
    def __init__(self, tsv_file, clean_dir, features=None, num_spectograms=10,single_frame=True,
                    hop_size=160, n_fft=320, fs=16000,frame_size=11, SNR=None, noise=None):

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
        self.single_frame = single_frame
        self.middle = int((frame_size - 1)/2)

    def __len__(self):
        return len(self.wav)
    
    
    def __getitem__(self, idx):
        print(idx)

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


        flatten_noise_spectograms = np.zeros((self.num_spectograms, int(flatten_length), len(self.snr)))

        if self.single_frame:
            flatten_clean_spectograms = np.zeros((self.num_spectograms, int(self.n_fft/2+1) , len(self.snr)))

        else:
            flatten_clean_spectograms = np.zeros((self.num_spectograms, int(flatten_length), len(self.snr)))

        #loop through SNR array 
        for s in range(len(self.snr)):
            snr_level = self.snr[s]
            noise_audio = utils.add_noise(clean_audio, noise_add, snr_level)
            #Get Noise Spectogram
            noise_spect = librosa.stft(noise_audio,n_fft=self.n_fft, hop_length=self.hop_size)
        
            #Get only the magnitudes for clean and noise spectograms
            
            magN, phaseN = librosa.magphase(noise_spect)

            #Getting equal size spectograms that is frame size long. Start at random spot

            if s == 0:
                #only on the first SNR, you need to calculate:
                # - clean magnitude
                # - start of the cropping
                # - clean magnitude cropped 

                magC, phaseC = librosa.magphase(clean_spect)
                spect_shape = magC.shape
                width = spect_shape[1]
                #start = np.random.randint(0,width-self.frame_size+1)
                starts = random.sample(range(0,width-self.frame_size+1),self.num_spectograms)
                for num, start in enumerate(starts):
                    #loop through each of the start frames
                    #crop the spectogram, flatten, and its the same window for all noise types 
                    # which means (num_spectogram) gives you a dimxtotal_noise size matrix and each column should be the same for each spectogram window
                    magC_Crop = magC[:,start:start + self.frame_size]

                    if self.single_frame:
                    
                        magC_Crop = magC_Crop[:,self.middle]
                

                    flatten_magC = magC_Crop.flatten()
                    flatten_clean_spectograms[num]= np.reshape(flatten_magC,(len(flatten_magC),1))
                

            for num, start in enumerate(starts):
                #adding noises
                #loop through each of the start frames
                #crop the specotgram
                #flatten it out 
                #and add it to one column of the matrix that represents the noise type
                #in this case, you are filling in each speoctgram first and then moving on to a new noise type 
                magN_Crop = magN[:,start:start + self.frame_size]

                #flatten spectogram 
                flatten_magN = magN_Crop.flatten()

                #adding it to the spectograms. will be size (flatten_length,# of SNR)
                #add the same clean flatten spectograms for each dimension
                flatten_noise_spectograms[num][:,s] = flatten_magN

        #np.save('spectograms/noise/multiple_noise/noise_' + str(idx) + '.npy', flatten_noise_spectograms)
        #np.save('spectograms/clean/single_frame/clean_' + str(idx) + '.npy', flatten_clean_spectograms)

        

        #Return shortened clean and noise spectogram pairs 
        sample = {'clean_mag': flatten_clean_spectograms, 'noise_mag': flatten_noise_spectograms}
        return sample