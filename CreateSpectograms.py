import torch
from LoadNoise import LoadData
import numpy as np
from torch.utils.data import DataLoader
import argparse
import os
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str,
						default='TIMIT/TRAIN', help="path for the data")
parser.add_argument('--noise_type', type=str,
						default='babble', help="noise type to add")
parser.add_argument('--num_spect', type=int,
						default=30, help="number of spectograms to create")


#JUST TO TEST THINGS OUT

#For the first stage of meta learning with one noise types and different SNR
#The training file will will be the same for all tasks, just have to create different dataloaders with the same noise type and different SNR

#For regular training with one noise type and different SNR
#training file has all the files with noise added at -6,-3,0,3,6 dB

#Testing files are the same for both since just need a new dataloader and specify what noise and SNR to test at 


#meta_training has multiple noise types as an array so the last dimension will be 5
#regular training with only one noise type will have last dimension as 1
#SNR input only needs to be one value if you just want to train/test at one noise type 
#tsv_file will be the meta_data file if u want to just add 1 noise type
#the tsv_file in regular training file has all noise types together

#same tsv_file since we are telling it what noise/snr to add 
args = parser.parse_args()
data_path = args.data_path
noise_type = args.noise_type

num_spectograms = args.num_spect
print('Spectogram number: %d' % num_spectograms)
print('Noise type: ' + noise_type)

noise_snr =[-6,-3,0,3,6] 
meta_training_data = LoadData(tsv_file='dataset/meta_data/train/train.txt', clean_dir=data_path, SNR=noise_snr, noise=noise_type, num_spectograms=num_spectograms)

#dataloaders
meta_train_loader = DataLoader(meta_training_data,batch_size=4610,shuffle=True,num_workers=0)

path1_name = './spectograms/spectograms_train' + str(num_spectograms) + '/noise/' + noise_type + '/train'
if not os.path.exists(path1_name):
        os.makedirs(path1_name)

path2_name = './spectograms/spectograms_train' + str(num_spectograms) + '/clean/train'
if not os.path.exists(path2_name):
        os.makedirs(path2_name)

#looping through the dataloader. Pytorch dataloader automatically randomizes the batches and gives u a new batch each iteration

for i,batch in enumerate(meta_train_loader):
    print('creating data....')
    clean = batch['clean_mag']
    noise = batch['noise_mag']

    print(clean.shape)
    print(noise.shape)
    break
    
print('done...')
print(clean.shape)
np.save('spectograms/spectograms_train' + str(num_spectograms) + '/clean/train/clean_frames_' + noise_type  + '.npy', clean)
for s, snr in enumerate(noise_snr):
    print(snr)
    print(noise[:,:,:,:,s].shape)
    np.save('spectograms/spectograms_train' + str(num_spectograms) + '/noise/' + noise_type + '/train/noise_'+ str(snr) + '.npy', noise[:,:,:,:,s])
    
