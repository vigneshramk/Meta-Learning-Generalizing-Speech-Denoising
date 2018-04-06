import torch
from LoadNoise import LoadData
import numpy as np
from torch.utils.data import DataLoader

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
meta_training_data = LoadData(tsv_file='dataset/meta_data/train/train.txt', clean_dir='TIMIT/TRAIN/',SNR=[-6,-3,0,3,6],noise='babble')
reg_training_data = LoadData(tsv_file='dataset/meta_data/train/train.txt', clean_dir='TIMIT/TRAIN/',SNR=[6],noise='babble')

#dataloaders
meta_train_loader = DataLoader(meta_training_data,batch_size=128,shuffle=True,num_workers=0)
reg_train_loader = DataLoader(reg_training_data,batch_size=128,shuffle=True,num_workers=0) 

#looping through the dataloader. Pytorch dataloader automatically randomizes the batches and gives u a new batch each iteration
for i,batch in enumerate(meta_train_loader):
    print('meta batch')
    clean = batch['clean_mag']
    noise = batch['noise_mag']
    print(clean.shape)
    print(noise.shape)
    break

for i,batch in enumerate(reg_train_loader):
    print('regular batch')
    clean = batch['clean_mag']
    noise = batch['noise_mag']
    print(clean.shape)
    print(noise.shape)
    break
"""

num_spectograms = 3

full_clean_spectograms = None
full_noise_spectograms = None

for s in range(num_spectograms):
    for i,batch in enumerate(train_loader):
        clean = batch['clean_mag']
        noise = batch['noise_mag']
        
        if full_clean_spectograms is None:
            full_clean_spectograms = clean
            full_noise_spectograms = noise
        else:
            print(full_clean_spectograms.shape)
            print(clean.shape)
            full_clean_spectograms = np.concatenate([full_clean_spectograms,clean])
            full_noise_spectograms = np.concatenate([full_noise_spectograms,noise])
        break


print(full_clean_spectograms.shape)
print(full_noise_spectograms.shape)

    
"""