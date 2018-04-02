import torch
from LoadNoise import LoadData
import numpy as np
from torch.utils.data import DataLoader

#JUST TO TEST THINGS OUT


#~/DataSets/LDC93S1/TIMITcorpus/TIMIT/TRAIN
#~/DataSets/LDC93S1/TIMITcorpus/TIMIT/TEST

#/dataset/reg_data/train
#/dataset/reg_data/test

#/dataset/meta_data/train
#/dataset/reg_data/test

#training_data = LoadData(tsv_file='dataset/meta_data/train/train.txt', clean_dir='/Users/tylervgina/DataSets/LDC93S1/TIMITcorpus/TIMIT/TRAIN/',SNR=-5,noise='babble')
training_data = LoadData(tsv_file='dataset/reg_data/train/train.txt',clean_dir='/Users/tylervgina/DataSets/LDC93S1/TIMITcorpus/TIMIT/TRAIN/',noise='babble')
train_loader = DataLoader(training_data,batch_size=32,shuffle=True,num_workers=0)

#For the first stage of meta learning with one noise types and different SNR
#The training file will will be the same for all tasks, just have to create different dataloaders with the same noise type and different SNR

#For regular training with one noise type and different SNR
#training file has all the files with noise added at -6,-3,0,3,6 dB

#Testing files are the same for both since just need a new dataloader and specify what noise and SNR to test at 


for i,batch in enumerate(train_loader):

    clean = batch['clean_mag']
    noise = batch['noise_mag']
    print(clean.shape)
    print(noise.shape)

