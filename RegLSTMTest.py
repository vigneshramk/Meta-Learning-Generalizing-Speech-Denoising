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
import PESQScore
import time
import lstm_mask
from TestAddNoiseLoader import TestSpect
from torch.utils.data import DataLoader


#CUDA_VISIBLE_DEVICES=2,3 python RegLSTMTest.py --noise_type bucc --noise_snr 0 --model_directory models/lstm_mask_normal_train/all_train/model_lstm.h5 --exp_name allbab0 --save_audio 0
parser = argparse.ArgumentParser()
parser.add_argument('--test_directory', type=str,
						default='../../Datasets/TIMIT/TEST', help="path for the data")
parser.add_argument('--noise_type', type=str,
						default='babble/', help="noise type") 
parser.add_argument('--noise_snr', type=str,
						default='-6', help="noise snr to test at") 
parser.add_argument('--model_directory', type=str,
						default='models/lstm_mask_normal_train/', help="path where model weight lies")
parser.add_argument('--save_audio', type=int,
						default=0, help="if u want to save the audio files")
parser.add_argument('--exp_name', type=str,
						default='test', help="name of your experiment")

 
args = parser.parse_args()
test_directory = args.test_directory
noise_type = args.noise_type
noise_snr = args.noise_snr
model_directory = args.model_directory
save_audio = args.save_audio
exp_name = args.exp_name

model_load_path = model_directory

print('Model Name ')
print(model_load_path + '\n')
print('noise type test')
print(noise_type + '\n')
print('noise snr test')
print(noise_snr + '\n')
print('experiment name')
print(exp_name + '\n')

model = lstm_mask.LSTM_Mask()
print('using mask!!!')
state_dict = torch.load(model_load_path, map_location=lambda storage, loc: storage)
model.load_state_dict(state_dict['state_dict'])


def test_mask(model,clean,noise):
    criterion = nn.MSELoss()
    noise_batch = lstm_mask.np_to_variable(noise)
    clean_batch = lstm_mask.np_to_variable(clean)

    approx_clean = model(noise_batch)
    loss = criterion(approx_clean, clean_batch)
    mse = loss.data[0]
    return approx_clean.data.cpu().numpy().T, mse


if torch.cuda.is_available():
    print('cuda available....')
    model.cuda()

model.eval()

MSE = []
PESQ_Noise = []
PESQ_Approx = []
STOI = []
SDR_Approx = []
#SDR_Noise = []

if not os.path.exists('results/'):
    os.makedirs('results/')

output_path = 'results/' + 'logfile_' + noise_type + '_' + noise_snr + '_' + exp_name + '.txt'
with open(output_path,'a') as f:
    f.write(model_load_path + '\n' + noise_type +  '\n' + noise_snr + '\n')

if not os.path.exists('results/'):
    os.makedirs('results/')
   
###dataloader for test

loader = TestSpect('dataset/meta_data/test/test.txt',test_directory,SNR=noise_snr,noise=noise_type)
test_loader = DataLoader(loader,batch_size=1,shuffle=True,num_workers=0)

for i, batch in enumerate(test_loader):
    print('Testing File: %d' % i)
   

    #get the clean magnitudes and the noise magnitude at the specific SNR
    clean_mag = batch['clean_mag'].numpy()
    noise_mag = batch['noise_mag'].numpy()
    

    approx_clean_mag, mse = test_mask(model, clean_mag, noise_mag)


    noise_audio = batch['noise_audio'].numpy()
    clean_audio = batch['clean_audio'].numpy()

    reshaped = np.reshape(approx_clean_mag,(approx_clean_mag.shape[0],approx_clean_mag.shape[1]))
    noise_audio = np.reshape(noise_audio,(noise_audio.shape[1]))
    clean_audio = np.reshape(clean_audio,(clean_audio.shape[1]))
    
    reconstruct_approx = utils.reconstruct_clean(noise_audio, reshaped)
    sdr_approx,sdr_noise = utils.calcluate_sdr(clean_audio, reconstruct_approx, noise_audio)
    print(sdr_approx)
    SDR_Approx.append(sdr_approx)
    print(sdr_noise)

    if save_audio==1 or i == 5:
        print('saving audio.....')
        
        wavfile.write('results/approx_' + noise_type + '_' + noise_snr + '_' + exp_name + '.WAV', 16000, reconstruct_approx)
        wavfile.write('results/actual_' + noise_type + '_' + noise_snr +  '_' + exp_name + '.WAV', 16000, noise_audio)
        wavfile.write('results/clean_' + exp_name + '.WAV', 16000, clean_audio) 
        


    """
    time.sleep(5)
    print('PESQ Scores')
    [clean,fs] = librosa.load(full_audio_clean_name,16000)
    [approx,fs] = librosa.load(approx_clean_name,16000)
    print(clean.shape)
    print(approx.shape)
    pesq_approx = PESQScore.pesq(full_audio_clean_name,approx_clean_name,16000)
    pesq_noise =PESQScore.pesq(full_audio_clean_name,full_audio_noise_name,16000)
    #pesq_clean = PESQScore.pesq(full_audio_clean_name,full_audio_clean_name,16000)
    print('done')
    
    PESQ_Approx.append(pesq_approx)
    PESQ_Noise.append(pesq_noise)
    
    print('Pesq Noise ' + str(pesq_noise))
    print('Pesq Approx ' + str(pesq_approx))
    print('Pesq Clean ' + pesq_clean)
    """
    MSE.append(mse)
    print('MSE %f' % mse) 
    
print(model_directory)
print(noise_type)
print(noise_snr)
#print('Mean Noise PESQ Score %f' % np.mean(PESQ_Noise))
#print('Mean Approx PESQ Score %f' % np.mean(PESQ_Approx))
print('Mean MSE Score %f' % np.mean(MSE))
print('Variance mse %f' % np.var(MSE))
print('SDR Approx %f' % np.mean(SDR_Approx))
print('SDR var %f'% np.var(SDR_Approx))

with open(output_path,'a') as f:
    f.write(str(np.mean(MSE))+ '\n' + str(np.min(MSE)) + '\n' + str(np.max(MSE)) + '\n' + str(np.var(MSE)))

    ###pass approx_clean_audio,clean_audio,noise_audio,clean_mag,noise_mag,approx_clean_mag into function
    ### returns all the scores: PESQ,STOI,SDR anythiniiiiig
    