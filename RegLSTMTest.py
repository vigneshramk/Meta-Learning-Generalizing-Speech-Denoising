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



parser = argparse.ArgumentParser()
parser.add_argument('--test_directory', type=str,
						default='../../Datasets/TIMIT/TEST', help="path for the data")
parser.add_argument('--noise_type', type=str,
						default='babble/', help="noise type") 
parser.add_argument('--noise_snr', type=str,
						default='-6', help="noise snr to test at") 
parser.add_argument('--model_directory', type=str,
						default='models/normal_train/', help="path where model weight lies")
parser.add_argument('--save_audio', type=bool,
						default=False, help="if u want to save the audio files")
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

if not os.path.exists('results/'):
    os.makedirs('results/')

output_path = 'results/' + 'logfile_' + noise_type + '_' + noise_snr + '_' + exp_name + '.txt'
with open(output_path,'a') as f:
    f.write(model_load_path + '\n' + noise_snr +  '\n' + noise_snr + '\n')


   
###dataloader for test

loader = TestSpect('dataset/meta_data/test/test.txt',test_directory,SNR=noise_snr,noise=noise_type)
test_loader = DataLoader(loader,batch_size=1,shuffle=True,num_workers=0)

for i, batch in enumerate(test_loader):
    print('Testing File: %d' % i)
   

    #get the clean magnitudes and the noise magnitude at the specific SNR
    clean_mag = batch['clean_mag'].numpy()
    noise_mag = batch['noise_mag'].numpy()

    approx_clean_mag, mse = test_mask(model, clean_mag, noise_mag)
    
    ### WOULD PASS THROUGH approx_clean_mag into reconstruct.
    ### RETURNS Approx_clean_audio
    ### this is just for testing
    #approx_clean_audio = utils.reconstruct_clean(noise_audio, approx_clean_mag)
    #approx_clean_name =test_file_path + 'approx_clean_' + noise_snr + '_' + exp_name + '.WAV'
    #print('approx clean audio length') 
#    print(approx_clean_audio.shape)
    if save_audio:
        print('saving audio...')
        #wavfile.write(approx_clean_name, 16000,approx_clean_audio)
   #     wavfile.write(full_audio_clean_name,16000,clean_audio_new)
   #     wavfile.write(full_audio_noise_name,16000,noise_audio_new)
        #np.save(test_file_path + 'approx_clean_mag_' + noise_snr + '_' + exp_name + '.npy', approx_clean_mag)
        #with open('log_' + test_load_path + noise_snr + '_' + exp_name + '.txt','a') as f:
        #    f.write(full_audio_clean_name + '\t' + full_audio_noise_name + '\n')
        #    f.write(full_audio_clean_name + '\t' + approx_clean_name + '\n' )
     
    # logging the scores

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
print(noise_snr)
#print('Mean Noise PESQ Score %f' % np.mean(PESQ_Noise))
#print('Mean Approx PESQ Score %f' % np.mean(PESQ_Approx))
print('Mean MSE Score %f' % np.mean(MSE))
print('Minimum mse %d' %np.argmin(MSE))        
print('Max mse %d' % np.argmax(MSE))

with open(output_path,'a') as f:
    f.write(str(np.mean(MSE))+ '\n' + str(np.argmin(MSE)) + '\n' + str(np.argmax(MSE)) + '\n')

    ###pass approx_clean_audio,clean_audio,noise_audio,clean_mag,noise_mag,approx_clean_mag into function
    ### returns all the scores: PESQ,STOI,SDR anythiniiiiig
    

  






