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
import PESQScore
import time
#mask = gru(noise_mag)
		#print(mask.size())

		#approx_mag = noise_mag * mask

		#loss = F.mse_loss(approxMag,magSpect)

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

def test(model, clean, noise):
    criterion = nn.MSELoss()
    magSz, totalTime = noise.shape
    specWidth = window_size*2+1
    inputSz = magSz * specWidth
    batch = np.zeros((totalTime-window_size*2, inputSz))
    for idx in range(totalTime-window_size*2):
        batch[idx, :] = noise[:, idx:idx+specWidth].flatten()
    batch_tensor = autoencoder.np_to_variable(batch)
    reconstruct = model(batch_tensor).data.cpu().numpy().T
    loss = criterion(autoencoder.np_to_variable(reconstruct),autoencoder.np_to_variable(clean[:,window_size:-1*window_size]))
    mse = loss.data[0]
    #mse = ((reconstruct - clean[:,window_size:-1*window_size]) ** 2).mean(axis=None)
    return reconstruct, mse

if torch.cuda.is_available():
    print('cuda available....')
    model.cuda()

model.eval()

MSE = []
PESQ_Noise = []
PESQ_Approx = []
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
    #print(full_audio_clean_name)
    #print(full_audio_noise_name)
    [clean_audio,fs] = librosa.load(full_audio_clean_name,16000)
    [noise_audio,fs] = librosa.load(full_audio_noise_name,16000)
"""
    clean_spect = librosa.stft(clean_audio,n_fft=320,hop_length=160)
    magC,phaseC =  librosa.magphase(clean_spect)
    clean_audio_new = librosa.istft(magC*phaseC,hop_length=160)

    noise_spect = librosa.stft(noise_audio,n_fft=320,hop_length=160)
    magN,phaseN = librosa.magphase(noise_spect)
    noise_audio_new = librosa.istft(magN*phaseN,hop_length=160)
    print(clean_audio_new.shape)
    print(noise_audio_new.shape)
    print('new audio sizes')
"""    
    approx_clean_mag, mse = test(model, clean_mag, noise_mag)
    
    ### WOULD PASS THROUGH approx_clean_mag into reconstruct.
    ### RETURNS Approx_clean_audio
    ### this is just for testing
    approx_clean_audio = utils.reconstruct_clean(noise_audio, approx_clean_mag)
    approx_clean_name =test_file_path + 'approx_clean_' + noise_snr + '_' + exp_name + '.WAV'
    print('approx clean audio length') 
    print(approx_clean_audio.shape)
    if save_audio:
        print('saving audio...')
        wavfile.write(approx_clean_name, 16000,approx_clean_audio)
   #     wavfile.write(full_audio_clean_name,16000,clean_audio_new)
   #     wavfile.write(full_audio_noise_name,16000,noise_audio_new)
        #np.save(test_file_path + 'approx_clean_mag_' + noise_snr + '_' + exp_name + '.npy', approx_clean_mag)
        with open(test_load_path + noise_snr + '_' + exp_name + '.txt','a') as f:
            f.write(full_audio_clean_name + '\t' + full_audio_noise_name + '\n')
            f.write(full_audio_clean_name + '\t' + approx_clean_name + '\n' )
    """ 
    # logging the scores
    time.sleep(10)
    print('PESQ Scores')
    [clean,fs] = librosa.load(full_audio_clean_name,16000)
    [approx,fs] = librosa.load(approx_clean_name,16000)
    print(clean.shape)
    print(approx.shape)
    pesq_approx = PESQScore.pesq(full_audio_clean_name,approx_clean_name,16000)
    pesq_noise =PESQScore.pesq(full_audio_clean_name,full_audio_noise_name,16000)
    pesq_clean = PESQScore.pesq(full_audio_clean_name,full_audio_clean_name,16000)
    print('done')
    
    PESQ_Approx.append(pesq_approx)
    PESQ_Noise.append(pesq_noise)
    
    
    print('Pesq Noise ' + pesq_noise)
    print('Pesq Approx ' + pesq_approx)
    print('Pesq Clean ' + pesq_clean)
    """
    MSE.append(mse)
    print('MSE %f' % mse) 
    
print(model_name)
print(noise_snr)
#print('Mean Noise PESQ Score %f' % np.mean(PESQ_Noise))
#print('Mean Approx PESQ Score %f' % np.mean(PESQ_Approx))
print('Mean MSE Score %f' % np.mean(MSE))
print('Minimum mse %d' %np.argmin(MSE))        
print('Max mse %d' % np.argmax(MSE))

    ###pass approx_clean_audio,clean_audio,noise_audio,clean_mag,noise_mag,approx_clean_mag into function
    ### returns all the scores: PESQ,STOI,SDR anythiniiiiig
    

  






