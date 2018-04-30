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
import random
from copy import deepcopy

parser = argparse.ArgumentParser()
parser.add_argument('--test_directory', type=str,
						default='../../Datasets/TIMIT/TEST', help="path for the data")
parser.add_argument('--noise_type', type=str,
						default='babble', help="noise type") 
parser.add_argument('--noise_snr', type=str,
						default='-6', help="noise snr to test at") 
parser.add_argument('--reg_model_directory', type=str,
						default='models/lstm_mask_normal_train/', help="path where model weight lies")
parser.add_argument('--maml_model_directory', type=str,
						default='models/lstm_mask_normal_train/', help="path where model weight lies")
parser.add_argument('--maml_lr', type=float,
						default=.01, help="path where model weight lies")
parser.add_argument('--reg_lr', type=float,
						default=.01, help="path where model weight lies")
parser.add_argument('--gradient_updates', type=int,
						default=3, help="path where model weight lies")
parser.add_argument('--batch_size', type=int,
						default=128, help="path where model weight lies")
parser.add_argument('--frame_size', type=int,
						default=32, help="path where model weight lies")
parser.add_argument('--save_audio', type=int,
						default=0, help="if u want to save the audio files")
parser.add_argument('--exp_name', type=str,
						default='test', help="name of your experiment")
parser.add_argument('--runs', type=int,
						default= 30, help="name of your experiment")

args = parser.parse_args()
test_directory = args.test_directory
noise_type = args.noise_type
noise_snr = args.noise_snr
reg_model_directory = args.reg_model_directory
maml_model_directory = args.maml_model_directory
maml_lr = args.maml_lr
reg_lr = args.reg_lr
K = args.gradient_updates
save_audio = args.save_audio
exp_name = args.exp_name

print('Regular Model Name')
print(reg_model_directory)
print('Meta Model Name')
print(maml_model_directory)
print('noise type test')
print(noise_type + '\n')
print('noise snr test')
print(noise_snr + '\n')
print('meta lr: %f'%maml_lr)
print('reg lr: %f'%reg_lr)
print('Gradient Updates %d '%K)
print('experiment name')
print(exp_name + '\n')

def test_mask(model,clean,noise):
    criterion = nn.MSELoss()
    noise_batch = lstm_mask.np_to_variable(noise)
    clean_batch = lstm_mask.np_to_variable(clean)

    approx_clean = model(noise_batch)
    loss = criterion(approx_clean, clean_batch)
    mse = loss.data[0]
    return approx_clean.data.cpu().numpy().T, mse



if not os.path.exists('meta_results/'):
    os.makedirs('meta_results/')

output_path = 'meta_results/' + 'logfile_' + noise_type + '_' + noise_snr + '_' + exp_name + '.txt'

with open(output_path,'a') as f:
    f.write(reg_model_directory + '\n' + maml_model_directory + '\n' + noise_type +  '\n' + noise_snr + '\n' + str(reg_lr) + '\n' + str(maml_lr) + '\n' + str(K))

total_runs = args.runs
total_SDR_reg = []
total_SDR_maml = []


loader = TestSpect('dataset/meta_data/test/test.txt',test_directory,SNR=noise_snr,noise=noise_type)
test_loader = DataLoader(loader,batch_size=1,shuffle=True,num_workers=0)

for runs in range(total_runs):
    print("RUN ....... %d" % runs)
    MSE_reg = []
    MSE_maml = []
    SDR_reg = []
    SDR_maml = []
    PESQ_reg = []
    PESQ_maml = []

    reg_model = lstm_mask.LSTM_Mask()
    maml_model = lstm_mask.LSTM_Mask()

    reg_state_dict = torch.load(reg_model_directory, map_location=lambda storage, loc: storage)
    maml_state_dict = torch.load(maml_model_directory, map_location=lambda storage, loc: storage)

    reg_model.load_state_dict(reg_state_dict['state_dict'])
    maml_model.load_state_dict(maml_state_dict['state_dict'])

    criterion_reg = nn.MSELoss()
    criterion_maml = nn.MSELoss()
    reg_optimizer = torch.optim.Adam(reg_model.parameters(), lr=reg_lr)
    maml_optimizer = torch.optim.Adam(maml_model.parameters(), lr=maml_lr) 

    if torch.cuda.is_available():
        print('cuda is available.....')
        reg_model.cuda()
        maml_model.cuda()



    reg_model.eval()
    maml_model.eval()

    

    batch_size = args.batch_size
    frame_size = args.frame_size
    testing_size = 100
    batch_train = np.zeros((batch_size,frame_size,161))
    batch_labels = np.zeros((batch_size,frame_size,161))


    for i, batch in enumerate(test_loader):

        clean_mag = batch['clean_mag'].numpy()
        noise_mag = batch['noise_mag'].numpy()

        if i < batch_size:
            print('Getting Update Data... %d' %i)
            spect_shape = clean_mag.shape
            width = spect_shape[1]

            start = random.sample(range(0,width-frame_size+1),1)
            clean_C = clean_mag[0,start[0]:start[0]+frame_size,:]
            noise_C = noise_mag[0,start[0]:start[0]+frame_size,:]

            batch_train[i,:,:] = noise_C
            batch_labels[i,:,:] = clean_C

        elif i == batch_size:
            print(batch_train.shape)
            print(batch_labels.shape)
            noise_batch = lstm_mask.np_to_variable(batch_train)
            clean_batch = lstm_mask.np_to_variable(batch_labels)
            noise_batch_copy = deepcopy(noise_batch)
            print('Applying Gradients....')
            for k in range(K):
                
                maml_approx = maml_model(noise_batch) 
                reg_approx = reg_model(noise_batch_copy)

                reg_loss = criterion_reg(reg_approx, clean_batch)
                maml_loss = criterion_maml(maml_approx,clean_batch)

                reg_optimizer.zero_grad()
                maml_optimizer.zero_grad()

                reg_loss.backward()
                maml_loss.backward()

                reg_optimizer.step()
                maml_optimizer.step()

                print('Reg loss %d: %f' % (k, reg_loss.data[0]))
                print('Maml loss %d: %f'% (k, maml_loss.data[0]))

        if i >= batch_size and i < batch_size + testing_size:
            print('Testing Models.... %d' %(i-batch_size + 1))

            noise_audio = batch['noise_audio'].numpy()
            clean_audio = batch['clean_audio'].numpy()

            

            reg_approx_mag, reg_mse = test_mask(reg_model, clean_mag, noise_mag)
            maml_approx_mag, maml_mse = test_mask(maml_model, clean_mag, noise_mag)

            noise_audio = np.reshape(noise_audio,(noise_audio.shape[1]))
            clean_audio = np.reshape(clean_audio,(clean_audio.shape[1]))

            reg_reshaped = np.reshape(reg_approx_mag,(reg_approx_mag.shape[0],reg_approx_mag.shape[1]))
            maml_reshape = np.reshape(maml_approx_mag,(maml_approx_mag.shape[0],maml_approx_mag.shape[1])) 

            reg_reconstruct = utils.reconstruct_clean(noise_audio, reg_reshaped)
            maml_reconstruct = utils.reconstruct_clean(noise_audio, maml_reshape)

            reg_sdr,sdr_noise = utils.calcluate_sdr(clean_audio, reg_reconstruct, noise_audio)
            maml_sdr,sdr_noise = utils.calcluate_sdr(clean_audio, maml_reconstruct, noise_audio)

    #        reg_pesq = utils.calcluate_pesq(clean_audio, reg_reconstruct)
    #        maml_pesq = utils.calcluate_pesq(clean_audio, maml_reconstruct)

            reg_pesq = 0
            maml_pesq = 0
            if save_audio==1 or i == batch_size:
                print('saving audio.....')
            
                wavfile.write('meta_results/reg_approx_' + noise_type + '_' + noise_snr + '_' + exp_name + '.WAV', 16000, reg_reconstruct)
                wavfile.write('meta_results/maml_approx_' + noise_type + '_' + noise_snr + '_' + exp_name + '.WAV', 16000, maml_reconstruct)
                wavfile.write('meta_results/actual_' + noise_type + '_' + noise_snr +  '_' + exp_name + '.WAV', 16000, noise_audio)
                wavfile.write('meta_results/clean_' + exp_name + '.WAV', 16000, clean_audio) 

            MSE_reg.append(reg_mse)
            MSE_maml.append(maml_mse)

            SDR_reg.append(reg_sdr)
            SDR_maml.append(maml_sdr)

            PESQ_reg.append(reg_pesq)
            PESQ_maml.append(maml_pesq)

            print('Regular MSE: %f SDR: %f PESQ: %f' % (reg_mse, reg_sdr, reg_pesq))
            print('MAML MSE: %f SDR: %f PESQ: %f' % (maml_mse, maml_sdr, maml_pesq))

        if i >= batch_size + testing_size:
            break

    total_SDR_reg.append(np.mean(SDR_reg))
    total_SDR_maml.append(np.mean(SDR_maml))



print('Done...')
print(reg_model_directory)
print(maml_model_directory)
print(noise_type)
print(noise_snr)
print(K)
print(batch_size)

print('Reg Mean MSE: %f Mean SDR %f Mean PESQ %f' % (np.mean(MSE_reg), np.mean(total_SDR_reg), np.var(total_SDR_reg)))
print('MAML Mean MSE: %f Mean SDR %f Mean PESQ %f' % (np.mean(MSE_maml), np.mean(total_SDR_maml), np.var(total_SDR_maml)))

with open(output_path,'a') as f:
    f.write(str(np.mean(MSE_reg))+ '\n' + str(np.mean(SDR_reg)) + '\n' + str(np.mean(PESQ_reg)) + '\n' 
            + str(np.mean(MSE_maml)) + '\n' + str(np.mean(SDR_maml) + '\n' + str(np.mean(PESQ_maml)) + '\n'))
