import numpy as np
import numpy.random as rand
import librosa
import argparse
import os
from subprocess import check_output
from scipy.io import wavfile

def sample(x, length, num, verbose=False):
    """
    Given audio x, sample `num` segments with `length` samples each
    """
    assert len(x) >= length
    segs = []
    start_idx_max = len(x)-length
    start_idx = np.around(rand.rand(num) * start_idx_max)
    for i in start_idx:
        segs.append(x[int(i):int(i)+length])
        if verbose:
            print('Take samples {} to {}...'.format(str(i),str(i+length)))
    return segs
    
def add_noise(x,n,snr=None):
    """
    Add user provided noise n with SNR=snr to signal x.
    SNR = 10log10(Signal Energy/Noise Energy)
    NE = SE/10**(SNR/10)
    """

    # Take care of size difference in case x and n have different shapes
    xlen,nlen = len(x),len(n)
    if xlen > nlen: # need to append noise several times to cover x range
        nn = np.tile(n,xlen/nlen+1)
        nlen = len(nn)
    else:
        nn = n
    if xlen < nlen: # slice a portion of noise
        nn = sample(nn,xlen,1)[0]
    else: # equal length
        nn = nn

    if snr is None: snr = (rand.random()-0.25)*20
    xe = x.dot(x) # signal energy
    ne = nn.dot(nn) # noise power
    nscale = np.sqrt(xe/(10**(snr/10.)) /ne) # scaling factor
    return x + nscale*nn


def reconstruct_clean(noise_audio, approx_clean_mag,frame_window=5):
    # use the noise audio to get the phase, and the missing frames
    # attach the missing noise frames
    # istft to reconstruct 
    noise_spect = librosa.stft(noise_audio, 320, 160)
    magN, phaseN = librosa.magphase(noise_spect)

    if magN.shape != approx_clean_mag.shape:
        print('Size not same. add noise frames')
        approx_clean_mag = np.hstack((magN[:,0:frame_window],approx_clean_mag, magN[:,-1*frame_window:] ))
        
    approx_clean_audio = librosa.istft(approx_clean_mag*phaseN,hop_length=160)
    return approx_clean_audio

def calcluate_sdr(clean,approx_clean,noise):
    len_clean = clean.shape[0]
    len_approx_clean = approx_clean.shape[0]

    if len_clean > len_approx_clean:
        clean = clean[0:len_approx_clean]
        noise = noise[0:len_approx_clean]
    else:
        approx_clean = approx_clean[0:len_clean]

    projection_approx = np.dot(clean,approx_clean) * clean / np.dot(clean,clean)
    projection_noise = np.dot(clean,noise) * clean / np.dot(clean,clean)

    e_approx = approx_clean - projection_approx
    e_noise = noise - projection_noise

    SDR_approx = 10.0 * np.log10(np.dot(projection_approx,projection_approx)/np.dot(e_approx,e_approx))
    SDR_noise = 10.0 * np.log10(np.dot(projection_noise,projection_noise)/np.dot(e_noise,e_noise))

    return SDR_approx,SDR_noise


def calcluate_pesq(clean, approx_clean):
    if not os.path.exists('./tmp/'):
        os.makedirs('./tmp/')
    wavfile.write('./tmp/clean.WAV', 16000, clean)
    wavfile.write('./tmp/approx_clean.WAV', 16000, approx_clean)
    pesqDIR = 'metrics/PESQ/src'
    testCMD = pesqDIR + '/PESQ +16000 ./tmp/clean.WAV ./tmp/approx_clean.WAV'
    PESQScore = check_output(testCMD, shell=True)
    
    return float(PESQScore[-5:])


def parse_arguments():
    # Command-line flags are defined here.
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-epochs', dest='num_epochs', type=int,
                        default=51, help="Number of epochs to train on.")
    parser.add_argument('--train_lr', dest='train_lr', type=float,
                        default=1e-5, help="The training learning rate.")
    parser.add_argument('--meta_lr', dest='meta_lr', type=float,
                        default=1e-4, help="The meta-training learning rate.")
    parser.add_argument('--exp_name' ,type=str, default= 'test', metavar = 'N',
                    help='Name of the experiment/weights saved ')                
    parser.add_argument('--noise_type', type=str, default='babble', metavar='N',
                    help='type of noise to add to test')
    parser.add_argument('--train_all' ,type=int, default=0, metavar='N',
                    help='testing text file')
    parser.add_argument('--reg_train' ,type=int, default=1, metavar='N',
                    help='testing text file')
    parser.add_argument('--num_spectograms' ,type=int, default=30, metavar='N',
                    help='number of spectograms training')
    parser.add_argument('--save_file_name' ,type=str, default='all_train', metavar='N',
                    help='name to save files')
    parser.add_argument('--test_file' ,type=str, default='../../Datasets/TIMIT/TEST', metavar='N',
                    help='name to save files')

    # # https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    # parser_group = parser.add_mutually_exclusive_group(required=False)
    # parser_group.add_argument('--render', dest='render',
    #                         action='store_true',
    #                         help="Whether to render the environment.")
    # parser_group.add_argument('--no-render', dest='render',
    #                         action='store_false',
    #                         help="Whether to render the environment.")
    # parser.set_defaults(render=False)

    return parser.parse_args()
