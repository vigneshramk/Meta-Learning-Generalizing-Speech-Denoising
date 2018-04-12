import numpy as np
import numpy.random as rand
import librosa

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
    
    approx_clean_audio = librosa.core.istft(approx_clean_mag*phaseN,hop_length=160)
    
    return approx_clean_audio
