from os import system
from subprocess import check_output

def pesq(cleanfileDIR, noisefileDIR, fs=16000):
    pesqDIR = 'metrics/PESQ/src'
    compileCMD = 'gcc -o '+ pesqDIR + '/PESQ '+ pesqDIR + '/*.c -lm -Wno-format'
    fs = '+'+str(fs)
    # cleanfileDIR = 'metrics/PESQ/conform/or105.wav'
    # noisefileDIR = 'metrics/PESQ/conform/dg105.wav'
    testCMD = pesqDIR + '/PESQ ' + fs + ' ' + cleanfileDIR + ' ' + noisefileDIR
    system(compileCMD)
    PESQScore = check_output(testCMD, shell=True)
    print(PESQScore)
    return PESQScore