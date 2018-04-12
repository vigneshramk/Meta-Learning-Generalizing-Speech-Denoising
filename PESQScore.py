from os import system
from subprocess import check_output

pesqDIR = 'metrics/PESQ/src'
compileCMD = 'gcc -o '+ pesqDIR + '/PESQ '+ pesqDIR + '/*.c -lm -Wno-format'
fs = '+8000'
cleanfileDIR = 'metrics/PESQ/conform/or105.wav'
noisefileDIR = 'metrics/PESQ/conform/dg105.wav'
testCMD = pesqDIR + '/PESQ ' + fs + ' ' + cleanfileDIR + ' ' + noisefileDIR
system(compileCMD)
PESQScore = check_output(testCMD, shell=True)
print(PESQScore)