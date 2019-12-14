import datetime
#import tensorflow as tf 
import matplotlib.pyplot as plt
import pydub #allows for manipulation of audio
from pydub.playback import play
import numpy as np
import glob as gb # glob lists the files of a certain filetype in a folder specified by the programmer
import os

def transformData(f, t = 3):
        duration = t*1000
        #retrieves audio
##        if type(f) != pydub.audio_segment.AudioSegment:
##            a = pydub.AudioSegment.from_mp3(file = f)[:duration].set_channels(1)
##        else:
        # converts mp3 data to numpy array
        
        a = pydub.AudioSegment.from_mp3(file = f)[:duration]

        y = np.array(a.get_array_of_samples()) #, ndmin = 2)
        y = np.float32(y)/2**15
        
        if a.channels == 2:
            return a.frame_rate, a.channels, np.float32(y.reshape((-1,2)))/2**15

        
        
        # tf.Tensor version:
        #y = tf.convert_to_tensor(y, dtype = tf.float32)
        
        #normalizes data and puts it into an np array:
        #y = np.array(list(map(lambda x:x/(2**15), a.get_array_of_samples())))

        return a.frame_rate, a.channels, y

def padData(data):
    maxx = max(list(map(lambda datum: len(datum), data)))
    padded = np.zeros((maxx, maxx))
    for datum in data:
        print(maxx)
        print(len(datum))
        padded[:len(datum)-1,:2] = datum
    return padded#list(map(lambda datum:(zeros[:len(datum),:2] = datum), data))

def main():
    if os.path.exists('../songs'):
        mp3_files = gb.glob('../songs/*.mp3') #list of mp3 file addresses in a folder called songs sitting outside of this directory

        data = list(map(lambda x: transformData(x)[2], mp3_files)) #gets list of audio data
        #maxx = max(list(map(lambda datum: len(datum), data))) #finds the max size
        minn = min(list(map(lambda datum: len(datum), data))) #finds the min size
        data = list(map(lambda  datum: datum[:minn], data))
        #data = list(map(lambda datum: np.pad(datum, (0, maxx - len(datum)),'constant'), data)) # Python kills process

            
        print(len(data[0]))
        print(len(data[1]))
##        print(len(data))
##        array = np.array(data)
##        print(len(padData(data)[0]))
        
if __name__ == "__main__":
    main()
