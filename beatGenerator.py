# Authors: Alexander Caines, Shep Sims, Andrew Taylor
# Description: This file retrieves and preprocesses song data for beat generation

#import matplotlib.pyplot as plt
#import pandas
#from keras.models import Sequential
#from keras.layers import Dense
#from keras.layers import LSTM
#from sklearn.preprocessing import MinMaxScaler
#from sklearn.metrics import mean_squared_error
import pydub #allows for manipulation of audio
from pydub.playback import play
import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Dense
from keras.models import Model
import numpy as np
import glob as gb # glob lists the files of a certain filetype in a folder specified by the programmer
import os

class BEATGENERATOR(object):
    def __init__(self):
        self.tensor = []

    # converts mp3 to numpy array
    def transformData(self, f, t = 3):
        duration = t*1000 #converts to milliseconds
        #retrieves audio
        a = pydub.AudioSegment.from_mp3(file = f)[:duration]

        # converts mp3 data to numpy array
        y = np.array(a.get_array_of_samples())

        #not what exactly what the channels represents other than two arbitrary filters in an audio file
        if a.channels == 2:
            y = y.reshape((-1,2))
        
        #normalizes elements and return
        #depending on what we use for our activation function, we may want to remove the zeroes and ones
        return a.frame_rate, a.channels, np.float32(y)/2**15
    
    # writes quantified audio data to txt
    def writeFile(self, v, filename, folder):
        if not os.path.exists(folder):
            os.makedirs(folder)
        f = open(folder + "/" + filename, "w+")
        for i in range(len(v)):
            f.write(str(v[i]))
        f.close()

    #transforms audio data back to audio
    def toAudio(self, rate, signal, channels):
        channel1 = signal[:,0]
        audio_segment = pydub.AudioSegment(
            channel1.tobytes(),
            frame_rate = rate,
            sample_width = channel1.dtype.itemsize,
            channels = channels
        )
        return audio_segment

    #plays a select audio file
    def playAudio(self, vector, frame_rate, channels):
        audio = self.toAudio(frame_rate, vector, channels)
        play(audio)
        
    def main(self):
        # I (Alexander) am unsure if ffmpeg works differently on different operating systems. So to be safe, I'm deferring to working with Windows.
        # I will check later if this works on linux. If you wish to check if the program runs on a MAC, install ffmpeg off the site I linked in the
        # else statement. After you have installed ffmpeg, replace 'Windows' with 'Darwin'
        if os.path.exists('../songs'): #for running on a windows machine
            mp3_files = gb.glob('../songs/*.mp3') #list of mp3 file addresses in a folder called songs sitting outside of this directory
            
            for i in range(len(mp3_files)):
                #the following returns an np array (vector) representing one mp3 file.
                # I believe each element represents audio data at one millisecond in the audio file
                # but I am not entirely sure. 
                frame_rate, channels, vector = self.transformData(mp3_files[i]) #Note, the framerate is in milliseconds

                ##stores the audio vector in a file
                #filename = str(mp3_files[i])[9:] + ".txt"
                #self.writeFile(vector, filename, "../vectorizedAudio") #should be a global array

                ##plays the audio
                #self.playAudio(vector, frame_rate, channels)

        else:
            f = "Hip Hop SFX.mp3"
            #the following returns an np array (vector) representing one mp3 file
            frame_rate, channels, vector = self.transformData(f) #framerate is in milliseconds

            ##stores the audio vector in a file
            #filename = str(f)+ ".txt"
            #self.writeFile(vector, filename, "../vectorizedAudio") #should be a global array
            
            ##plays the audio
            #self.playAudio(vector, frame_rate, channels)
            
        # else:
        #     print("Please install  ffmpeg for "+osys+". http://www.ffmpeg.org/download.html")
        #     print("Support for " + osys + " will be implemented soon")

if __name__ == "__main__":
    BEATGENERATOR().main()
