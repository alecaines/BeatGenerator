# Authors: Alexander Caines, Shep Sims, Andrew Taylor
# Description: This file retrieves and preprocesses song data for beat generation

import pydub #allows for manipulation of audio
from pydub.playback import play
import numpy as np
import glob as gb # glob lists the files of a certain filetype in a folder specified by the programmer
import os

from keras.layers import Lambda, Input, Dense
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K

class BEATGENERATOR(object):
    def __init__(self):
        pass

    # converts mp3 to numpy array
    def transformData(self, f):
        #retrieves audio
        if type(f) != pydub.audio_segment.AudioSegment:
            a = pydub.AudioSegment.from_mp3(file = f)
        else:
            a = f
        # converts mp3 data to numpy array
        y = np.array(a.get_array_of_samples())
        print(y.shape)
        #not what exactly what the channels represents other than two arbitrary filters in an audio file
        if a.channels == 2:
            y = y.reshape((-1,2))

        print(y.shape)
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
        print(signal.shape)
        channel1 = signal[:,0]
        channel2 = signal[:,1]
        print(signal[:,0])
        audio_segment = pydub.AudioSegment(
            channel1.tobytes(),
            frame_rate = rate,
            sample_width = channel1.dtype.itemsize,
            channels = channels
        )
        return audio_segment

    def playAudio(self, audio_segment):
        play(audio_segment)
        
    def main(self):
        # I (Alexander) am unsure if ffmpeg works differently on different operating systems. So to be safe, I'm deferring to working with Windows.
        # I will check later if this works on linux. If you wish to check if the program runs on a MAC, install ffmpeg off the site I linked in the
        # else statement. After you have installed ffmpeg, replace 'Windows' with 'Darwin'
        print(os.path.exists('../songs'))
        if os.path.exists('../song'): #for running on a windows machine
            mp3_files = gb.glob('../songs/*.mp3') #list of mp3 file addresses in a folder called songs sitting outside of this directory
            
            for i in range(len(mp3_files)):
                #the following returns an np array (vector) representing one mp3 file.
                # I believe each element represents audio data at one millisecond in the audio file
                # but I am not entirely sure. 
                frame_rate, channels, vector = self.transformData(mp3_files[i]) #Note, the framerate is in milliseconds

                #filename = str(mp3_files[i])[9:] + ".txt"
                #self.writeFile(vector, filename, "../vectorizedAudio") #should be a global array
                self.playAudio(vector, frame_rate, channels)

        else:
            f = "Hip Hop SFX.mp3"
            #the following returns an np array (vector) representing one mp3 file
            frame_rate, channels, vector = self.transformData(f) #framerate is in milliseconds
            filename = str(f)+ "test.txt"
            #self.writeFile(vector, filename, "../vectorizedAudio") #should be a global array

            #original_dim = len(vector)
            #x_train = vector
            #x_test = vector

            #input_shape = (original_dim,)
            #intermediate_dim = 512
            #batch_size = 128
            #latent_dim = 2
            #epochs = 50

            #inputs = Input(shape=input_shape, name="encoder_input")
            #x = Dense(intermediate_dim, activation="relu")(inputs)
            
            audio_decompressed = self.toAudio(frame_rate, vector,channels)
            frame_rate2, channels2, vector2 = self.transformData(audio_decompressed)
            audio_decompressed2 = self.toAudio(frame_rate2, vector2,channels2)
            self.playAudio(audio_decompressed)
            self.playAudio(audio_decompressed2)
        # else:
        #     print("Please install  ffmpeg for "+osys+". http://www.ffmpeg.org/download.html")
        #     print("Support for " + osys + " will be implemented soon")

if __name__ == "__main__":
    BEATGENERATOR().main()
