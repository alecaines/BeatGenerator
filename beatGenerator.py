# Authors: Alexander Caines, Shep Sims, Andrew Taylor
# Description: This file retrieves and preprocesses song data for beat generation

import pydub #allows for manipulation of audio
import scipy
import numpy as np
import glob as gb #locates files of a certain filetype
import os
import platform #just in case pydub (and ffmpeg) only work on windows

class BEATGENERATOR(object):
    def __init__(self):
        pass

    # converts mp3 to numpy array
    def transformData(self, f, normalized = True):
        #retrieves audio
        a = pydub.AudioSegment.from_mp3(file = f)

        # converts mp3 data to numpy array
        y = np.array(a.get_array_of_samples())

        #not what exactly what the channels represents
        if a.channels == 2:
            y = y.reshape((-1,2))
        
        #normalizes elements
        if normalized:
            return a.frame_rate, np.float32(y)/2**15
        else:
            return a.frame_rate, y
    
    #restricts data to a certain length (milliseconds)
    def processData(self, v, f):
        for i in range(len(v)):
            f.write(str(f[i]))

    def main(self):
        if platform.system() == 'Windows':
            if os.path.exists('../songs'): #for running on Alexander's machine
                mp3_files = gb.glob('../songs/*.mp3') #list of mp3 file addresses in a folder called songs

                #the following returns an np array (vector) representing one mp3 file
                #each element represents audio data at one millisecond in the audio file
                frame_rate, vector = self.transformData(mp3_files[0]) #framerate is in milliseconds
                
                filename = mp3_files[0]+".txt"
                f = open(filename, "w+")
                self.processData(vector, f) #should be a global array
                f.close()

            else:
                f = "Hip Hop SFX.mp3"
                #the following returns an np array (vector) representing one mp3 file
                frame_rate, train = self.transformData(f) #framerate is in milliseconds
                
                print(frame_rate)
            
        else:
            print("Support for " + str(platform.system()) + " will be implemented soon")

if __name__ == "__main__":
    BEATGENERATOR().main()