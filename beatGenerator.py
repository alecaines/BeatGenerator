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
    def transformData(self, f, normalized = True, t = 3):
        duration = t*1000 #converts to miliseconds

        #retrieves audio
        a = pydub.AudioSegment.from_mp3(file = f)[:duration]

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
    
    # writes quantified audio data to txt
    def proofOfWork(self, v, filename):
        f = open(filename, "w+")
        for i in range(len(v)):
            f.write(str(v[i]))
        f.close()


    def main(self):
        osys = platform.system()
        if osys == 'Windows':
            if os.path.exists('../songs'): #for running on Alexander's machine
                mp3_files = gb.glob('../songs/*.mp3') #list of mp3 file addresses in a folder called songs
                
                for i in range(len(mp3_files)):
                    #the following returns an np array (vector) representing one mp3 file
                    #each element represents audio data at one millisecond in the audio file
                    frame_rate, vector = self.transformData(mp3_files[i]) #framerate is in milliseconds
                    filename = str(mp3_files[i])[9:] + ".txt"
                    self.proofOfWork(vector, filename) #should be a global array

            else:
                f = "Hip Hop SFX.mp3"
                #the following returns an np array (vector) representing one mp3 file
                frame_rate, vector = self.transformData(f) #framerate is in milliseconds
                
                print(frame_rate)
            
        else:
            print("Please install  ffmpeg for "+osys+". http://www.ffmpeg.org/download.html")
            print("Support for " + osys + " will be implemented soon")

if __name__ == "__main__":
    BEATGENERATOR().main()