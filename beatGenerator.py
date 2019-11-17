# Authors: Alexander Caines, Shep Sims, Andrew Taylor
# Description: This file retrieves and preprocesses song data for beat generation

import pydub #allows for manipulation of audio
import scipy
import numpy as np
import glob as gb #locates files of a certain filetype
import platform #just in case pydub (and ffmpeg) only work on windows

class BEATGENERATOR(object):
    def __init__(self):
        pass

    def readData(self, f, normalized = False):
        a = pydub.AudioSegment.from_mp3(file = f)

        y = np.array(a.get_array_of_samples())
        if a.channels == 2:
            y = y.reshape((-1,2))
        if normalized:
            return a.frame_rate, np.float32(y)/2**15
        else:
            return a.frame_rate, y

    def main(self):
        if platform.system() == 'Windows':
            mp3_files = gb.glob('../songs/*.mp3') #list of mp3 file addresses

            print(self.readData(mp3_files[0]))
        else:
            print("Support for " + str(platform.system()) + " will be implemented soon")

if __name__ == "__main__":
    BEATGENERATOR().main()
