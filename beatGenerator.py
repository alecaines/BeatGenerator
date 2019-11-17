# Authors: Alexander Caines, Shep Sims, Andrew Taylor
# Description: This file retrieves and preprocesses song data for beat generation

import pydub #allows for manipulation of audio
import scipy
import numpy as np

class BEATGENERATOR(object):
    def __init__(self):
        pass

    def readData(self, normalized = False):
        a = pydub.AudioSegment.from_mp3(file = "Hip Hop SFX.mp3")
        y = np.array(a.get_array_of_samples())
        if a.channels == 2:
            y = y.reshape((-1,2))
        if normalized:
            return a.frame_rate, np.float32(y)/2**15
        else:
            return a.frame_rate, y

    def main(self):
        print(self.readData())

if __name__ == "__main__":
    BEATGENERATOR().main()
