# Authors: Alexander Caines, Shep Sims, Andrew Taylor
# Description: This file plays either all or a select audio

import numpy as np
import glob as gb
import pydub
from pydub.playback import play

class AUDIOPLAYER(object):
    def __init__(self):
        pass

    def playAudio(self, vector, frame_rate, channels):
        audio = self.toAudio(frame_rate, vector, channels)
        play(audio)

    def main(self):
        if os.path.exists("../songs"):
            mp3_files = gb.glob('../songs/*.mp3')

            for i in range(len(mp3_files)):
                if mp3_files[i]:
                    vector = np.loadtxt(str(mp3_files[i])[9:]+".txt")
                    self.playAudio(100, vector, 2)

if __name__ == "__main__":
    AUDIOPLAYER().main()
