# BeatGenerator
## Authors
Shep Sims, Andrew Taylor, Alexander Caines 

## Objective
The objective of this project create a generative model for songs given a certain genre. Specifically, we wish to generate beats typically used by rappers during production. The project was deemed feasible as there exist a myriad of published papers and individual projects concerning the matter over a variety of different genres. We are not interested in producing symphonic or drawn-ought instrumentals. Rap beats often are short samples of “licks” from different individual instruments repeated throughout the course of the song. We wish to produce a sound byte that mimics these licks for a given instrument over a predetermined set of meters.

## Model
After reviewing different papers on music generation, we found that the most commonly used networks were long short term memory, generative adversarial networks, and recurrent neural networks. The models that seemed to yield the highest precision and accuracy (given the objectives set out by the individual researchers) were CNN’s and different variations of LSTM’s. These networks, however, were used to generate--as we mentioned before--sound-bytes that falls under the same genre as symphonic music. Because of the relatively small length of the sound-bytes and their desired highly repetitive nature, we will set out to implement a canonical recurrent neural net. We will use Tensorflow as the framework for implementing the model.

## Preparing Environment
In order to run the model, ffmpeg must be installed on the system. For Windows machines, this video https://www.youtube.com/watch?v=w1q7POTlJeY provides clear instructions for installing ffmpeg. The builds for all machines can be found  at http://www.ffmpeg.org/download.html but the necessary preparations for linux and mac are unknown to me (Alexander Caines).

## Libraries
Because pydub allows for transformation and partitioning of audio files, we use it for data-processing.

## Datasets
There are many massive datasets online that provide genre or instrument specific tracks that could be used to train the model.  One such dataset is the “Drum_space” dataset on github which provides 33,000 unique drum beats which has been used to generate new beat tracks by user altsoph.  Should we find the generation of beats more easily implementable than planned, there also exists the “Million Song Dataset” on Kaggle which provides 50,000 genre-labeled audio tracks which could be trained to produce genre-specific tracks rather than beats only. 

## Sources:
Libraries  
https://pydub.com/  
https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/    
  
Data  
https://unbelievablebeats.com/free-beats-free-downloads  
  
Assistance  
https://stackoverflow.com/questions/53633177/  how-to-read-a-mp3-audio-file-into-a-numpy-array-save-a-numpy-array-to-mp3?noredirect=1&lq=1  
(some pydub) https://www.youtube.com/watch?v=4E7N7W1lUkU  
(installing ffmpeg) https://www.youtube.com/watch?v=w1q7POTlJeY    
  
Additional Resources  
https://nips2017creativity.github.io/doc/Hierarchical_Variational_Autoencoders_for_Music.pdf  
https://keras.io/examples/variational_autoencoder/  
https://blog.keras.io/building-autoencoders-in-keras.html - Variational AutoEncoder Example
https://github.com/keras-team/keras/blob/master/examples/variational_autoencoder.py - Variational AutoEncoder Example (code)
https://www.reddit.com/r/WeAreTheMusicMakers/comments/3ajwe4/the_largest_midi_collection_on_the_internet/  
https://codepen.io/teropa/details/JLjXGK  
https://github.com/tensorflow/magenta/tree/master/magenta/models/drums_rnn  
https://arxiv.org/ftp/arxiv/papers/1804/1804.07300.pdf  
https://blog.goodaudience.com/using-tensorflow-autoencoders-with-music-f871a76122ba  



## Target Function
The appropriate target function for this project would be a sound byte from an ideal instrument’s lick. The methodology of the lick would revolve around comparing the original input and the predicted output. The extent to which style transfer was a success determines the accuracy of the model.
