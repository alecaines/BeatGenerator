# BeatGenerator
## Authors
Shep Sims, Andrew Taylor, Alexander Caines 

## Objective
   This project creates a generative model for songs given a certain genre. Specifically, we wish to generate beats mimicing those typically used by rappers during production, as they are not inheritly tied to the lyics imposed on top of them. The project was deemed feasible as many scholarly articles and papers have been published on the subject. These sources proved incredibly useful to us as we decided on the finer points of our model implementation. We in particular found Roberts, Engel, and Eck’s paper on applying hierarchical variational autoencoders to music helpful when we made design choices for our network. We also consulted Kotecha and Young’s paper on generating music with LSTMs when we made high level network structure decisions. Notably we did not incorporate an LSTM in the final version of our model but we considered it based on the results published by Kotecha and Young. Lastly, we took some inspiration for the core idea of our project from Tero Parviainen’s freely available Neural Drum Machine. 

## Introduction
Rap beats often are short samples of “licks” from different individual instruments repeated throughout the course of the song. We wish to produce a sound byte that mimics these licks for a given instrument over a predetermined set of meters.###

## Preparing Environment
   In order to run the model, ffmpeg must be installed on the system. For Windows machines, this video https://www.youtube.com/watch?v=w1q7POTlJeY provides clear instructions for installing ffmpeg. The builds for all machines can be found  at http://www.ffmpeg.org/download.html but the necessary preparations for linux and mac are unknown to me (Alexander Caines).

## Running the program
   Clone the code from the master branch to a directory. Make sure that in the directory above the cloned repository there is a folder called songs containing inputs. We provide a training set of 33 songs in a zipped folder along with the submission called 'songs' that we recommend placing here to use for training. 
To run the program, open a fresh terminal window and navigate to the directory where the cloned repository is located and type "python3 tester.py". The result of the training will be saved to an mp3 file called generated_music.mp3 within this directory. 

## Lambdas branch
   The lambdas branch contains a cleaner training datset than the main branch. However, the dimensionality of each input is too large enough for weak machines to process, resulting in python killing the model during training. 

## Target Function
   The target function for this project would be a sound byte from an ideal instrument’s lick. The methodology of the lick would revolve around comparing the original input and the predicted output. The extent to which style transfer was a success determines the accuracy of the model.
   
   #New but check me on this
   Autoencoder models generally attempt to reconstruct input examples as output.  Here, the target function attempts this by measuring the difference between the encoded input audio and output, and aims to minimize this reconstruction error.  

## Libraries
   PyDub: Provides methods for powerful transformation and partitioning of audio files, used here for data-processing.

## Datasets
   There are many massive datasets online that provide genre or instrument specific tracks that could be used to train the model, however, many of these datsets contain copyrights or other restrictive permissions for use, or the filetypes are not easily handled by python.  As such, we use a set of mp3 audio files gathered from a distributor of copyright free beats on youtube, user "heroboard - Music for Creators." The training set consisted of 27 of the 33 songs we had and the testing set consisted of the remaining 6 songs.  

   
## Preprocessing
   The data comes in as an mp3 file which python does not know how to handle, so we provide a transformData method to handle this.  This method accepts an mp3 filepath, and using the pydub library creates a vectorized representation of the audio segment from it.  This representation is truncated to the first 3s of audio as larger representations do not allow running # ! $ ! $ ^ &

   It should be noted that the output of our model is not consistent with the goals of the project. The output the model returned was an mp3 file with nothing but white noise as its contents. The corrupted state of the data was due to the state of the inputs. When constructing the training dataset, we appended the transformed audio data to a numpy array. Rather than appending it to a list, it concatenated each subsequent datum with the previous--creating one large input matrix. Upon recommendation from Professor Watson, we attempted to implement tensors rather than numpy arrays. However, we encountered errors with the dimensionality of each input--as they were inconsistent. This was due to the way that pydub extracts audio data from a file. Given that each song has a different sample rate, the number of samples (elements of the audio matrix) differs between song, even when holding the length of the song constant (3000 milliseconds in our implementation). There were attempts made at padding the data after it had been collected. This involved adding each audio datum to a list (which would not concatenate the data), finding the datum with the largest size, and then padding with respect to that using np.pad. This attempt can be seen in the lambdas branch. However, due to the enormity of the datum's dimensionality, the python killed the process during the first epoch.
   
## Model Architecture 

Our model follows the typical structure of an auto-encoder.  We feed the preprocessed, vectorized audio into a neural network serving as an encoder to generate latent representations of the data.  We then pass the significantly reduced representation to another neural network serving as a decoder which seeks to reconstruct the encoder's input. Both the encoder and decoder are standalone neural networks consisting of two fully connected dense layers and an input layer (vectorized audio files for the encoder and latent representations for the decoder). 

vectorized representation -> Encoder -> latent representation -> Decoder -> vectorized representation 


The canonical optimizer for variational autoencoders is typically regarded to be the adam optimizer, as it combines many of the advantages of other optimizers. It also decreases training time by continuously updating the learning rate as training persists by using the moving average of the stochastic gradient, resulting in better training times for nearly all datasets, especially when training momentum decreases over time (as in large datasets). As adam has proven to work well for messy data sets (or data sets of inconsistent form) and our audio data was inconsistent (meaning that the same musical elements or styles were not present across the entire dataset), adam seemed like the obvious best choice for our optimization function. 
For our loss function we choose to use mean squared error as this allows for reliable computation of the difference in predicted and actual output. We also considered using cross-entropy loss but decided against it because the task at hand was not to categorize the audio but produce audio similar to that of the training set. 

Our model implements a random noise seed in the encoding process to create variance in the latent representation before being fed into the decoder. This noise vector is created by sampling the distribution of the data at the output of the encoder layer, which is commonly referred to as the "bottleneck" as it is the point at which the minimum representation of the input lies. In sampling the bottleneck we determine the probability distribution by calculating the average value and the log variance of the bottlenecked representation.  We then 

HERE’S WHERE WE NEED TO TALK ABOUT THE KL_LOSS TERM. OUR CODE THEN DEFINES A “VAE_LOSS” THAT EQUALS “K.mean(reconstruction_loss + kl_loss)”. This combined loss then gets added to the vae in line 190: vae.add_loss(vae_loss). I’m not sure how to talk about the meaning/reasoning of combining these two loss functions.   

## Sources:
Libraries  
https://pydub.com/  
https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/    
  
Data  
https://www.youtube.com/channel/UClWu1Gr3TVsJkZwXFKfG0fg  
Assistance  
https://stackoverflow.com/questions/53633177/  how-to-read-a-mp3-audio-file-into-a-numpy-array-save-a-numpy-array-to-mp3?noredirect=1&lq=1  
(some pydub) https://www.youtube.com/watch?v=4E7N7W1lUkU  
(installing ffmpeg) https://www.youtube.com/watch?v=w1q7POTlJeY    
  
Additional Resources  
https://www.one-tab.com/page/WB4AxkPYQtuSG4UeNKEAMw
https://nips2017creativity.github.io/doc/Hierarchical_Variational_Autoencoders_for_Music.pdf  
https://keras.io/examples/variational_autoencoder/  
https://blog.keras.io/building-autoencoders-in-keras.html - Variational AutoEncoder Example
https://github.com/keras-team/keras/blob/master/examples/variational_autoencoder.py - Variational AutoEncoder Example (code)
https://www.reddit.com/r/WeAreTheMusicMakers/comments/3ajwe4/the_largest_midi_collection_on_the_internet/  
https://codepen.io/teropa/details/JLjXGK  
https://github.com/tensorflow/magenta/tree/master/magenta/models/drums_rnn  
https://arxiv.org/ftp/arxiv/papers/1804/1804.07300.pdf  
https://blog.goodaudience.com/using-tensorflow-autoencoders-with-music-f871a76122ba  


