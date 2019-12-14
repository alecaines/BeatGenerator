# Authors: Alexander Caines, Shep Sims, Andrew Taylor
# Description: This file retrieves and preprocesses song data for beat generation

import datetime
import tensorflow as tf 
import matplotlib.pyplot as plt
import pydub #allows for manipulation of audio
from pydub.playback import play
import numpy as np
import glob as gb # glob lists the files of a certain filetype in a folder specified by the programmer
import os

from keras.layers import Lambda, Input, Dense, Concatenate
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K

class BEATGENERATOR(object):


    def __init__(self):
        self.tensor = np.array([]) # ndmin = 2) #Creates a 2D array where each row would represent 1 input song
        self.frame_rates = np.array([]) # Frame_rates and channels would remain as 1D arrays, where each index is an input song.
        self.channels = np.array([])

        # tf.Tensor version:
        #self.tensor = tf.constant([])
        # Also tf.stack() might work for self.tensor. 
        #self.frame_rates = tf.constant([])
        #self.channels = tf.constant([])

        #lists can hold other lists of inconsistent dimensions as elements whereas numpy arrays cannot
        #self.tensor = []
        #self.frame_rates = []
        #self.channels = []

    # converts mp3 to numpy array
    def transformData(self, f):
        
        #retrieves audio
        if type(f) != pydub.audio_segment.AudioSegment:
            a = pydub.AudioSegment.from_mp3(file = f).set_channels(1)
        else:
            a = f
        if a.channels == 2:
            y = y.reshape((-1,2))

        # converts mp3 data to numpy array
        y = list(map(lambda x:x/2**15, a.get_array_of_samples()))
        #y = np.array(a.get_array_of_samples()) #, ndmin = 2)
        #y = np.float32(y)/2**15
        
        # tf.Tensor version:
        #y = tf.convert_to_tensor(y, dtype = tf.float32)
        
        #normalizes data and puts it into an np array:
        #y = np.array(list(map(lambda x:x/(2**15), a.get_array_of_samples())))

        return a.frame_rate, a.channels, y
    
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
##        print(signal.shape)
##        channel1 = signal
##        channel2 = signal[:,1]
        audio_segment = pydub.AudioSegment(
            signal.tobytes(),
            frame_rate = rate,
            sample_width = signal.dtype.itemsize,
            channels = channels
        ) + 6
        return audio_segment

    def playAudio(self, audio_segment):
        play(audio_segment)

    def sampling(self,args):
        
        z_mean, z_log_var = args

        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        # by default, random_normal has mean = 0 and std = 1.0
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon
        
    def main(self):
        if os.path.exists('../songs'):
            mp3_files = gb.glob('../songs/*.mp3') #list of mp3 file addresses in a folder called songs sitting outside of this directory
            count = 0
            #print(mp3_files)
            #for i in range(len(mp3_files)): #uncomment for submission
            data = list(map(lambda x: self.transformData(x), mp3_files[:3]))   
            minn = min(list(map(lambda datum: len(datum[2]), data)))
            training_data = list(map(lambda datum: datum[2][:minn], data))
            
            self.tensor = tf.constant(training_data)
            self.channels = list(map(lambda datum: (datum[1]), data))
            self.frame_rates = list(map(lambda datum: (datum[0]), data))

            print(self.tensor[0].shape)
            
##            for i in range(2): #for testing
##                print("count: ", count)
##                frame_rate, channels, vector = self.transformData(mp3_files[i]) #Note, the framerate is in milliseconds
##                print(vector[0])
##                minn = min(list(map(lambda datum: len(datum), vector)))
##                vector = list(map(lambda  datum: datum[:minn], vector))
##
##                #nparray implementation:
##                self.tensor = np.append(self.tensor, vector, axis = 0)
##                self.frame_rates = np.append(self.frame_rates, frame_rate)
##                self.channels = np.append(self.channels, channels)
##
##                #Tensor implementation:
##                #self.tensor = tf.concat(self.tensor, vector, axis = 0)
##                #self.frame_rates = tf.concat(self.frame_rates, frame_rate)
##                #self.channels = tf.concat(self.channels, channels)
##                
##                count+=1
##                print("loaded", str(count)+str("/")+str(len(mp3_files)))
##                #filename = str(mp3_files[i])[9:] + ".txt"
##                #self.writeFile(vector, filename, "../vectorizedAudio") #should be a global array
##                #self.playAudio(vector, frame_rate, channels)
        else:
            f = "Hip Hop SFX.mp3"
            #the following returns an np array (vector) representing one mp3 file
            frame_rate, channels, vector = self.transformData(f) #framerate is in milliseconds
            self.tensor = np.append(self.tensor, vector)
            self.frame_rates = np.append(self.frame_rates, frame_rate)
            self.channels = np.append(self.channels, channels)
            #filename = str(f)+ "test.txt"

            
            audio_decompressed = self.toAudio(frame_rate, vector,channels)
            frame_rate2, channels2, vector2 = self.transformData(audio_decompressed)
            audio_decompressed2 = self.toAudio(frame_rate2, vector2,channels2)

        
        #Hyper Paramters for model: 
        original_dim = 7570944 #3 #264600 # currently set to 3s of audio
        input_shape = (original_dim, )
        intermediate_dim = 512
        batch_size = 128
        latent_dim = 2
        epochs = 1
        training_data = self.tensor
##        print(type(self.tensor))
##        print("length of input: ", len(self.tensor))
##        print("Lengh of 1 song: ", input_length)
##        print(self.tensor[0])
        #Build encoder model:
        inputs = Input(shape = input_shape, name = 'encoder_input')
        x = Dense(intermediate_dim, activation='relu')(inputs)
        z_mean = Dense(latent_dim, name='z_mean')(x)
        z_log_var = Dense(latent_dim, name='z_log_var')(x)

        # use reparameterization trick to push the sampling out as input
        input_tensor = Concatenate(axis = -1)([z_mean, z_log_var])

        z = Lambda(self.sampling, output_shape=(latent_dim,), name = 'z')([z_mean, z_log_var])

        
        # instantiate encoder model     
        encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
        encoder.summary()

        # build decoder model
        latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
        x = Dense(intermediate_dim, activation='relu')(latent_inputs)
        outputs = Dense(original_dim, activation='sigmoid')(x)

        # instantiate decoder model
        decoder = Model(latent_inputs, outputs, name='decoder')
        decoder.summary()

        # instantiate VAE model
        outputs = decoder(encoder(inputs)[2])
        vae = Model(inputs, outputs, name='vae_mlp')

        reconstruction_loss = mse(inputs, outputs)
        reconstruction_loss *= original_dim
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        vae.add_loss(vae_loss)
        vae.compile(optimizer='adam')
        print("here0")
        vae.summary()

        # Train the model:
        # Need to seperate out separate songs in the input.
        log_dir="../logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir = log_dir, write_graph = True ,write_grads = True, histogram_freq = 1)
        vae.fit(x =training_data[:28][:100], epochs=epochs, batch_size=batch_size, callbacks = [tensorboard_callback])
        print("here1")
        vae.summary()

        #generate
        prediction = (2**15)*(vae.predict(x = training_data[28:][:100], batch_size = batch_size))
        print("here2")
        print(type(prediction))
        print(len(prediction))
        print(prediction)

        #results
        audio = self.toAudio(self.frame_rates[0], prediction, 1) #get this to work for each element in the training set
        
        #store audio
        filename = str(datetime.datetime.now)+".mp3"
        audio.export("generated_music.mp3", format = "mp3")
        
if __name__ == "__main__":
    BEATGENERATOR().main()
