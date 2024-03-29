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
        self.tensor = np.array([]) 
        self.frame_rates = np.array([])
        self.channels = np.array([])
        self.sample_size = 90000
        self.epochs = 500
        self.intermediate_dim = 512
        self.batch_size = 128
        self.latent_dim = 2

    # converts mp3 to numpy array
    def transformData(self, f, t = 3):
        duration = t*1000
        #retrieves audio
        if type(f) != pydub.audio_segment.AudioSegment:
            a = pydub.AudioSegment.from_mp3(file = f)[:duration].set_channels(1)
        else:
            a = f

        samples = a.get_array_of_samples()[:self.sample_size]
        # converts mp3 data to numpy array
        y = list(map(lambda sample:sample/2**15,samples))
        
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
            data = []
            for i in range(len(mp3_files)): #uncomment for submission
                frame_rate, channels, vector = self.transformData(mp3_files[i]) #Note, the framerate is in milliseconds
                data.append(vector)
                self.frame_rates = np.append(self.frame_rates, frame_rate)
                self.channels = np.append(self.channels, channels)
                count+=1
                print("loaded", str(count)+str("/")+str(len(mp3_files)))
            minn = min(list(map(lambda datum: len(datum), data))) #finds the min size
            data = list(map(lambda datum: datum[:minn], data))
            self.tensor = np.array(data)  
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
        sample_size = self.sample_size #238464#1 #7570944 #3 #264600 # currently set to 3s of audio
        input_shape = (sample_size, )
        intermediate_dim = self.intermediate_dim
        batch_size = self.batch_size
        latent_dim = self.latent_dim
        epochs = self.epochs
        training_data = self.tensor[:28]
        testing_data = self.tensor[28:]

        #Build encoder model:
        inputs = Input(shape = input_shape, name = 'encoder_input')
        x = Dense(intermediate_dim, activation='relu')(inputs)
        x = Dense(intermediate_dim, activation='relu')(x)
        x = Dense(intermediate_dim, activation='relu')(x)
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
        x = Dense(intermediate_dim, activation='relu')(x)
        x = Dense(intermediate_dim, activation='relu')(x)
        outputs = Dense(sample_size, activation='sigmoid')(x)

        # instantiate decoder model
        decoder = Model(latent_inputs, outputs, name='decoder')
        decoder.summary()

        # instantiate VAE model
        outputs = decoder(encoder(inputs)[2])
        vae = Model(inputs, outputs, name='vae_mlp')

        reconstruction_loss = mse(inputs, outputs)
        reconstruction_loss *= sample_size
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        vae.add_loss(vae_loss)
        vae.compile(optimizer='adam')
        vae.summary()

        # Train the model:
        # Need to seperate out separate songs in the input.
        #log_dir="../logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        #tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir = log_dir, write_graph = True ,write_grads = True, histogram_freq = 1)
        vae.fit(x =training_data, epochs=epochs, batch_size=batch_size)#, callbacks = [tensorboard_callback])
        vae.summary()

        #generate
        prediction = (2**15)*(vae.predict(x = testing_data, batch_size = batch_size))
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
