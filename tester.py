# Authors: Alexander Caines, Shep Sims, Andrew Taylor
# Description: This file retrieves and preprocesses song data for beat generation

import datetime
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
##        print("Channels: " , a.channels, "\nDuration: ", a.duration_seconds, "\nSample Width: " , a.sample_width, "\nFrame Width: " , a.frame_width)
        y = np.array(a.get_array_of_samples())
##        print(y[200])
        
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

##    def play_results(self, model, data, batch_size = 128):
##        filename = str(datetime.datetime.now)
##        encoder, decoder = model
##        x_test = data
##
##        z_mean, _, _ = encoder.predict(x_test, batch_size = batch_size)
##
##        # displays something about the latent space
##        plt.figure()
##        plt.scatter(z_mean[:,0], z_mean[:,1], c = y_test)
##        plt.colorbar()
##        plt.xlabel("z[0]")
##        plt.ylabel("z[1]")
##        plt.savefig(filename)
##        plt.show()
##
##        ##return and play audio
##        #audio = decoder.predict()
##        #playAudio(audio)
        
    def main(self):
        if os.path.exists('../songs'):
            mp3_files = gb.glob('../songs/*.mp3') #list of mp3 file addresses in a folder called songs sitting outside of this directory
            count = 0
            #for i in range(len(mp3_files)): #uncomment for submission
            for i in range(1): #for testing
                frame_rate, channels, vector = self.transformData(mp3_files[i]) #Note, the framerate is in milliseconds
                self.tensor = np.append(self.tensor, vector)
                self.frame_rates = np.append(self.frame_rates, frame_rate)
                self.channels = np.append(self.channels, channels)
                count+=1
                print("loaded", str(count)+str("/")+str(len(mp3_files)))
                #filename = str(mp3_files[i])[9:] + ".txt"
                #self.writeFile(vector, filename, "../vectorizedAudio") #should be a global array
                #self.playAudio(vector, frame_rate, channels)
        else:
            f = "Hip Hop SFX.mp3"
            #the following returns an np array (vector) representing one mp3 file
            frame_rate, channels, vector = self.transformData(f) #framerate is in milliseconds
            self.tensor = np.append(self.tensor, vector)
            self.frame_rates = np.append(self.frame_rates, frame_rate)
            self.channels = np.append(self.channels, channels)
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
            #self.playAudio(audio_decompressed)
            #self.playAudio(audio_decompressed2)

        
        #Hyper Paramters for model: 
        original_dim = 1 #3 #264600 # currently set to 3s of audio
        input_shape = (original_dim, )
        intermediate_dim = 512
        batch_size = 128
        latent_dim = 2
        epochs = 1
        training_data = self.tensor
        print(type(self.tensor))
        print(len(self.tensor))
        print(self.tensor[0])
        #Build encoder model:
        inputs = Input(shape = input_shape, name = 'encoder_input')
        x = Dense(intermediate_dim, activation='relu')(inputs)
        z_mean = Dense(latent_dim, name='z_mean')(x)
        z_log_var = Dense(latent_dim, name='z_log_var')(x)

        # use reparameterization trick to push the sampling out as input
        # note that "output_shape" isn't necessary with the TensorFlow backend
        input_tensor = Concatenate(axis = -1)([z_mean, z_log_var])
        # z = Lambda(self.sampling)([self,input_tensor])
        z = Lambda(self.sampling, output_shape=(latent_dim,), name = 'z')([z_mean, z_log_var])
        # z = lambda z_mean, z_log_var: self.sampling
        
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
        vae.summary()

        # Train the model:
        vae.fit(x =training_data, epochs=epochs, batch_size=batch_size)
        vae.summary()

        #generate
        prediction = (2**15)*(vae.predict(x = training_data, batch_size = batch_size))
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
