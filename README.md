# BeatGenerator
## Authors
Shep Sims, Andrew Taylor, Alexander Caines 

## Objective
The objective of this project create a generative model for songs given a certain genre. Specifically, we wish to generate beats typically used by rappers during production. The project was deemed feasible as there exist a myriad of published papers and individual projects concerning the matter over a variety of different genres. We are not interested in producing symphonic or drawn-ought instrumentals. Rap beats often are short samples of “licks” from different individual instruments repeated throughout the course of the song. We wish to produce a sound byte that mimics these licks for a given instrument over a predetermined set of meters.

## Model
After reviewing different papers on music generation, we found that the most commonly used networks were long short term memory, generative adversarial networks, and recurrent neural networks. The models that seemed to yield the highest precision and accuracy (given the objectives set out by the individual researchers) were CNN’s and different variations of LSTM’s. These networks, however, were used to generate--as we mentioned before--sound-bytes that falls under the same genre as symphonic music. Because of the relatively small length of the sound-bytes and their desired highly repetitive nature, we will set out to implement a canonical recurrent neural net. We will use Tensorflow as the framework for implementing the model.

## Datasets
There are many massive datasets online that provide genre or instrument specific tracks that could be used to train the model.  One such dataset is the “Drum_space” dataset on github which provides 33,000 unique drum beats which has been used to generate new beat tracks by user altsoph.  Should we find the generation of beats more easily implementable than planned, there also exists the “Million Song Dataset” on Kaggle which provides 50,000 genre-labeled audio tracks which could be trained to produce genre-specific tracks rather than beats only. 

## Target Function
The appropriate target function for this project would be a sound byte from an ideal instrument’s lick. The methodology of the lick would revolve around comparing the original input and the predicted output. The extent to which style transfer was a success determines the accuracy of the model.
