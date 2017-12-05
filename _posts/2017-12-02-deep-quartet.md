---
layout: post
published: true
title: Deep Quartet
---
# Music Generation with Deep Neural Networks, Bach and Beyond

## By Karl Konz

![alt text](/img/DeepBachImg1.jpg "Deep Bach")


For this post I will explore the limits of the [deepBach](https://arxiv.org/abs/1612.01010) modeling approach to generate music with deep neural networks. The code is open sourced and can be found under [SONY deepBach](https://github.com/Ghadjeres/DeepBach), the Bach chorale corpus for that research is readily availble in the music21 package. I wanted to explore expanding the corpus to include compositions from after the baroque peroid as well. One of the constraints of this modeling techinque is that there has to be the same number of voices for each training composition. The Bach files used for the initial research have four voices so I decided to add string quartets from the classical, romantic, and nationalist periods as they have the same number of voices.


## Additional data, string quartets

In addition to the roughly 350 Bach chorale files from the original research, I am added 189 string quartet midi files from Beethoven, Mozart, Shostakovich, Brahms, Schumann, Schubert, and others. To gather the additional string quartet midi files, I wrote a script to scrape the files I wanted from site [Kunst Der Fuge](http://kunstderfuge.com/) with [Selenium](http://selenium-python.readthedocs.io/) using the Python wrapper to authenticate into the site and download the files of interest. By adding the string quartet files there was a 830% increase in the the count of notes from what was used in the original deepBach study. While there were less files of string quartets, they were generally much longer compositions and thus contained more notes. This is a bit like have pictures of different resolution. I experimented with subsetting the string quartets into smaller files, but that approach did not yeild very complelling or statistically accurate results.


<img src="/img/Bach__Notes.svg" alt="StringNotes" />


<img src="/img/Strings__Notes.svg" alt="StringNotes" />


All of which is included in this repository [Project Build](https://github.com/KKONZ/SpringBoard/tree/master/Capstone%201) which includes the code used to download the files and the code used to conduct the inferential tests. 

# Training the model

The model utilizes stacked lstm models as illustrated in the image below:


![alt text](/img/LSTMref.JPG "Model reference")


I also adjusted the Adam optimizer to slow the learning rate from the default of .001 to .0009, to do this I had to change the source code in the deepBach.py file.I had issues running the source code on my windows machine and ended up having to just hard code the pickled data set path and name instead of using the os python package. The code I used to actually train the AI music embedded below is available [KKONZ deepBach](https://github.com/KKONZ/DeepBach). My version has the custom dataset pickle file location hardcoded as I couldn't get the Sony version to work for that.  If you are using a windows or a linux machine the code below should work for you.
An easy way of running this code is to launch an Azure Deep Learning VM, clone the repo, and start training away! I have had issues running the code and chose to hard code the pickle path for custom datasets in my version of the repo.
Regardless of the platform you choose to use if interested in running this code, you can clone the project while it is on my github page. Note that this version does not use Keras 2 yet.

After cloning the github repository you will also need to download a couple libraries

[music21](http://web.mit.edu/music21/) and [tqdm](https://pypi.python.org/pypi/tqdm):

```python
git clone "http://github.com/kkonz/DeepBach"
pip install music21
pip install tqdm
```

Then using the following parameters, you can adjust the following out of the box. To adjust optimizer settings you will need to do so in the deepBach.py file.

```
usage: deepBach.py [-h] [--timesteps TIMESTEPS] [-b BATCH_SIZE_TRAIN]
                   [-s SAMPLES_PER_EPOCH] [--num_val_samples NUM_VAL_SAMPLES]
                   [-u NUM_UNITS_LSTM [NUM_UNITS_LSTM ...]] [-d NUM_DENSE]
                   [-n {deepbach,skip}] [-i NUM_ITERATIONS] [-t [TRAIN]]
                   [-p [PARALLEL]] [--overwrite] [-m [MIDI_FILE]] [-l LENGTH]
                   [--ext EXT] [-o [OUTPUT_FILE]] [--dataset [DATASET]]
                   [-r [REHARMONIZATION]]

optional arguments:
  -h, --help            show this help message and exit
  --timesteps TIMESTEPS
                        models range (default: 16)
  -b BATCH_SIZE_TRAIN, --batch_size_train BATCH_SIZE_TRAIN
                        batch size used during training phase (default: 128)
  -s SAMPLES_PER_EPOCH, --samples_per_epoch SAMPLES_PER_EPOCH
                        number of samples per epoch (default: 89600)
  --num_val_samples NUM_VAL_SAMPLES
                        number of validation samples (default: 1280)
  -u NUM_UNITS_LSTM [NUM_UNITS_LSTM ...], --num_units_lstm NUM_UNITS_LSTM [NUM_UNITS_LSTM ...]
                        number of lstm units (default: [200, 200])
  -d NUM_DENSE, --num_dense NUM_DENSE
                        size of non recurrent hidden layers (default: 200)
  -n {deepbach,skip}, --name {deepbach,skip}
                        model name (default: deepbach)
  -i NUM_ITERATIONS, --num_iterations NUM_ITERATIONS
                        number of gibbs iterations (default: 20000)
  -t [TRAIN], --train [TRAIN]
                        train models for N epochs (default: 15)
  -p [PARALLEL], --parallel [PARALLEL]
                        number of parallel updates (default: 16)
  --overwrite           overwrite previously computed models
  -m [MIDI_FILE], --midi_file [MIDI_FILE]
                        relative path to midi file
  -l LENGTH, --length LENGTH
                        length of unconstrained generation
  --ext EXT             extension of model name
  -o [OUTPUT_FILE], --output_file [OUTPUT_FILE]
                        path to output file
  --dataset [DATASET]   path to dataset folder
  -r [REHARMONIZATION], --reharmonization [REHARMONIZATION]
                        reharmonization of a melody from the corpus identified
                        by its id
```


I then added a midi file of the Led Zeppelin track Kashmir and reharmonized the model to that track and going to california as well, both were trained in the same manner as the code above, but were delibrately named in a way to be indexed in the first position. Here are sample outputs from those models:

## Results:

Kashmir:

<iframe width="100%" height="394" src="https://musescore.com/user/27137243/scores/4839364/embed" frameborder="0" allowfullscreen></iframe><span><a href="https://musescore.com/user/27137243/scores/4839364">Deep Kashmir</a> by <a href="https://musescore.com/user/27137243">Konzert</a></span>

Going to California:

<iframe width="100%" height="394" src="https://musescore.com/user/27137243/scores/4839372/embed" frameborder="0" allowfullscreen></iframe><span><a href="https://musescore.com/user/27137243/scores/4839372">Deep Going To California</a> by <a href="https://musescore.com/user/27137243">Konzert</a></span>






