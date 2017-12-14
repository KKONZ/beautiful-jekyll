---
layout: post
published: true
title: Deep Quartet
---
# Music Generation with Deep Neural Networks, Bach and Beyond

## By Karl Konz

![alt text](/img/DeepBachImg1.jpg "Deep Bach")


For my first data science post I really wanted to do a music related project, a subject very dear to me. I grew up in a musical family, my mother was a musician and all of the kids in my family played musical instruments. Prior to becoming a lawyer, my brother earned a bachelors degree in music. I took it a step further and earned a masters degree before looking to greener pastures for a day job. I was lucky enough to study at one of the finest orchestral graduate programs in the world, CSU. Most of my classmates during that time now have big orchestra jobs such as the metropolitan opera and the oregon symphony. The program was very structured and very much like the Steve Jobs quote that everyone should learn computer programming or law because it teaches you how to think, well this program seemed to fall into that category for me. I am convinced that having studied under Tom Freer laid the foundation for me to have the capacity to do well in data science. I have done a couple data science projects with music prior to this project, but this one has been by far the most fulfilling. 

***For this post I will explore the limits of the [deepBach](https://arxiv.org/abs/1612.01010) modeling approach to generate music with deep neural networks.***

The code is open sourced and can be found under [SONY deepBach](https://github.com/Ghadjeres/DeepBach), the Bach chorale corpus for that research is readily availble in the music21 package. I wanted to explore expanding the corpus to include compositions from after the baroque peroid as well. One of the constraints of this modeling techinque is that there has to be the same number of voices for each training composition. The Bach files used for the initial research have four voices so I decided to add string quartets from the classical, romantic, and nationalist periods as they have the same number of voices.

## Additional data, string quartets

In addition to the roughly 350 Bach chorale files from the original research, I am added 189 string quartet midi files from Beethoven, Mozart, Shostakovich, Brahms, Schumann, Schubert, and others. To gather the additional string quartet midi files, I wrote a script to scrape the files I wanted from site [Kunst Der Fuge](http://kunstderfuge.com/) with [Selenium](http://selenium-python.readthedocs.io/) using the Python wrapper. By adding the string quartet files there was a 830% increase in the the count of notes from what was used in the original deepBach study. So there was just over half the number of string quartet files as Bach files, however the string quartet files contain much more notes. I experimented with subsetting the string quartets into smaller files to sort of standardize the size of the files, but that approach did not yeild very complelling or statistically accurate results. I also conducted a z test to check the proportion of the notes in terms of frequency and note duration combinations as plotted below, the p value from the test was 7x10-43, thus there is evidence of a difference in the proportion of notes between the Bach chorales and the new string quartets I added. We can see that notes with smaller durations are more common in the string quartet files than in the Bach Chorale files. In other words, it is more common to have faster notes in the string quartets than in the Bach compositions.


<img src="/img/Bach__Notes.svg" alt="StringNotes" />


<img src="/img/Strings__Notes.svg" alt="StringNotes" />


All of which is included in this repository [Project Build](https://github.com/KKONZ/SpringBoard/tree/master/Capstone%201) which includes the code used to download the files and the code used to conduct the inferential tests. 

# Training the model

The first step in training this model is to transform the data into different one-hot encodings. This is the process of transforming the data into binary vectors where each categorical value is mapped to integer values. Each vector will be all zeros except the index with the given value which is represented as a 1. The code block below is borrowed from the [tensorflow]("https://www.tensorflow.org/api_docs/python/tf/one_hot") website.

```python
indices = [0, 1, 2]
depth = 3
tf.one_hot(indices, depth)  # output: [3 x 3]
# [[1., 0., 0.],
#  [0., 1., 0.],
#  [0., 0., 1.]]

indices = [0, 2, -1, 1]
depth = 3
tf.one_hot(indices, depth,
           on_value=5.0, off_value=0.0,
           axis=-1)  # output: [4 x 3]
# [[5.0, 0.0, 0.0],  # one_hot(0)
#  [0.0, 0.0, 5.0],  # one_hot(2)
#  [0.0, 0.0, 0.0],  # one_hot(-1)
#  [0.0, 5.0, 0.0]]  # one_hot(1)

indices = [[0, 2], [1, -1]]
depth = 3
tf.one_hot(indices, depth,
           on_value=1.0, off_value=0.0,
           axis=-1)  # output: [2 x 2 x 3]
# [[[1.0, 0.0, 0.0],   # one_hot(0)
#   [0.0, 0.0, 1.0]],  # one_hot(2)
#  [[0.0, 1.0, 0.0],   # one_hot(1)
#   [0.0, 0.0, 0.0]]]  # one_hot(-1)

```

The model utilizes stacked lstm models as illustrated in the image below:


![alt text](/img/LSTMref.JPG "Model reference")


I had issues running the source code on my windows machine and ended up having to just hard code the pickled data set path and name instead of using the os python package. The code I used to actually train the AI music embedded below is available [KKONZ deepBach](https://github.com/KKONZ/DeepBach). My version has the custom dataset pickle file location hardcoded as I couldn't get the Sony version to work for that.  If you are using a windows or a linux machine the code below should work for you.
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


## Conclusions:
When choosing a file to reharmonize, it seemed that  files where all of the voices are moving train more compelling output than when the voices more or less just outline the harmonic changes in block chords. Overall I was happy with the output from this modeling technique and this has been an incredibly rewarding project to work on.
