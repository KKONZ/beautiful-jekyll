---
layout: post
published: true
title: Deep Quartet
---
# Music Generation with Deep Neural Networks, Bach and Beyond

## By Karl Konz

![alt text](/img/DeepBachImg1.jpg "Deep Bach")


For this post I will explore the limits of the [deepBach](http://www.flow-machines.com/deepbach-polyphonic-music-generation-bach-chorales/) modeling approach to generate music with deep neural networks. The benchmarking used for the majority of bach AI models speaks for itself with this approach. For the first time as far as I know, in the bass/lowest voice, experts we're dupped more than not into believing that the model output was actually composed by Bach! 
...look up....


## Additional data, string quartets

One of the constraints of this modeling techinque is that there has to be the same number of voices for each training composition. The Bach files used for the initial research have four voices. In addition to these files I am adding string quartet midi files from ... look up all of the composers downloaded, resulting in an increased corpus of roughly ~%.

To gather the additional string quartet midi files, I wrote a script to scrape the files I wanted from site [Kunst Der Fuge](http://kunstderfuge.com/) with [Selenium](http://selenium-python.readthedocs.io/) using the Python wrapper to authenticate into the site and download the files of interest.

The original source code from deepBach is available [here]. 
In addition to the string quartets I also downloaded a Kashmir and Going to California from Led Zeppelin to use for reharmonizing.

An easy way of running this code is to launch an Azure Deep Learning VM, clone the repo, and start training away! I have had issues running the code and chose to hard code the pickle path for custom datasets in my version of the repo.

# Data

Starting with Bach to train music makes a lot of sense to me. Bach's approach to composition was more conservative and restrictive than many later composers. Compositional rules and constraints gradually peeled away through the years up to more recent abrast compositions which are not included in the training corpus for this post. There is a lot of overlap from the roughly 150-200 year peroid of music that I have included in terms of theory so I expect for the output to be be compelling and different than that of just training on Bach.

# Modeling 

This approach utilizes Stacked LSTM models and psuedo-gibbs sampling. To learn more, see the research paper from CSL here:

This project utilizes tensorflow and Keras. It embedds the data into a one hot array for the music in a sparse representation. It uses an Adam Optimizer and ... erorr testing (db check cross entropy??)


``` python
git clone 'http://github.com/kkonz/DeepBach
```

After navigating to the new created deepBach directory, which created the first model for my first example below the code

```python
python3 deepBach.py --dataset datasets/custom_dataset --ext ReharmonizeBachxxxx - 200 200 -d 200 -t 10 -p -i 40000 -r 45 -o 'ReharmBWVxxx.mid'
```

Which produces the output below:

...youtube

Next I added Kashmir led zeppelin track and reharmonized the model to that track and going to california as well, both were trained in the same manner as the code above, but were delibrately named in a way to be indexed in the first position. Here are sample outputs from those models:

Kashmir:


Going to California:


For this corpus, the default learning rate in the deepBach source code was too high, lower to .0009 from .001 acheived better results.



