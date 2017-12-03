---
layout: post
published: true
title: Deep Quartet
---
# Music Generation with Deep Neural Networks

## By Karl Konz

The first time I heard the sample output of the deepBach project from Sony CSL I distinctingly remember getting goosebumps and a bit overwhelmed by how compeling the music that had been produced was to me. Of course if you go to their youtube post you will see a variety of responses along the lines of 'the "author" didn't take blah blah blah into account', everyone is an expert. ;) But the benchmarking used for the majority of bach AI models speaks for itself. For the first time, in the bass/lowest voice, experts could we're dupped more than not into believing that the model output was actually bach! 
...look up....

 Having perfor There is something incredibly special about how music compliments himstory. One of my favorite Beethoven symphonies, #3, in its inception praised Neapoleon as Ludwig believed he embodied the democratic and anti-monarchical ideals of the French Revolution, but desavowed himself from Neapoleon by the time of its completion. Or the tight rope that Shostakovich walked having received immediate celebrity status for his first symphony, which he composed while at uni


Having competed for principal and section percussion auditions, bach was frequently an item on the audition list.

Inspiration

I have always

This post will expand on the research done for the DeepBach project from Sony CSL and broaden the corpus to not only Bach Chorales but also include string quartets scrapped from the web site KunstDerFuge with Selenium using the Python wrapper.

Inline:
![alt text](/img/DeepBachImg1.jpg "Deep Bach")
)



The base version of this code is available here:...

To learn more and to listen to sample output from the original project click the link [HERE](http://www.flow-machines.com/deepbach-polyphonic-music-generation-bach-chorales/).


To scrape the additional files for the training corpus follow my github repo here:...
To gather the data I wanted for this project I registered a paid account to [Kunst Der Fuge](http://kunstderfuge.com/) and used [Selenium](http://selenium-python.readthedocs.io/) to authenticate into the site and grab the files for this project programmatically. The code for how I gathered the dataset is available HERE TODO!!! Update the Selenium script. In addition to the string quartets I also downloaded a Kashmir and Going to California from Led Zeppelin to use for reharmonizing.

An easy way of running this code is to launch an Azure Deep Learning VM, clone the repo, and start training away! I have had issues running the code and chose to hard code the pickle path for custom datasets in my version of the repo.

# Data

Starting with Bach to train music makes a lot of sense to me. Prior to working in data science I actually was an orchestral musician having received my masters degree at CSU studying under world renown percussionist/teacher/inventor Tom Freer. Through out my undergraduate and graduate degrees the music theory of Bach was studied quite a bit. Bach had a very rule and pattern based approach to composition through what is commonly referred to as counterpoint. Such rules at paralell fifths where not allowed. This is when 2 notes that are a fifth interval apart, say a c and a g. An example of such a violation would be if you had a chord of c and g on one beat and the next be the voices moved in paralell to a d and a. In my opinion, becuase Bach's approach to composition was more conservative and restrictive than many later composers, it is easier to find a signal from the noise when training a model than compositions from others. Compositional rules and constraints gradually peeled away through the years up to more recent abrast compositions which are not included in the training corpus for this post.

# Modeling 

This approach utilizes Stacked LSTM models and psuedo-gibbs sampling. To learn more, see the research paper from CSL here:

This project utilizes tensorflow and Keras. It embedds the data into a one hot array for the music in a sparse representation. It uses an Adam Optimizer and ... erorr testing (db check cross entropy??)


``` python

```

The core project use the default optimization settings. The learning rate looked to high for me so I lowered it from .001 to .0009. The training results were hitting about 100% with the defaults and about 92% after lowering the learning rate. The testing results were about 84%.

## A. Reharmonizing Bach theme with Chorale/String Chorale corpus

This is a compositional approach that has been around for hundreds of years, in which a composer will that the theme of an eariler peroid composer and reharmonize the general theme of the earlier piece. For the first model we try lets tak
