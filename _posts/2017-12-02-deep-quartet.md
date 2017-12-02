---
layout: post
published: true
title: Deep Quartet
---
# Music Generation with Deep Neural Networks

## Inspiration 

This post will expand on the research done for the DeepBach project from Sony CSL and broaden the corpus to not only Bach Chorales but also include string quartets scrapped from the web site KunstDerFuge with Selenium using the Python wrapper.




The base version of this code is available here:...

For an overview of their DeepBach project with sample output, see the url HERE [I'm an inline-style link]http://www.flow-machines.com/deepbach-polyphonic-music-generation-bach-chorales/.


To scrape the additional files for the training corpus follow my github repo here:...
http://kunstderfuge.com/

An easy way of running this code is to launch an Azure Deep Learning VM, clone the repo, and start training away! I have had issues running the code and chose to hard code the pickle path for custom datasets in my version of the repo.

# Data

Starting with Bach to train music makes a lot of sense to me. Prior to working in data science I actually was an orchestral musician having received my masters degree at CSU studying under world renown percussionist/teacher/inventor Tom Freer. Through out my undergraduate and graduate degrees the music theory of Bach was studied quite a bit. Bach had a very rule and pattern based approach to composition through what is commonly referred to as counterpoint. Such rules at paralell fifths where not allowed. This is when 2 notes that are a fifth interval apart, say a c and a g. An example of such a violation would be if you had a chord of c and g on one beat and the next be the voices moved in paralell to a d and a. In my opinion, becuase Bach's approach to composition was more conservative and restrictive than many later composers, it is easier to find a signal from the noise when training a model than compositions from others. Compositional rules and constraints gradually peeled away through the years up to more recent abrast compositions which are not included in the training corpus for this post.

# Modeling 

This approach utilizes Stacked LSTM models and psuedo-gibbs sampling. To learn more, see the research paper from CSL here:

This project utilizes tensorflow and Keras. It embedds the data into a one hot array for the music in a sparse representation. It uses an Adam Optimizer and ... erorr testing (db check cross entropy??)

The core project use the default optimization settings. The learning rate looked to high for me so I lowered it from .001 to .0009. The training results were hitting about 100% with the defaults and about 92% after lowering the learning rate. The testing results were about 84%.

## A. Reharmonizing Bach theme with Chorale/String Chorale corpus

This is a compositional approach that has been around for hundreds of years, in which a composer will that the theme of an eariler peroid composer and reharmonize the general theme of the earlier piece. For the first model we try lets tak
