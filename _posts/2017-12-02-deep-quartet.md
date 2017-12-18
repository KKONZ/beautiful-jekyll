---
layout: post
published: true
title: Deep Quartet
---
# Music Generation with Deep Neural Networks, Bach and Beyond

## By Karl Konz

![alt text](/img/DeepBachImg1.jpg "Deep Bach")



Prior to becoming a data scientist, I was lucky enough to study in one of the finest graduate orchestral percussion programs in the world. Most of my classmates during that time now have big orchestra jobs such as the Metropolitan opera and the Oregon Symphony. The program was very structured and formulaic and I am convinced that having studied under Tom Freer laid the foundation for me to have the capacity to do well in data science. So, thank you Tom! Whenever I get a chance, I love to work on a music data modeling project. This is the most ambition project I have tried yet and has also turned out to be the most rewarding.

***For this post I will explore the limits of the [deepBach](https://arxiv.org/abs/1612.01010) modeling approach to generate music with deep neural networks.***

The code is open sourced and can be found under [SONY deepBach](https://github.com/Ghadjeres/DeepBach), the Bach chorale corpus for that research is readily availble in the music21 package. I wanted to explore expanding the corpus to include compositions from after the baroque peroid as well. One of the constraints of this modeling techinque is that there has to be the same number of voices for each training composition. The Bach files used for the initial research have four part harmonies so I decided to add string quartets from the classical, romantic, and nationalist periods as they have the same number of voices.

## Additional data, string quartets

In addition to the roughly 350 Bach chorale files from the original research, I am added 189 string quartet midi files from Beethoven, Mozart, Shostakovich, Brahms, Schumann, Schubert, and others. To gather the additional string quartet midi files, I wrote a script to scrape the files I wanted from site [Kunst Der Fuge](http://kunstderfuge.com/) with [Selenium](http://selenium-python.readthedocs.io/) using the Python wrapper. Below is a snippet for this was acheived, I used the gecko driver with firefox to obtain the additional string quartets.

If you have pip installed on your computer, you can use the following command to download the python library

```python
pip install -U selenium
```

Follow the [Gecko driver]("https://github.com/mozilla/geckodriver/releases") instructions to download the appropriate driver for Firefox.

Next setup the preferences for the browser

```python
# Instantiate a webdriver profile
firefox_profile = webdriver.FirefoxProfile()
# Disable the prompt asking to whether and where to save the file
firefox_profile.set_preference("browser.helperApps.neverAsk.saveToDisk", "audio/midi")a
firefox_profile.set_preference("browser.download.folderList", 2)
firefox_profile.set_preference("browser.download.dir", "<Path to download file into>")
firefox_profile.set_preference("browser.download.manager.showWhenStarting",False)

# Launch the browser with the appropriate settings
driver = webdriver.Firefox(firefox_profile=firefox_profile)
# Get the Kunst Der Fuge URL  
driver.get('http://www.kunstderfuge.com/-/db/log-in.asp')
```

Before continuing, make sure that you have subscribed to Kunst Der Fuge website if you haven't already. I am frugal so I chose to pay less and scrape the files I wanted. If you want to skip this process and pay more you can download the midi database.

Next Authenticate into the website in order to access the files.

```python
# Create a username object with your username
username = driver.find_element_by_id("Email")
# Send the Username to the approprate element with the 'Email' id
username.send_keys("<username>") 
# Create a password object with your password
password = driver.find_element_by_id("Password")
# Send the password to the appropriate element with the 'password id
password.send_keys("<password>") # Enter password
# Click the submit button
driver.find_element_by_name("Submit").click()

```

After authenticating into the website, we will need to navigate to the midi section of the website.

```python
# find all the midi elements by xpath
MIDI = driver.find_elements_by_xpath("//*[text()='MIDI']")
# Extract the midi elements 
for ii in MIDI:
    link = ii.get_attribute('href')
# navigate to the link midi attribute
driver.get(link)
```

The last step is to download the actual string quartets, below is the method used to extract the Beethoven files. The full script can be found [here]("https://github.com/KKONZ/SpringBoard/blob/master/Capstone%201/KunstDerFuge.com-Selenium.ipynb), note that the format of the website varies a bit from composer to composer.

```python
# find the Beethoven Chamber element by xpath
BeethovenChamber = driver.find_elements_by_xpath("//*[text()='Chamber music']")
# extract the URL for the the xpath
for ii in BeethovenChamber:
    link = ii.get_attribute('href')
# Navigate to the Beethoven Chamber music page
driver.get(link)
# Click on each of the quartet midi links 
elems = driver.find_elements_by_xpath("//a[contains(@href, 'quartet')]")
for elem in elems:
    elem.click()
    time.sleep(3)

```

By adding the string quartet files there was a 830% increase in the the count of notes from what was used in the original deepBach study. So there was just over half the number of string quartet files as Bach files, however the string quartet files contain much more notes. I experimented with subsetting the string quartets into smaller files to sort of standardize the size of the files, but that approach did not yeild very complelling or statistically accurate results. 

I also conducted a z test to check the proportion of the notes in terms of frequency and note duration combinations as plotted below, the p value from the test was 7x10-43, thus there is evidence of a difference in the proportion of notes between the Bach chorales and the new string quartets I added. Below is the code I used to conduct that test


I used the music21 package to analyze the note compositions of the Bach data set and the String Quartets separetly. If pip is installed, the following snippet can be used to download the package.

```python
pip install music21
```



We can see that notes with smaller durations are more common in the string quartet files than in the Bach Chorale files. In other words, it is more common to have faster notes in the string quartets than in the Bach compositions. See my [Project Build](https://github.com/KKONZ/SpringBoard/tree/master/Capstone%201) for more details on this project.


<img src="/img/Bach__Notes.svg" alt="StringNotes" />


<img src="/img/Strings__Notes.svg" alt="StringNotes" />


I put in fairly exhaustive efforts somehow balance the midi files since there were fewer of those compositions but their size was on average so much bigger than the chorale files. I wanted to start by taking each of the string quartet files and chopping them up into 20 measure sequences. This lended itself to hours and hours which turned into months of banging my head against the wall. I ended up pushing a couple contriubtions to the MIT maintained package music21 but was experiencing a lot of issues. Mainly that notes with complex subdivisions would end up ringing forever when writing the music 'streams' to midi and by the end of the sequences there would often be 5-10 notes with endless durations making for a dissonant and noisy files. I also tried using pretty-midi which appeared to note produce any noticeable issues with note complexity but was evidentally producing a number of files that would cause errors for some reason mid-train. I successfully trained over 100 of the chopped files but ended up abandoning that approach because the cleaning process was creating issues and the test accuracy was also not very good with this approach compared to using the full files.


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
![alt text](/img/VITfirst.jpg "Deep Bach")
This model follows metadata sequences in which the conditional probability distribution is defined below:

{pi,t(Vit|V\i,t,M, θi,t)} i∈[4],t∈[T]

Vit indicates the voice i at time index t and V\i,t are all variables in V except for the variable Vit. So that the time can be invariant so that sequences of any size can be used, the parameters are shared between all conditional probability distributions in the same voice:

θi:= θi,t, pi:= pi,t ∀t ∈ [T].

Then each of the conditional probability distributions are fit to the data by maximizing the log-likigood. This results in four classification problems represented mathematically below:

maxθiXtlog pi(Vti|V\i,t,M, θi), for i ∈ [4], 

This in effect predicts a note, based off of the value of its neighboring notes. Each classifier is fit using four neural networks. Two of which are deep neural networks, one dedicated to summing past information and the other summing future information in conjunction with a non-recurrent NN for notes occuring at the same time. The output from the last recurrent neural network is preserved and the three outputs are merged and used in the fourth neural network with output:

pi(Vti|V\i,t,M, θ)

The illustration below shows the stacked models described above:

![alt text](/img/LSTMref.JPG "Model reference")

Generation in the depedency networks is done by utlizing pseudo-gibbs sampling, where the conditional distributions are potentially incompatible and that the conditional distributions are not neccesssairly from a joint distribution p(V). This Markov chain does converge to another stationary distribution and applications on real data demonstrated this method yeilded accurate joint probabilities.

The results from this modeling technique seem to speak for themselves. See the results of the “Bach or Computer” experiment below. The figure shows the distribution of the votes between “Computer” (bluebars) and “Bach” (red bars) for each model and each level of expertiseof the voters (from 1 to 3). The J.S.Bach field are actual Bach compositions and MLP stands for a multi-layer perceptron and MaxEnt stands for Maximum Entropy


![alt text](/img/DeepBachBench.JPG "Deep Bach Bench")

I had issues running the source code on my windows machine and ended up having to just hard code the pickled data set path and name instead of using the os python package. The code I used to actually train the AI music embedded below is available [KKONZ deepBach](https://github.com/KKONZ/DeepBach). I ran the code on an Azure Deep Learning VM with NVIDIA GPUs to shorten the training time. Note that this version does not use Keras 2 yet. After cloning the github repository you will also need to download a couple libraries
[music21](http://web.mit.edu/music21/) and [tqdm](https://pypi.python.org/pypi/tqdm):

```python
git clone "http://github.com/kkonz/DeepBach"
pip install music21
pip install tqdm
```

Then using the following parameters, you can adjust the following parameters and start modeling right out of the box. To adjust optimizer settings you will need to do so in the deepBach.py file.

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
