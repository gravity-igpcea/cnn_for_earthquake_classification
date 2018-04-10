We use ConvNetQuake to train our own model for events classification based on the wenchuan aftershocks and use it to classify earthquakes from one day continous waveform(2008-07-25) of one station (MXI).

![The data of MXI,2008-07-25,and marked events](./data/XX.MXI_dayplot_0.png),etc

ConvNetQuake
=============

Perol., T, M. Gharbi and M. Denolle. Convolutional Neural Network for Earthquake detection and location. [preprint arXiv:1702.02073](https://arxiv.org/abs/1702.02073), 2017.

## Installation
* Download repository
* Install dependencies: `pip install -r requirements.txt`
* Add directory to python path: `./setpath.sh`
* Run tests: `./runtests.sh` (THIS NEEDS TO BE EXTENDED)
* Download the [data](https://www.dropbox.com/sh/3p9rmi1bcpvnk5k/AAAV8n9VG_e0QXOpoofsSH0Ma?dl=0) (roughly 70 Gb) and symlink to `data` 
`ln -s data Downloads/data`
* Download the [pre-trained models](https://www.dropbox.com/sh/t9dj8mmfx1fmxfa/AABSJQke8Ao6wfRnKMvQXipta?dl=0) and symlink to `models` 
`ln -s models Downloads/models`

## Data

Our model is trained on data from wenchuan aftershocks. 
The continuous waveform data is in ./data

The `data` directory contains:
* `XX.MXI.2008207000000.mseed`: the continious waveform data 
* `dayplot.py`: a script to plot the continious waveform
* `merge_dayplot.py`: a merge script
* `XX.MXI_dayplot_[0-64800].png`: marked earthquakes of the day 

## Trained model

The directory `trained_model` contains:
* `convnetquake`: trained model using over 20000 earthquakes slices (30s) and over 60000 slices of noises (30s)


## Detecting events in continuous waveform data

### From .mseed

./bin/predict_from_stream.py --stream_path data --checkpoint_dir trained_model/ConvNetQuake  --n_clusters 2 --window_size 30 --window_step 31 --output predict_MXI_one_day --plot --save_sac


It will generate a dir "predict_MXI_one_day",which contains:
 
* `viz`: the image of events,the name of the image contain its probility(prob) and its starttime,like "MXI_0.50053_2008-07-25T03_07_18.000000Z.png"
* `viz_not`: the image of noise,notice the  higher prob,the more likely it is an events,actually when the prob>0.1 there is a large chance it is an event. 
* `sac`: the slice data of viz

It proved using overlapping windows will get better results,however,the events will be identified more than once,you can choose the one with highest probility.To run,just uncomment "
#lists = np.arange(0,30,5)
" in predict_from_stream.py

![we merge all our identified slices with prob>0.1 and plot] (./XX.MXI_dayplot.png)
