#!/usr/bin/env python
# -------------------------------------------------------------------
# File Name : predict_from_stream.py
# Creation Date : 03-12-2016
# Last Modified : Sun Jan  8 13:33:08 2017
# Author: Thibaut Perol <tperol@g.harvard.edu>
# -------------------------------------------------------------------
""" Detect event and predict localization on a stream (mseed) of
continuous recording. This is the slow method. For fast detections
run bin/predict_from_tfrecords.py
e.g,
./bin/predict_from_stream.py --stream_path data/streams/GSOK029_12-2016.mseed
--checkpoint_dir model/convnetquake --n_clusters 6 --window_step 10 --output
output/december_detections --max_windows 8640
"""
import os
import setproctitle
import argparse

import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf
import shutil
import tqdm
import pandas as pd
import time
import fnmatch
from obspy.core import read
import quakenet.models as models
from quakenet.data_pipeline import DataPipeline
import quakenet.config as config
from quakenet.data_io import load_stream


def fetch_window_data(stream):
    """fetch data from a stream window and dump in np array"""
    cfg = config.Config()
    data = np.empty((cfg.win_size, 3))
    for i in range(3):
        data[:, i] = stream[i].data.astype(np.float32)
    data = np.expand_dims(data, 0)
    return data
def filter_small_ampitude(st_event):
    if len(st_event) == 3:
        n_samples = min(len(st_event[0].data), len(st_event[1].data))
        n_samples = min(n_samples, len(st_event[2].data))
    else:
        n_samples = len(st_event[0].data)

    a_e = 1.0 * len(filter(lambda x: -1e-12<= x <= 1e-12, st_event[0].data)) / n_samples
    a_n = 1.0 * len(filter(lambda x: -1e-12<= x <= 1e-12, st_event[1].data)) / n_samples
    a_z = 1.0 * len(filter(lambda x: -1e-12<= x <= 1e-12, st_event[2].data)) / n_samples
   # print (87,a_e,a_n,a_z)
    return a_e,a_n,a_z

def data_is_complete(stream):
    """Returns True if there is 1001*3 points in win"""
    cfg = config.Config() 
    try:   
        data_size = len(stream[0].data) + len(stream[1].data) + len(stream[2].data)
    except:
        data_size = 0
    data_lenth=int(cfg.win_size)*3
    if data_size == data_lenth:
        return True
    else:
        return False

def preprocess_stream(stream):
    stream = stream.detrend('constant')
    ##add by mingzhao,2017/12/2
    stream =stream.filter('bandpass', freqmin=0.5, freqmax=20)
    ##########
    return stream

def main(args):
    setproctitle.setproctitle('quakenet_predict')


    ckpt = tf.train.get_checkpoint_state(args.checkpoint_dir)

    cfg = config.Config()
    cfg.batch_size = 1
    cfg.n_clusters = args.n_clusters
    cfg.add = 1
    cfg.n_clusters += 1

    # Remove previous output directory
    if os.path.exists(args.output):
        shutil.rmtree(args.output)
    os.makedirs(args.output)
    if args.plot:
        os.makedirs(os.path.join(args.output,"viz"))
        os.makedirs(os.path.join(args.output, "viz_not"))
    if args.save_sac:
        os.makedirs(os.path.join(args.output,"sac"))

    # Load stream
    #stream_path = args.stream_path
    #stream_file = os.path.split(stream_path)[-1]
    #print "+ Loading Stream {}".format(stream_file)
    #stream = read(stream_path)
    #print '+ Preprocessing stream'
    #stream = preprocess_stream(stream)

    # # change to use a dir list,2017/12/07
    stream_path = args.stream_path
    try:
        stream_files = [file for file in os.listdir(stream_path) if
                    fnmatch.fnmatch(file, '*.mseed')]
    except:
        stream_files = os.path.split(stream_path)[-1]
    # stream data with a placeholder
    samples = {
        'data': tf.placeholder(tf.float32,
                                   shape=(cfg.batch_size, cfg.win_size, 3),
                                   name='input_data'),
        'cluster_id': tf.placeholder(tf.int64,
                                         shape=(cfg.batch_size,),
                                         name='input_label')
    }

    # set up model and validation metrics
    model = models.get(args.model, samples, cfg,
                       args.checkpoint_dir,
                       is_training=False)
    with tf.Session() as sess:
        model.load(sess, args.step)
        print 'Predicting using model at step {}'.format(
            sess.run(model.global_step))

        step = tf.train.global_step(sess, model.global_step)

        n_events = 0
        time_start = time.time()
        for stream_file in stream_files:
            stream_path1 = os.path.join(stream_path, stream_file)
            print " + Loading stream {}".format(stream_file)
            stream = load_stream(stream_path1)
            #print stream[0],stream[1].stats
            print " + Preprocess stream"
            stream = preprocess_stream(stream)
            print " -- Stream is ready, starting detection"

    # Create catalog name in which the events are stored
            catalog_name = os.path.split(stream_file)[-1].split(".mseed")[0] + ".csv"
            output_catalog = os.path.join(args.output,catalog_name)
            print 'Catalog created to store events', output_catalog

    # Dictonary to store info on detected events
            events_dic ={"start_time": [],
                 "end_time": [],
                 "cluster_id": [],
                 "clusters_prob": []}
    # Windows generator
           # win_gen = stream.slide(window_length=args.window_size,
           #                step=args.window_step,
           #                include_partial_windows=False)
            if args.save_sac:
                first_slice = stream.slice(stream[0].stats.starttime,stream[0].stats.starttime+args.window_size)
                first_slice.write(os.path.join(args.output,"sac",
                    "{}_{}_{}.sac".format(stream[0].stats.station,'0',
                                                   str(stream[0].stats.starttime).replace(':','_'))),
                                   format="SAC")
            if args.max_windows is None:
                total_time_in_sec = stream[0].stats.endtime - stream[0].stats.starttime
                max_windows = (total_time_in_sec - args.window_size) / args.window_step
            else:
                max_windows = args.max_windows
            try:
                lists = [0]
                #lists = np.arange(0,30,5)
                for i in lists:

                    win_gen = stream.slide(window_length=args.window_size, step=args.window_step,offset=i,
                                                                                 include_partial_windows=False)
                    for idx, win in enumerate(win_gen):
                        if data_is_complete(win):
                            ampl_e, ampl_n, ampl_z = filter_small_ampitude(win)
                            if ampl_e > 0.3 or ampl_n > 0.3 or ampl_z > 0.3:
                                continue
                            ampm_e = max(abs(win[0].data))
                            ampm_n = max(abs(win[1].data))
                            ampm_z = max(abs(win[2].data))
                            if ampm_e < 600 and ampm_n < 600 and ampm_z < 600:
                                continue
                    # Fetch class_proba and label
                            to_fetch = [samples['data'],
                                model.layers['class_prob'],
                                model.layers['class_prediction']]

                    # Feed window and fake cluster_id (needed by the net) but
                    # will be predicted
                            feed_dict = {samples['data']: fetch_window_data(win.copy().normalize()),
                                    samples['cluster_id']: np.array([0])}
                            sample, class_prob_, cluster_id = sess.run(to_fetch,feed_dict)
                        else:
                            continue

                    # # Keep only clusters proba, remove noise proba
                        clusters_prob = class_prob_[0,1::]
                        cluster_id -= 1

                    # label for noise = -1, label for cluster \in {0:n_clusters}

                        is_event = cluster_id[0] > -1
                        #print cluster_id[0],is_event
                        probs = '{:.5f}'.format(max(list(clusters_prob)))
                        save_event = (float(probs) - 0.1)>=0
                        #print probs,save_event
                        if is_event:
                            n_events += 1
                            print "event {} ,cluster id {}".format(is_event,class_prob_)
                            events_dic["start_time"].append(win[0].stats.starttime)
                            events_dic["end_time"].append(win[0].stats.endtime)
                            events_dic["cluster_id"].append(cluster_id[0])
                            events_dic["clusters_prob"].append(list(clusters_prob))


                        if idx % 1000 ==0:
                            print "Analyzing {} records".format(win[0].stats.starttime)

                        if args.plot :
                            import matplotlib
                            matplotlib.use('Agg')
                            win_filtered = win.copy()

                            if is_event:
                    # if args.plot:

                        # win_filtered.filter("bandpass",freqmin=4.0, freqmax=16.0)
                                win_filtered.plot(outfile=os.path.join(args.output,"viz",
                        ####changed at 2017/11/25,use max cluster_prob instead of cluster_id
                        #                "event_{}_cluster_{}.png".format(idx,cluster_id)))
                                         "{}_{}_{}.png".format(win[0].stats.station,str(probs),
                                                               str(win[0].stats.starttime).replace(':', '_'))))
                            else:

                                win_filtered.plot(outfile=os.path.join(args.output, "viz_not",
                                         "{}_{}_{}.png".format(win[0].stats.station,str(probs),
                                                               str(win[0].stats.starttime).replace(':', '_'))))

                        if args.save_sac and save_event:
                           # win_filtered = win.copy()
                            save_start=win[0].stats.starttime-12
                            save_end=win[0].stats.endtime+10
                            win_filtered = stream.slice(save_start,save_end)
                            win_filtered.write(os.path.join(args.output,"sac",
                                "{}_{}_{}.sac".format(win[0].stats.station,str(probs),
                                                               str(win[0].stats.starttime).replace(':','_'))),
                                               format="SAC")

                        if idx >= max_windows:
                            print "stopped after {} windows".format(max_windows)
                            print "found {} events".format(n_events)
                            break

            except KeyboardInterrupt:
                print 'Interrupted at time {}.'.format(win[0].stats.starttime)
                print "processed {} windows, found {} events".format(idx+1,n_events)
                print "Run time: ", time.time() - time_start

    # Dump dictionary into csv file
    #TODO
	    df = pd.DataFrame.from_dict(events_dic)
   	    df.to_csv(output_catalog)

    print "Run time: ", time.time() - time_start

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--stream_path",type=str,default=None,
                        help="path to mseed to analyze")
    parser.add_argument("--checkpoint_dir",type=str,default=None,
                        help="path to directory of chekpoints")
    parser.add_argument("--step",type=int,default=None,
                        help="step to load, if None the final step is loaded")
    parser.add_argument("--n_clusters",type=int,default=None,
                        help= 'n of clusters')
    parser.add_argument("--model",type=str,default="ConvNetQuake",
                        help="model to load")
    parser.add_argument("--window_size",type=int,default=30,
                        help="size of the window to analyze")
    parser.add_argument("--window_step",type=int,default=31,
                        help="step between windows to analyze")
    parser.add_argument("--max_windows",type=int,default=None,
                        help="number of windows to analyze")
    parser.add_argument("--output",type=str,default="output/predict",
                        help="dir of predicted events")
    parser.add_argument("--plot", action="store_true",
                     help="pass flag to plot detected events in output")
    parser.add_argument("--save_sac",action="store_true",
                     help="pass flag to save windows of events in sac files")
    args = parser.parse_args()

    main(args)
