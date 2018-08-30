#!/usr/bin/env python
# -------------------------------------------------------------------
# File Name : create_dataset_noise.py
# Creation Date : 05-12-2016
# Last Modified : Fri Dec  9 12:26:58 2016
# Author: Thibaut Perol <tperol@g.harvard.edu>
# -------------------------------------------------------------------
"""
Load ONE .mseed and Benz et al., 2015 catalog and create tfrecords of noise.
The tfrecords are the noise traces and a label -1.
e.g.,
./bin/preprocess/create_dataset_noise.py \
--stream data/streams/GSOK029_2-2014.mseed \
--catalog data/catalog/Benz_catalog.csv\
--output data/tfrecords/GSOK029_2-2014.tfrecords
"""
import os
import numpy as np
from quakenet.data_pipeline import DataWriter
import tensorflow as tf
from obspy.core import read
from quakenet.data_io import load_catalog
from obspy.core.utcdatetime import UTCDateTime
import fnmatch
import json
import random
#import re
flags = tf.flags
flags.DEFINE_string('stream_path', None,
                    'path to the stream to process')
flags.DEFINE_string('catalog', None,
                    'path to a complete catalog to avoid events')
flags.DEFINE_string('output_dir', None,
                    'path to the directory in which the tfrecords are saved')
flags.DEFINE_bool("plot", False,
                  "If we want the event traces to be plotted")
flags.DEFINE_float('window_size', 2,
                   'size of the window samples (in seconds)')
flags.DEFINE_float('window_step', 50,
                   'size of the window step(in seconds)')
flags.DEFINE_float('max_windows', 50000,
                   'number of windows to generate')
flags.DEFINE_boolean("save_mseed",False,
                     "save the windows in mseed format")
FLAGS = flags.FLAGS


def preprocess_stream(stream):
    stream = stream.detrend('constant')
    ##add by mingzhao,2017/12/2
    #stream = stream.filter('bandpass', freqmin=0.5, freqmax=20)
    #stream = stream.resample(10.0)
    ##########
    #return stream.normalize()
    return stream
def remove_repeat(st_event,n_samples):
    dic={}
    a=[]
    for i in range(3):
        for item in st_event[i].data:
            if item in dic.keys():
                dic[item]+=1
            else:
                dic[item]=1
        mm=max(dic.values())
        a.append(1.0 * mm / n_samples)
    #print (a)
    return a
def write_json(metadata,output_metadata):
    with open(output_metadata, 'w') as outfile:
        json.dump(metadata, outfile)

def filter_catalog(cat, starttime, endtime):

    #cat = cat[(cat.utc_timestamp > starttime)
    #          & (cat.utc_timestamp < endtime)]
    #add stream name,2017/12/2


    cat = cat[(cat.utc_timestamp > starttime)
              & (cat.utc_timestamp < endtime)]
    return cat
def filter_small_ampitude(st_event,n_samples):
    ampl_e = max(abs(st_event[0].data))
    ampl_n = max(abs(st_event[1].data))
    ampl_z = max(abs(st_event[2].data))
    a_e = 1.0 * len(filter(lambda x: -1e-12<= x <= 1e-12, st_event[0].data)) / n_samples
    a_n = 1.0 * len(filter(lambda x: -1e-12<= x <= 1e-12, st_event[1].data)) / n_samples
    a_z = 1.0 * len(filter(lambda x: -1e-12<= x <= 1e-12, st_event[2].data)) / n_samples
   # print (87,a_e,a_n,a_z)
    return a_e,a_n,a_z
def main(_):

# Create dir to store tfrecords
    if not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)

    # Load stream
    stream_path = FLAGS.stream_path
    print stream_path
    stream_file = os.path.split(stream_path)[-1]
    print "+ Loading Stream {}".format(stream_file)
    stream = read(stream_path)
    print '+ Preprocessing stream'
    stream = preprocess_stream(stream)
    #stream.resample(10.0)
    # Dictionary of nb of events per tfrecords
    metadata = {}
    output_metadata = os.path.join(FLAGS.output_dir,"metadata.json")

    # Load Catalog
    print "+ Loading Catalog"
    cat = load_catalog(FLAGS.catalog)
    starttime = stream[0].stats.starttime.timestamp
    endtime = stream[-1].stats.endtime.timestamp
    print "startime", UTCDateTime(starttime)
    print "endtime", UTCDateTime(endtime)
    #print stream[0].stats
    #m2 = re.search(cat.stname.values[:], stream_file.split(".")[1])
    #print m2.group()
    cat = filter_catalog(cat, starttime, endtime)
    #cat = cat[(cat.stname == str(stream_file.split(".")[1]))]


   # cat = cat[(cat.stname == str(stream_file.split(".")[1])) or
   #       (cat.stname == str(stream_file.split(".")[1][:-1]))]
    #print cat
    print "First event in filtered catalog", cat.Date.values[0], cat.Time.values[0]
    print "Last event in filtered catalog", cat.Date.values[-1], cat.Time.values[-1]
    cat_event_times = cat.utc_timestamp.values

    # Write event waveforms and cluster_id=-1 in .tfrecords
    n_tfrecords = 0
    output_name = "noise_" + stream_file.split(".mseed")[0] + \
                  "_" + str(n_tfrecords) + ".tfrecords"
    output_path = os.path.join(FLAGS.output_dir, output_name)
    writer = DataWriter(output_path)

    # Create window generator
    win_gen = stream.slide(window_length=FLAGS.window_size,
                           step=FLAGS.window_step,
                           include_partial_windows=False)

   #Create window generator and shuffle the order,2017/12/4
    #win_gen = [tr for tr in stream.slide(window_length=FLAGS.window_size,
    #                       step=FLAGS.window_step,
    #                    include_partial_windows=False)]

    #random.shuffle(win_gen)

    if FLAGS.max_windows is None:
        total_time = stream[0].stats.endtime - stream[0].stats.starttime
        max_windows = (total_time - FLAGS.window_size) / FLAGS.window_step
    else:
        max_windows = FLAGS.max_windows

    # Create adjacent windows in the stream. Check there is no event inside
    # using the catalog and then write in a tfrecords with label=-1


    n_tfrecords = 0

    for idx, win in enumerate(win_gen):

        # If there is not trace skip this waveform
        n_traces = len(win)
        if n_traces == 0:
            continue
        # Check trace is complete
        if len(win)==3:
            n_samples = min(len(win[0].data),len(win[1].data))
            n_samples = min(n_samples, len(win[2].data))
            #to get rid of super small amplitude,2017/12/6
            ampl_e,ampl_n,ampl_z=filter_small_ampitude(win,n_samples)
            if ampl_e > 0.3 or ampl_n > 0.3 or ampl_z > 0.3:
                continue
            #a = remove_repeat(win, n_samples)
            #if a[0] > 0.3 or a[1] > 0.3 or a[2] > 0.3:
            #    continue
        else:
            n_sample = 10
            ampl_e = max(abs(win[0:-1].data))
            if ampl_e < 1e-10:
                continue
        n_pts = win[0].stats.sampling_rate * FLAGS.window_size + 1
        # Check if there is an event in the window
        window_start = win[0].stats.starttime.timestamp
        window_end = win[-1].stats.endtime.timestamp
        ##add window extend to broaden the window,so that more events can be avoid,2017/12/07
        window_start_extend = window_start - FLAGS.window_step
        window_end_extend = window_end + FLAGS.window_step
        after_start = cat_event_times > window_start_extend
        before_end = cat_event_times < window_end_extend

        #print window_start_extend,window_end_extend
        try:
            cat_idx = np.where(after_start == before_end)[0][0]
            event_time = cat_event_times[cat_idx]
            is_event = True
            assert window_start_extend < cat.utc_timestamp.values[cat_idx]
            assert window_end_extend > cat.utc_timestamp.values[cat_idx]
            print "avoiding event {}, {}".format(cat.Date.values[cat_idx],
                                                 cat.Time.values[cat_idx])
        except IndexError:
            # there is no event
            is_event = False
            if (len(win)==3) and (n_pts == n_samples):
                # Write tfrecords
                #writer.write(win.normalize(), -1)
                writer.write(win.copy().normalize(),-1)
                #writer.write(win.copy().resample(10).filter('bandpass', freqmin=0.5, freqmax=20).normalize(),
                           #  -1)
                # Plot events
                if FLAGS.plot:
                    import matplotlib
                    matplotlib.use('Agg')
                    #trace = win[0].filter('bandpass', freqmin=0.5, freqmax=20)
                    trace = win[0]
                    viz_dir = os.path.join(
                        FLAGS.output_dir, "viz", stream_file.split(".mseed")[0])
                    if not os.path.exists(viz_dir):
                        os.makedirs(viz_dir)
                   # trace.resample(10).plot(outfile=os.path.join(viz_dir,
                   #                                 "noise_{}.png".format(str(window_start))))
                    trace.plot(outfile=os.path.join(viz_dir,
                                                           ####changed at 2017/11/25,use max cluster_prob instead of cluster_id
                                                           #                "event_{}_cluster_{}.png".format(idx,cluster_id)))
                                                           "noise_{}_{}.png".format(win[0].stats.station,
                                                                                 str(win[0].stats.starttime).replace(
                                                                                     ':', '_'))))
        if idx % 1000  ==0 and idx != 0:
            print "{} windows created".format(idx)
            # Save num windows created in metadata
            metadata[output_name] = writer._written
            print "creating a new tfrecords"
            n_tfrecords +=1
            output_name = "noise_" + stream_file.split(".mseed")[0] + \
                          "_" + str(n_tfrecords) + ".tfrecords"
            output_path = os.path.join(FLAGS.output_dir, output_name)
            writer = DataWriter(output_path)

        if idx == max_windows:
            break

    # Cleanup writer
    print("Number of windows  written={}".format(writer._written))
    writer.close()

    # Write metadata
    metadata[stream_file.split(".mseed")[0]] = writer._written
    write_json(metadata, output_metadata)


if __name__ == "__main__":
    tf.app.run()
