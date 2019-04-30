#!/usr/bin/python2

import sys
import os
import getopt

import numpy as np
import matplotlib.pyplot as plt

from dataset import load_features
from classifier import classify

def usage():
    print "usage: " + sys.argv[0] + " -i input-feature-path -o output-dir"


def generate_tp(features):
    '''
    generate talking profile by computing the difference between 
    each two consecutive frames.

    Args:
        features: A dict mapping keys to the corresponding data
        matrix fetched.

    Returns:
        A dict of talking profile.
    '''

    print "generating talking profile..."
    tp = {}
    for key in features.keys():
        f = features[key]
        tp_f = []
        for i in range(len(f) - 1):
            tp_f.append(np.array(f[i+1]) - np.array(f[i]))
        tp[key] = tp_f
    return tp

def main(feature_path, out_dir):
    tp_mean = []
    fp_mean = []
    for pid in range(32):
        # load
        features, min_l = load_features(feature_path + "/" + str(pid+1))
        
        # generate talking profile
        tp = generate_tp(features)

        tps = []
        fps = []
        for frame_size in range(5, min_l, 5):
            X = []
            Y = []
            f_X = []
            overlap = frame_size / 10
            for k in tp.keys():
                for i in range(0, len(tp[k]) - frame_size + overlap, frame_size):
                    if k[3] == 'f':
                        if i == 0:
                            f_X.append(np.array(tp[k][i:i+frame_size]))
                        else:
                            f_X.append(np.array(tp[k][i-overlap:i-overlap+frame_size]))
                    else: 
                        if i == 0:
                            X.append(np.array(tp[k][i:i+frame_size]))
                        else:
                            X.append(np.array(tp[k][i-overlap:i-overlap+frame_size]))
                        if k[0:4] == '%03dr' % (pid+1):
                            Y.append(1)
                        else:
                            Y.append(0)

            print "classfying..."
            result = classify(X, Y, f_X)
            tps.append(result['tp'])
            fps.append(result['fp'])

        for i, t, f in zip(range(len(tps)), tps, fps):
            if i >= len(tp_mean):
                tp_mean.append([])
                fp_mean.append([])
            tp_mean[i].append(t) 
            fp_mean[i].append(f)

        # draw result plot
        plt.plot(range(5, min_l, 5), tps, 'b', label='True Positive')
        plt.plot(range(5, min_l, 5), fps, 'r', label='False Positive')
        plt.legend(loc = 'best')
        plt.ylabel('Rate')
        plt.xlabel('Frame Size K')
        #plt.show()
        plt.savefig(os.path.join(out_dir, str(pid+1) + ".jpg"))
        plt.clf()
    
    # compute average result
    for i in range(len(tp_mean)):
        tp_mean[i] = sum(tp_mean[i]) / len(tp_mean[i])
        fp_mean[i] = sum(fp_mean[i]) / len(fp_mean[i])
    plt.plot(range(5, len(tp_mean) * 5, 5), tp_mean[:-1], 'b', label='True Positive')
    plt.plot(range(5, len(fp_mean) * 5, 5), fp_mean[:-1], 'r', label='False Positive')
    plt.legend(loc = 'best')
    plt.ylabel('Rate')
    plt.xlabel('Frame Size K')
    #plt.show()
    plt.savefig(os.path.join(out_dir, "mean.jpg"))

if __name__ == '__main__':
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'i:o:')
    except getopt.GetoptError, err:
        usage()
        sys.exit(2)

    for o, a in opts:
        if o == '-i':
            feature_path  = a
        elif o == '-o':
            out_dir = a
        else:
            assert False, "unhandled option"
    
    main(feature_path, out_dir)