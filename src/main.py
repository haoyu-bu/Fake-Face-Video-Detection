import sys
import os
import getopt
import csv
import numpy as np
import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler
from sklearn.utils import shuffle
from sklearn import svm
import sklearn
from collections import Iterable
try:
   import cPickle as pickle
except:
   import pickle

def usage():
    print "usage: " + sys.argv[0] + " -i input-feature-path -o output-dir"

def read_features(feature_path):
    print "loading features..."
    blink, head_tran, head_rot, gaze, mouth, body, scales = {}, {}, {}, {}, {}, {}, {}
    for f_name in os.listdir(feature_path):
        if f_name[7:12] == "blink":
            # print "blink"
            # blink (left, right)
            with open(os.path.join(feature_path, f_name), "rb") as f:
                bl = pickle.load(f)
                # bl = [(float(l) + float(r)) / 2 for l, r in bl]
                blink[f_name[:6]] = bl
        elif f_name[7:11] == "face":
            # print "face"
            ht, hr, g, m = [], [], [], []
            scale = 0
            with open(os.path.join(feature_path, f_name), "r") as face_csv:
                csv_reader = csv.DictReader(face_csv)
                for item in csv_reader:
                    scale += np.linalg.norm(np.array([float(item[' x_27']), float(item[' y_27'])]) 
                            - np.array([float(item[' x_8']), float(item[' y_8'])]))
                    ht.append([float(item[' pose_Tx']), float(item[' pose_Ty']), float(item[' pose_Tz'])])
                    hr.append([float(item[' pose_Rx']), float(item[' pose_Ry']), float(item[' pose_Rz'])])
                    # g.append([float(item[' gaze_angle_x']), float(item[' gaze_angle_y'])])
                    g.append([float(item[' gaze_0_x']), float(item[' gaze_0_y']), float(item[' gaze_0_z']), 
                                float(item[' gaze_1_x']), float(item[' gaze_1_y']), float(item[' gaze_1_z'])])                    
                    m.append(np.linalg.norm(np.array([float(item[' x_51']), float(item[' y_51'])]) 
                            - np.array([float(item[' x_57']), float(item[' y_57'])])))
                head_tran[f_name[:6]] = ht
                head_rot[f_name[:6]] = hr
                gaze[f_name[:6]] = g
                mouth[f_name[:6]] = m
                scales[f_name[:6]] = np.mean(scale)
        elif f_name[7:11] == "body":
            # print "body"
            b = []
            b_dict = {}
            with open(os.path.join(feature_path, f_name), "r") as f:
                for line in f:
                    line = str(line).split(' ')
                    b_dict[line[0]] = np.linalg.norm(np.array([float(line[31]), float(line[32])]) - np.array([float(line[34]), float(line[35])]))
            for k in sorted(b_dict.keys()):
                b.append(b_dict[k])
            body[f_name[:6]] = b
    
    # clean data
    noise = []
    min = sys.maxint
    name = None
    for k in blink:
        if len(blink[k]) !=  len(body[k]) or len(blink[k]) !=  len(gaze[k]):
            print k, len(blink[k]), len(gaze[k]), len(body[k])
            noise.append(k)
        else:
            if len(blink[k]) < min:
                min = len(blink[k]) 
                name = k
    print "min length", min, name
    for k in noise:
        del blink[k]
        del head_rot[k]
        del head_tran[k]
        del gaze[k]
        del mouth[k]
        del body[k]

    features = {}
    for key in body.keys():
        body[key] /= scales[key]
        mouth[key] /= scales[key]

        f = []
        for i in range(len(body[key])):
            fi = []
            for x in [blink[key][i], head_tran[key][i], head_rot[key][i], gaze[key][i], mouth[key][i], body[key][i]]:
                if isinstance(x, Iterable):
                    fi.extend(x)
                else:
                    fi.append(x)
            f.append(fi)
        features[key] = np.array(f)
    
    # normalization
    f_matrix = []
    for k in features:
        f_matrix.extend(features[k])
    f_matrix = np.array(f_matrix)
    
    std = []
    # mean = []
    for i in range(len(f_matrix[0])):
        std.append(np.std(f_matrix[:, i], ddof = 1))
        # mean.append(np.mean(f_matrix[:, i]))

    for k in features:
        for i in range(len(features[k][0])):
            # features[k][:, i] -= mean[i]
            features[k][:, i] /= std[i]

    return features, min

def generate_tp(features, type='first'):
    print "generating talking profile..."
    tp = {}
    for key in features.keys():
        f = features[key]
        tp_f = []
        if type == 'first' or type == 'second':
            for i in range(len(f) - 1):
                tp_f.append(np.array(f[i+1]) - np.array(f[i]))
            if type == 'second':
                for i in range(len(tp_f) - 1):
                    tp_f[i] = tp_f[i+1] - tp_f[i]
        tp[key] = tp_f
    return tp

def classify(X, Y, f_X):
    X = np.reshape(X, (np.shape(X)[0], -1))
    f_X = np.reshape(f_X, (np.shape(f_X)[0], -1))
    rus = RandomUnderSampler(return_indices=True)
    X, Y, id_rus = rus.fit_sample(X, Y)

    X, Y = shuffle(X, Y)
    X = np.array(X)
    f_X = shuffle(f_X)

    clf = svm.SVC(gamma='scale')
    result = {}

    skf = sklearn.model_selection.StratifiedKFold(n_splits=len(Y)/2)
    skf.get_n_splits(X, Y)
    fps = []
    tps = []
    i = 0
    for train_index, test_index in skf.split(X, Y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        neg = list(y_test).index(0)
        if i < len(f_X):
            X_test[neg] = f_X[i]
            i += 1
        clf.fit(X_train, y_train)
        predict = clf.predict(X_test)
        _, fp, _, tp = sklearn.metrics.confusion_matrix(y_test, predict).ravel()
        fps.append(fp)
        tps.append(tp)
    result['recall'] = np.mean(tps)
    result['fp'] = np.mean(fps)
    print result['recall'], result['fp']

    return result

def main(feature_path, out_dir):
    tp_mean = []
    fp_mean = []
    for pid in range(32):
        # load
        features, min_l = read_features(feature_path + "/" + str(pid+1))
        
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
            tps.append(result['recall'])
            fps.append(result['fp'])

        for i, t, f in zip(range(len(tps)), tps, fps):
            if i >= len(tp_mean):
                tp_mean.append([])
                fp_mean.append([])
            tp_mean[i].append(t) 
            fp_mean[i].append(f)
    
        plt.plot(range(5, min_l, 5), tps, 'b', label='True Positive')
        plt.plot(range(5, min_l, 5), fps, 'r', label='False Positive')
        plt.legend(loc = 'best')
        plt.ylabel('Rate')
        plt.xlabel('Frame Size K')
        #plt.show()
        plt.savefig(os.path.join(out_dir, str(pid+1) + ".jpg"))
        plt.clf()
    
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