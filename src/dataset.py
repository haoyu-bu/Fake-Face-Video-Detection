from collections import Iterable
import csv
import os
import sys
import numpy as np
try:
   import cPickle as pickle
except:
   import pickle

def load_features(feature_path):
    '''
    load features

    Args:
        feature_path: the path of features.

    Returns:
        A dict mapping keys to the corresponding data
        matrix fetched. Each key is a video.
    '''

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
                    b_dict[line[0]] = np.linalg.norm(np.array([float(line[31]), float(line[32])])
                                     - np.array([float(line[34]), float(line[35])]))
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
            for x in [blink[key][i], head_tran[key][i], head_rot[key][i], 
                gaze[key][i], mouth[key][i], body[key][i]]:
                if isinstance(x, Iterable):
                    fi.extend(x)
                else:
                    fi.append(x)
            f.append(fi)
        features[key] = np.array(f)
    
    features = normalize(features)

    return features, min

def normalize(features):
    '''
    normalize each column of the matrix.

    Args:
        features: A dict mapping keys to the corresponding data
        matrix fetched.

    Returns:
        Normalized dict.
    '''

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
    
    return features