#!/usr/bin/env python
# coding: utf-8

import numpy as np
import cv2
import os
from os.path import join
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import json


# Load data
def read_data(path):
    n_class = 30
    data = []
    class_table = {}
    for i, class_ in enumerate(os.listdir(path)):
        class_table[i] = class_
        class_ = join(path, class_)
        imgs = [cv2.imread(join(class_, file)) for file in os.listdir(class_) if file[0] != '.']
        data.append(imgs)
    return data, class_table

data, class_table = read_data('./database')


# Evaluate
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_AP(similarity, cur_class):
    ## calculate AP
        total, correct = 0, 0
        AP = 0.0
        for j in range(len(similarity)):
            total += 1
            if similarity[j][1] == cur_class:
                correct += 1
                AP += correct / total
        AP /= correct
        return AP
        
def evaluate(feature):
    n = len(feature)
    cur_class = 0
    AP_class = []
    tmp_AP, count = 0.0, 0
    
    for i in range(n):
        similarity = []
        for j in range(n):
            if j == i:
                continue
            sim =  cosine_similarity(feature[i][0], feature[j][0])
            similarity.append([sim, feature[j][1]])
        similarity = sorted(similarity, key=lambda x: x[0], reverse=True)
        
        ap = get_AP(similarity, feature[i][1])

        if feature[i][1] != cur_class:
            AP = tmp_AP / count
            AP_class.append([AP, cur_class])
            # init class & AP
            cur_class = feature[i][1]
            tmp_AP = ap
            count = 1
        else:
            tmp_AP += ap
            count += 1
        
    ## output result
    MAP = 0.0
    AP_class = sorted(AP_class, key=lambda x:x[0], reverse=True)
    for i in range(len(AP_class)):
        print('Class: {a}, AP: {b}'.format(a = class_table[AP_class[i][1]], b = AP_class[i][0]))
        MAP += AP_class[i][0]
    MAP /= len(AP_class)
    print('MAP: {a}'.format(a = MAP))


# Color feature
def color_histogram(data):
    feature = []
    for i in tqdm(range(len(data))):
        for j in range(len(data[i])):
            hist = []
            for channel in range(3):
                hist.extend(cv2.calcHist([data[i][j]], [channel], None, [16], [0, 256]))
            hist = np.array(hist)
            hist = hist.flatten()
            feature.append([hist, i])
    # normalize
    hist_features = np.array([f[0] for f in feature])
    for i in range(len(hist_features[0])):
        mean = np.mean(hist_features[:, i], axis=0)
        std = np.std(hist_features[:, i], axis=0)
        hist_features[:, i] = (hist_features[:, i] - mean) / std
    for i in range(len(feature)):
        feature[i][0] = hist_features[i]
    return feature

## get features and evaluate
colorFeature = color_histogram(data)
print('[Color feature]')
evaluate(colorFeature)

## plot feature histogram - take 30th for example
plt.bar(np.arange(len(colorFeature[30][0])), colorFeature[30][0])
plt.title('example feature')
plt.show()


# Texture feature
def build_filters():
    filters = []
    ksize = [7, 9, 11, 13, 15] # gabor尺度 - 5個
    lamda = np.pi/2.0 # 波長
    for theta in np.arange(0, np.pi, np.pi / 4): # gabor方向 - 4個
        for K in ksize:
            kernel = cv2.getGaborKernel((K, K), 1.0, theta, lamda, 0.5, 0, ktype=cv2.CV_32F)
            kernel /= 1.5 * kernel.sum()
            filters.append(kernel)
    return filters

def process(img, filters):
    accum = np.zeros_like(img)
    for kernel in filters:
        filtered = cv2.filter2D(img, cv2.CV_8UC3, kernel)
        np.maximum(accum, filtered, accum)
    return accum

### Gabor feature extraction
def getGabor(img, filters):
    feature = []
    for i in range(len(filters)):
        tmp = process(img, filters[i])
        mean, std = np.mean(tmp), np.std(tmp)
        feature.append(mean)
        feature.append(std)
    feature = np.array(feature)
    return feature

def gabor_filter(data, filters):
    feature = []
    for i in tqdm(range(len(data))):
        for j in range(len(data[i])):
            feat = getGabor(data[i][j], filters)
            feature.append([feat, i])
    return feature

## get features and evaluate
filters = build_filters()
textFeature = gabor_filter(data, filters)
print('[Texture feature]')
evaluate(textFeature)

## save features
for i in range(len(textFeature)):
    textFeature[i][0] = textFeature[i][0].tolist()
with open('texture.json', 'w') as f:
    json.dump(textFeature, f)


## load features
with open('texture.json', 'r') as f:
    textFeature = json.load(f)

## plot feature distribution
print('[Visualization]')
feature = np.array([ i[0] for i in textFeature])
label = [ i[1] for i in textFeature]
embedded = TSNE(n_components=2).fit_transform(feature)
## init color map
cmap = cm.rainbow(np.linspace(0.0, 1.0, max(label)+1))
np.random.shuffle(cmap)
plt.scatter(embedded[:, 0], embedded[:, 1], marker='o', color=cmap[label[:]])
plt.title('feature distribution')
plt.show()


# Local feature

def getSift(img):
    # create sift and compute features
    sift = cv2.xfeatures2d.SIFT_create(nfeatures=500)
    kp, features = sift.detectAndCompute(img, None)
    return features

def sift_with_kmeans_clustering(data):
    n_clusters = 256
    feature = []
    all_features = []
    for i in tqdm(range(len(data))):
        for j in range(len(data[i])):
            feat = getSift(data[i][j])
            feature.append([feat[n] for n in range(feat.shape[0])])
            for n in range(feat.shape[0]):
                all_features.append(feat[n])
    # apply kmeans clustering
    kmeans = KMeans(n_clusters=n_clusters).fit(all_features)
    return feature, kmeans

def retrieve_codebooks(feature, kmeans):
    cur = 0
    codebooks = []
    n_clusters = 256
    # retrieve codebook for each feature
    for i in tqdm(range(len(data))):
        for j in range(len(data[i])):
            tmp_feature = [0.0 for _ in range(n_clusters)]
            clusters = kmeans.predict(feature[cur])
            for n in range(clusters.shape[0]):
                tmp_feature[clusters[n]] += 1
            tmp_feature = [x / clusters.shape[0] for x in tmp_feature]
            codebooks.append([tmp_feature, i])
            cur += 1
    return codebooks

# get features and evaluate
localFeature, kmeans = sift_with_kmeans_clustering(data)
codebooks = retrieve_codebooks(localFeature, kmeans)
print('[Local feature]')
evaluate(codebooks)

# save features
with open('localFeature.json', 'w') as f:
    json.dump(codebooks, f)


# load features
with open('localFeature.json', 'r') as f:
    localFeature = json.load(f)

# plot feature distribution
print('[Visualization]')
feature = np.array([ i[0] for i in localFeature])
label = [ i[1] for i in localFeature]
embedded = TSNE(n_components=2).fit_transform(feature)
# Init color map
cmap = cm.rainbow(np.linspace(0.0, 1.0, max(label)+1))
np.random.shuffle(cmap)
plt.scatter(embedded[:, 0], embedded[:, 1], marker='o', color=cmap[label[:]])
plt.title('feature distribution')
plt.show()
