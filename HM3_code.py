import math
import random
import time
import csv
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from collections import Counter

# cosine similarity
def cosine_similarity(A, B):
    dot_product = np.dot(A, B.T)
    norm_a = np.linalg.norm(A)
    norm_b = np.linalg.norm(B)
    return dot_product / (norm_a * norm_b)

#load data from CSV
def isValidNumberString(s):
  if len(s) == 0:
    return False
  if  len(s) > 1 and s[0] == "-":
      s = s[1:]
  for c in s:
    if c not in "0123456789.":
      return False
  return True

def stringsToNumbers(myList):
    for i in range(len(myList)):
        if (isValidNumberString(myList[i])):
            myList[i] = float(myList[i])

def lineToTuple(line):
    # remove leading/trailing witespace and newlines
    cleanLine = line.strip()
    # get rid of quotes
    cleanLine = cleanLine.replace('"', '')
    # separate the fields
    lineList = cleanLine.split(",")
    # convert strings into numbers
    stringsToNumbers(lineList)
    lineTuple = tuple(lineList)
    return lineTuple

def loadCSV(fileName):
    fileHandler = open(fileName, "rt")
    lines = fileHandler.readlines()
    fileHandler.close()
    del lines[0] # remove the header
    dataset = []
    for line in lines:
        instance = lineToTuple(line)
        dataset.append(instance)
    return dataset

#calculate the distance.
def distance(instance1, instance2,dist_type):
    dist = 0
    if instance1 == None or instance2 == None:
        return float("inf")
    elif (dist_type == "Euclidean"):
        for i in range(1, len(instance1)):
            dist += (instance1[i] - instance2[i])**2
    elif(dist_type == "Cosine"):
        instance1_temp = np.array(instance1[1:]).reshape(1,-1)
        instance2_temp = np.array(instance2[1:]).reshape(1,-1)
        dist = 1-cosine_similarity(instance1_temp,instance2_temp)
    else:
        min = 0
        max = 0
        for i in range(1, len(instance1)):
            if (instance1[i] > instance2[i]):
                min += instance2[i]
                max += instance1[i]
            else:
                min += instance1[i]
                max += instance2[i]
        dist = 1-min/max
        
    return dist

#assign data point to center
def createEmptyListOfLists(numSubLists):
    myList = []
    for i in range(numSubLists):
        myList.append([])
    return myList

def assign(instance, centroids,dist_type):
    minDistance = distance(instance, centroids[0],dist_type)
    minDistanceIndex = 0
    for i in range(1, len(centroids)):
        d = distance(instance, centroids[i],dist_type)
        if (d < minDistance):
            minDistance = d
            minDistanceIndex = i
    return minDistanceIndex

def assignAll(instances, centroids,dist_type):
    clusters = createEmptyListOfLists(len(centroids))
    clusters_item_number = createEmptyListOfLists(len(centroids))
    for i,instance in enumerate(instances):
        clusterIndex = assign(instance, centroids,dist_type)
        clusters[clusterIndex].append(instance)
        clusters_item_number[clusterIndex].append(i)
    return clusters,clusters_item_number
#calculate next center
def meanInstance(name, instanceList):
    numInstances = len(instanceList)
    if (numInstances == 0):
        return
    numAttributes = len(instanceList[0])
    means = [name] + [0] * (numAttributes-1)
    for instance in instanceList:
        for i in range(1, numAttributes):
            means[i] += instance[i]
    for i in range(1, numAttributes):
        means[i] /= float(numInstances)
    return tuple(means)

def computeCentroids(clusters):
    centroids = []
    for i in range(len(clusters)):
        name = "centroid" + str(i)
        centroid = meanInstance(name, clusters[i])
        centroids.append(centroid)
    return centroids

#computer SSE of clustering
def computeWithinss(clusters, centroids,dist_type):
    result = 0
    for i in range(len(centroids)):
        centroid = centroids[i]
        cluster = clusters[i]
        for instance in cluster:
            if dist_type == "Euclidean":
                result += distance(instance,centroid,dist_type)
            elif dist_type == "Cosine":
                result += distance(instance,centroid,dist_type)**2
            else:
                result += distance(instance,centroid,dist_type)**2
    return result


def kmeans(data, k,dist_type,initcenter=None):
    result = {}
    if (initcenter == None or len(initcenter) < k):
        random.seed(time.time())
        center = random.sample(data, k)
    else:
        center = initcenter
        
    prevcenter = []
    #clusters:includes point in this cluster
    #center: center of each cluster
    pbar = tqdm(total = 500,desc="kmeans processing")
    i = 0
    withinss = 0
    withinss_pre = 0
    condition = 1
    while (condition):
        withinss_pre = withinss
        clusters,clusters_item_number = assignAll(data, center,dist_type)
        prevcenter = center
        center = computeCentroids(clusters)
        withinss = computeWithinss(clusters, center,dist_type)
        i += 1
        pbar.update(1)
        if (center == prevcenter or (withinss_pre<withinss and i>1)  or i >=499):
            condition =0
    pbar.close()
        
    result["clusters"] = clusters
    result["center"] = center
    result["withinss"] = withinss
    result["clusters_item_number"] = clusters_item_number
    return result,i

def vote_labeling (result,label):
    cluster_label = []
    label_temp = []
    for i in range(len(result["clusters_item_number"])):
        for j in range(len(result["clusters_item_number"][i])):
            index = result["clusters_item_number"][i][j]
            label_temp.append(label[index])
        counter = Counter(label_temp)
        most_common_element = counter.most_common(1)[0][0]
        cluster_label.append(most_common_element)
#

dataset = loadCSV("./kmeans_data/data.csv")
lable = loadCSV("./kmeans_data/label.csv")
for i in tqdm(range(3),desc="training case"):
    if i == 0:
        result,iteration = kmeans(dataset,10,"Euclidean")
    elif i == 1:
        result,iteration = kmeans(dataset,10,"Cosine")
    else:
        result,iteration = kmeans(dataset,10,"Jarcard")
    
    if i == 0:
        print("finish Euclidean iteration:{}".format(iteration))
    elif i == 1:
        print("finish Cosine iteration:{}".format(iteration))    
    else:
        print("finish Jarcard iteration:{}".format(iteration))


