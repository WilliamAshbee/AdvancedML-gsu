import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def hamming(data):
    distanceMatrix = np.zeros((data.shape[0],data.shape[0]))
    for a in range(data.shape[0]):
        for b in  range(data.shape[0]):
            distanceMatrix[a,b] = np.sum(data[a,:] != data[b,:])
            distanceMatrix[b,a] = distanceMatrix[a,b]
    return distanceMatrix

def testHamming():
    testHamming = np.array([[1,2,3,4],[1,2,3,4],[2,1,3,4]])
    assert np.sum(hamming(testHamming)) == 8

testHamming()


def PCAEIGS(data):

    if data.shape == (120,264):
        print('dna')

    A = data.T

    Amean = np.mean(A,axis = (1),keepdims=True)

    Abar = A-Amean #broadcasting occurs


    AbAbT = Abar.dot(Abar.T) #

    assert np.sum(AbAbT - AbAbT.T)**2 < 0.00001  # symmetric
    assert AbAbT.shape[0]==AbAbT.shape[1] #square matrix

    eVals,eVects = np.linalg.eig(AbAbT)

    return eVals,eVects, Abar


sequences = []
with open("HW4.fas",'r') as infile:
    for line in infile.readlines():
        sequence = True
        for char in line.strip():
            if not (char == 'A' or  char == 'C' or char == 'G' or char == 'T'):
                sequence = False
        if sequence:
            sequences.append(line)
            assert line == sequences[-1]
            assert len(line.strip()) == 264

dnaNumData = np.zeros((len(sequences),264))

row  = -1
for dna in sequences:
    row+=1
    column = -1
    for char in dna:
        column += 1
        if char == 'A':
            dnaNumData[row, column] = 1
        if char == 'T':
            dnaNumData[row, column] = 2
        if char == 'C':
            dnaNumData[row, column] = 3
        if char == 'G':
            dnaNumData[row, column] = 4


D = hamming(dnaNumData)
assert D.shape == (120,120)
###################################end data preprocessing


def testPca():
    testData = np.array([
        [1,1],
        [0,1],
        [2,1],
        [0,-1],
        [-3,1],
        [0,3]])

    assert testData.shape == (6,2)
    testEvals,testEvects, testAb = PCAEIGS(testData)
    expectedValues = (14,8)
    expectedVectors = [(1,0),(0,1)]

    args = list(np.argsort(testEvals))
    args.reverse()


    for index in args:
        assert testEvals[index] == expectedValues[index]
        for el in range(len(testEvects[index])):
            assert el == 0 or el == 1
            assert testEvects[index][el] == expectedVectors[index][el]


testPca()


def dnaPca():
    dnaEigVal,dnaEigVect,dnaAb = PCAEIGS(D)

    assert np.sum(np.abs(dnaEigVal.imag)) < 0.0001 # imaginary part of complex numbers is nearly zero
    args = list(np.argsort(dnaEigVal))
    #print(args)
    args.reverse()

    args = args[:2]

    #print("args", args)

    newData = dnaEigVect[:2].dot(dnaAb)


    colors = [[0,0,0]]

    plt.scatter(dnaEigVect[0].dot(dnaAb), dnaEigVect[1].dot(dnaAb), c=colors, alpha=0.5)
    plt.title('PCA - numpy - buggy')#
    #plt.show()
    plt.savefig('pca-numpy-buggy.png')
    plt.clf()


dnaPca()

def mds():
    a = -.5*D**2

    c = np.zeros((D.shape[0],D.shape[0]))

    a00 = np.mean(a)
    ai0 = np.mean(a,axis = 1,keepdims=True)
    a0j = np.mean(a,axis = 0,keepdims=True)

    assert list(ai0.shape) == list(a0j.shape)[::-1]

    c2 = a - ai0 - a0j + a00 # broadcasting occurs multiple times

    assert c2.shape == c.shape

    mdsEigs = np.linalg.eig(c2)
    args = np.argsort(mdsEigs[0])

    evals = mdsEigs[0][args]
    evects = mdsEigs[1][args]
    assert len(evects[0]) == 120
    #print(evals[-2:])
    #print (max(evals))
    #print(evects[-2])

    # Create data
    colors = [[0,0,0]]

    # Plot
    plt.scatter(evects[-1], evects[-2], c=colors, alpha=0.5) #largest two eigenvectors become x and y axis
    plt.title('mds - numpy - buggy')
    #plt.show()
    plt.savefig('mds-numpy-buggy.png')
    plt.clf()


mds()#buggy, i ran out of time to figure out why

def sklearnPca():
    # https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html
    pca = PCA(n_components=2)
    pca.fit(dnaNumData)

    X_pca = pca.transform(dnaNumData)

    plt.scatter(X_pca[:,0], X_pca[:, 1])
    plt.title("sklearn pca - correct")
    #plt.show()
    plt.savefig('sklearn-pca-correct.png')
    plt.clf()
    return X_pca

xyPca = sklearnPca()

def sklearnMds():
    #https://jakevdp.github.io/PythonDataScienceHandbook/05.10-manifold-learning.html
    model = MDS(n_components=2, dissimilarity='precomputed', random_state=1)
    mdsOut = model.fit_transform(D)
    plt.scatter(mdsOut[:, 0], mdsOut[:, 1])
    plt.title("mds sklearn - correct")
    #plt.show()
    plt.savefig('mds-sklearn-correct.png')
    plt.clf()
    return mdsOut #x,y

xyMds = sklearnMds()

def sklearnKmeans(xy,title):
    #https://jakevdp.github.io/PythonDataScienceHandbook/05.11-k-means.html
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(xy)
    y_kmeans = kmeans.predict(xy)
    plt.scatter(xy[:, 0], xy[:, 1], c=y_kmeans, s=50, cmap='viridis')
    plt.title(title)
    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
    #plt.show()
    return plt

p = sklearnKmeans(xyPca, "pca kmeans sklearn - correct")
p.savefig('pca-kmeans-sklearn-correct.png')
p.clf()

p = sklearnKmeans(xyMds, "mds kmeans sklearn - correct")
p.savefig('mds-kmeans-sklearn-correct.png')
p.clf()
###########going to attempt numpy kmeans. tired of debugging the numpy pca and mds

#generate seeds

def generateSeeds(xy,k = 3): #seems to depend on initialization to a great degree.
#    seedArgs = np.random.choice(xy.shape[0],k, replace=False)
#    return xy[seedArgs,:]
    return np.random.randn(k,2)*5.0

xy = np.array([[0,1],[0,2],[1,1]])
seeds = generateSeeds(xy)
#print(dm)
#print(seeds)
def findClusters(seeds, xy ):
    #get distance from every point to every seed
    dm = np.zeros((xy.shape[0],seeds.shape[0]))
    for i in range(xy.shape[0]): #k
        for j in range(seeds.shape[0]): #number of points
            diff = xy[i, :] - seeds[j, :]
            dist = np.sqrt(diff.dot(diff.T))
            dm[i,j] = dist
    clusters = np.argmin(dm, axis = 1)
    assert clusters.shape == (xy.shape[0],)
    return clusters

clusters = findClusters(seeds,xy)

#print(clusters)

def findCentroid(cluster):
    if len(cluster) == 0:
        raise ValueError('cluster has zero values/bug, rerun')

    assert cluster.shape[1] == 2

    return np.mean(cluster,axis = 0)

testCluster = xy

#print(xy)
def findCentroids(xy,clusterArgs):
    assert xy.shape[1] == 2
    k = np.max(clusterArgs) + 1
    newSeeds = np.zeros((k,2))
    for i in range(k):
        newSeeds[i,:] = findCentroid(xy[clusterArgs == i])
        #print(newSeeds[i,:])
    return newSeeds

seed = findCentroids(xy,np.array([0,1,1]))

#print (seeds)
def kMeans(xy,k):
    seeds = None
    oldCentroids = None
    centroids = None

    while True:
        try:

            seeds = generateSeeds(xy,k)

            oldCentroids = seeds
            centroids = oldCentroids + 10 # so loop will start
            while np.sum(np.abs(oldCentroids - centroids)) > .1:
                oldCentroids = centroids
                clusterArgs = findClusters(seeds, xy)
                centroids = findCentroids(xy, clusterArgs)
            break

        except:
            print('bad initialization, rerunning kmeans')
            continue



    return centroids, clusterArgs, xy

#print(kMeans(xy,2))


centroids, clusterArgs, xy = kMeans(xyPca,3)

def plotKmeans(centroids, clusterArgs, xy, title):
    #print(centroids)
    x = xy[:,0]
    y = xy[:,1]
    #plt.scatter(x[clusterArgs==0], y[clusterArgs==0])
#    plt.scatter(x,y)
    #plt.title("cluster 0")
    #plt.show()
    fig, axs = plt.subplots(1, sharex=True, sharey=True)
    #fig, ax = plt.subplots()
    #ax.scatter(x, y)
    axs.scatter(x[clusterArgs==0],y[clusterArgs==0])
    axs.scatter(x[clusterArgs==1],y[clusterArgs==1])
    axs.scatter(x[clusterArgs==2],y[clusterArgs==2])
    #.set_title('A single plot')
    axs.title.set_text(title)
    #plt.show()
    return plt

p = plotKmeans(centroids,clusterArgs,xy, "kmeans from scratch (buggy) - pca sklearn (correct) ")
p.savefig("kmeans-scratch-pca-sklearn.png")
p.clf()
centroids, clusterArgs, xy = kMeans(xyMds,3)
p = plotKmeans(centroids,clusterArgs,xy, "kmeans from scratch (buggy) - mds sklearn (correct) ")
p.savefig("kmeans-scratch-mds-sklearn.png")
p.clf()