import numpy as np
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


def hamming(data):
    distanceMatrix = np.zeros((data.shape[0],data.shape[0]))
    for a in range(data.shape[0]):
        for b in  range(data.shape[0]):
            distanceMatrix[a,b] = np.sum(data[a,:] != data[b,:])
            distanceMatrix[b,a] = distanceMatrix[a,b]
    return distanceMatrix


testHamming = np.array([[1,2,3,4],[1,2,3,4],[2,1,3,4]])



assert np.sum(hamming(testHamming)) == 8

D = hamming(dnaNumData)
import matplotlib.pyplot as plt
from sklearn.manifold import MDS

model = MDS(n_components=2, dissimilarity='precomputed', random_state=1)
out = model.fit_transform(D)
plt.scatter(out[:, 0], out[:, 1])
plt.axis('equal');
plt.show()

