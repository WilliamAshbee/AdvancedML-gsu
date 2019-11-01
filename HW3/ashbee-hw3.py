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




def PCAEIGS(data):

    if data.shape == (120,264):
        print('dna')

    A = data.T

    Amean = np.mean(A,axis = (1),keepdims=True)

    Abar = A-Amean #broadcasting occurs

    AbAbT = Abar.dot(Abar.T) #

    assert AbAbT.shape[0]==AbAbT.shape[1] #square matrix

    eigs = np.linalg.eig(AbAbT)

    return (eigs), Abar



testData = np.array([
    [1,1],
    [0,1],
    [2,1],
    [0,-1],
    [-3,1],
    [0,3]])

testEigs, testAb = PCAEIGS(testData)
print(testEigs)
expectedValues = (14,8)
expectedVectors = [(1,0),(0,1)]

args = list(np.argsort(testEigs[0]))
args.reverse()
#for val in sorted(testEigs[0],reverse = True):
#    print(val)


for index in args:
    assert testEigs[0][index] == expectedValues[index]
    for el in range(len(testEigs[1][index])):
        assert el == 0 or el == 1
        assert testEigs[1][index][el] == expectedVectors[index][el]
        print (el)

print(testEigs[0])

dnaEigs,dnaAb = PCAEIGS(dnaNumData)

print("eigenvalues", dnaEigs[0].shape, dnaEigs[0])
print("total complex", np.sum(np.abs(np.absolute(dnaEigs[0]))))
print("max complex", np.max(np.abs(np.absolute(dnaEigs[0]))))
print("totalComplex")
#assert np.sum(np.abs(np.absolute(dnaEigs[0]))) < .00001
args = list(np.argsort(dnaEigs[0]))
args.reverse()

args = args[:2]


#print(np.reshape(dnaEigs[1][0],(1,-1)).shape)
print(dnaEigs[1][args].shape)

newData = dnaEigs[1][args].dot(dnaAb)
#print(vectors.dot(dnaAb))

#import matplotlib
import matplotlib.pyplot as plt

# Create data
colors = [[0,0,0]]

# Plot
plt.scatter(newData[0,:], newData[1,:], c=colors, alpha=0.5)
plt.title('PCA')#
plt.xlabel('x')
plt.ylabel('y')
plt.show()
print(newData.shape)

#mds

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
assert D.shape == (120,120)

a = -.5*D**2

c = np.zeros((D.shape[0],D.shape[0]))
#c = np.zeros((D.shape[0],D.shape[0]))

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
print(evals[-2:])
print (max(evals))
print(evects[-2])

#for i in range(D.shape[0]):
#    ai0 = np.sum(a,axis = )
#    for j in range(D.shape[0]):
#        cij = a[i,j] - ai0 - a0j + a00

# Plot
plt.scatter(evects[-1], evects[-2], c=colors, alpha=0.5) #largest two eigenvectors become x and y axis
plt.title('mds')
plt.xlabel('x')
plt.ylabel('y')
plt.show()



import matplotlib.pyplot as plt
from sklearn.manifold import MDS

model = MDS(n_components=2, dissimilarity='precomputed', random_state=1)
out = model.fit_transform(D)
plt.scatter(out[:, 0], out[:, 1])
plt.axis('equal')
plt.show()


#https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html
#https://jakevdp.github.io/PythonDataScienceHandbook/05.10-manifold-learning.html