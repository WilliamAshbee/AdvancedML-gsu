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

    A = data.T

    Amean = np.mean(A,axis = (1),keepdims=True)

    Abar = A-Amean

    AbAbT = Abar.dot(Abar.T)
    assert AbAbT.shape[0]==AbAbT.shape[1]
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
        assert testEigs[1][index][el] == expectedVectors[index][el]


print(testEigs[0])

dnaEigs,dnaAb = PCAEIGS(dnaNumData)
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
#N = 500
#x = np.random.rand(N)
#y = np.random.rand(N)
colors = [[0,0,0]]
area = np.pi*3

# Plot
plt.scatter(newData[0,:], newData[1,:], c=colors, alpha=0.5)
plt.title('PCA')#
plt.xlabel('x')
plt.ylabel('y')
plt.show()
print(newData.shape)

#mds
