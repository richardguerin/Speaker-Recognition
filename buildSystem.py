import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn import neighbors
from sklearn import metrics
import sys

def readData(fname):
	data = []
	target = []
	for line in fname:
		parts = line.split(",")
		data.append([float(p) for p in parts[0:-1]])
		target.append(float(parts[-1]))
	npData = np.array(data)
	npTarget = np.array(target)

	return npData,npTarget

def main():
	labels = []
	trainFile = open(sys.argv[1])
	testFile = open(sys.argv[2])

	labels = trainFile.readline().rstrip().split(",")
	labels = testFile.readline().rstrip().split(",")

	trainSet, trainTarget = readData(trainFile)
	testSet, testTarget = readData(testFile)

	
	model = GaussianNB()
	model.fit(trainSet, trainTarget)

	predicted = model.predict(testSet)
	print(predicted)
	print(metrics.classification_report(testTarget, predicted))
	print(metrics.confusion_matrix(testTarget, predicted))
	
	print(model.score(testSet, testTarget))

main()
