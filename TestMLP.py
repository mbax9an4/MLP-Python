import numpy as np
from MLP import *
from UtilityFunctions import *
import scipy.io as sio
from sklearn import preprocessing

def main():
	#read data file from .mat format
	filename = "data/datab.mat"
	loadData = sio.loadmat(filename)

	#convert data from dictionary to numpy array
	data = np.asarray(loadData.get('data'))

	#scale the data so it has zero mean and unit standard deviation
	# data = preprocessing.scale(data)

	#sclae the data so it is between -1 and 1
	data = UtilityFunctions.scaleData(data)

	#read into array the labels data from file
	labels = np.asarray(loadData.get('labels'))

	#shuffle the data and labels arrays together so that the index correspondence is not lost
	[data, labels] = UtilityFunctions.shuffle_in_unison(data, labels)

	#split shuffled data into training and testing, done equally here, should change
	(trainingData, testingData) = np.vsplit(data, 2)
	(trainingLabels, testingLabels) = np.vsplit(labels, 2)

	#construct, train and test the neural network
	(examples, features) = data.shape
	(classes,) = np.unique(labels).shape

	#give the parameters optimised here for the neural network
	mlp = MLP(features, 20, classes)
	# print mlp.weightIH

	#train the defined neural network with the training set and labels, for a number of epochs and with a learning rate
	mlp.train(trainingData, trainingLabels, 200, 0.01)
	# print mlp.weightIH

	#test the model we have trained and get the error(not error rate) and the predicted labels for each example in the testing set given
	(error, predLabels) = mlp.test(testingData, testingLabels)

	print error


#needed in order to be ale to run main 
if __name__ == "__main__":
	main()