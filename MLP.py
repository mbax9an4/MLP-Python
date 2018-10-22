import numpy as np
import random
import math

class MLP:

	#constructor that defines the size of the neural network, and randomly initialises the parameters we want to learn
	def __init__(self, featuresNo, hiddenNodes, classes):
		#the parameters we are trying to learn
		self.biasIH = 1
		self.biasHO = 1
		self.weightIH = np.random.rand(featuresNo, hiddenNodes)
		self.weightHO = np.random.rand(hiddenNodes, classes)

	#function to run through tanh the weighted sum of the previous layer
	def tanh(self, input):
		negI = input * -1
		return (np.exp(input) - np.exp(negI)) / (np.exp(input) + np.exp(negI))

	#function to run through sigmoid the weighted sum of the previous layer
	def sigmoid(self, input):
		negI = input * -1
		return 1 / (1 + np.exp(negI))

	#main caller function for the activation functions, so we can change easily between sigmoid, tanh and softmax
	def activationF(self, input, afIndex):
		if afIndex == 1: #tanh function
			output = self.tanh(input)
		else: #sigmoid function
			output = self.sigmoid(input)

		return output

	#main caller function for the activation functions for the backpropagation phase, when the derivativea are required
	def activationFDer(self, input, afIndex):
		if afIndex == 1: #tanh function
			output = self.tanh(input)
			output = 1 - np.power(output, 2)
		else: #sigmoid function
			output = self.sigmoid(input)
			output = output * (1 - output)

		return output

	#send + or - epsilon to generate the two predictions, method that simulates the forward propagation phase again but with the +- epsilon variations, it returns the probability of the predicted label
	def gradientChecking(self, example, epsilon, trueLabel):

		hiddFP = np.dot(np.transpose(self.weightIH), example.transpose()) + self.biasIH
		hiddFP = self.activationF(hiddFP,1)

		outputFP = np.dot(np.transpose(hiddFP), self.weightHO + epsilon) + self.biasHO
		label = self.activationF(outputFP,1).transpose()


		if int(trueLabel) == 0:
			return 1 - label[int(trueLabel)]
		else:
			return label[int(trueLabel)]


	#main train method that will train the model given
	def train(self, data, labels, epochs, lr):
		learningRate = lr

		#get the size of the neural network 
		(hiddenNodes, classes) = self.weightHO.shape
		(featuresNo, hiddenNodes) = self.weightIH.shape

		#initialise the update arrays, they will store the values we want to update with, at the end of the backpropagation phase
		deltaWeightHO = np.zeros((hiddenNodes, classes))
		deltaWeightIH = np.zeros((featuresNo, hiddenNodes))

		#parameters for gradient checking 
		(exNo,a) = data.shape
		diff = np.zeros(exNo*epochs)

		#get an order set of the labels existing in the labels set
		labelsSet = np.unique(labels)

		#train the network for a given number of epochs
		for epoch in range(epochs):
			iterator = 0

			#go through each example in the dataset
			for example in data:

				#forward propagation phase

				#compute the weighted sum of the input layer and the weights between the I-H layers and pass the result through the chosen activation function
				hiddFP = np.dot(np.transpose(self.weightIH), example) + self.biasIH
				hiddFP = self.activationF(hiddFP,1)

				#compute the weighted sum of the hidden layer and the weights between the H-O layers and pass the result through the softmax function to obtain probabilities
				outputFP = np.dot(np.transpose(hiddFP), self.weightHO) + self.biasHO
				predLabel = self.activationF(outputFP,3)

				#back propagation phase
				#get the true label for the current example from the labels array
				trueLabel = np.full(predLabel.shape,labels[iterator])

				#compute the error for the output layer, as given by the derivative, with the SSM error function
				outputErr = self.activationFDer(outputFP, 1).transpose() * (trueLabel - predLabel)

				#need to add an extra dimension so array is hiddenNodesx1 so it can be transposed
				hiddFP = hiddFP[np.newaxis]

				#compute the update for the weights between the O-H layers 
				deltaWeightHO = learningRate * outputErr * (hiddFP.transpose() * self.weightHO)

				#gradient checking array, since we don't want the learning rate
				gradCHO = outputErr * (hiddFP.transpose() * self.weightHO)

				#add another dimension explicitly so we can transpose
				outputErr = outputErr[np.newaxis]

				#compute the error for the hidden layer as given by the derivative with the SSM error function
				hiddErr = np.dot(outputErr, self.weightHO.transpose())
				hiddErr = self.activationFDer(hiddFP, 1) * hiddErr

				#add another dimension explicitly so we can transpose the array
				example = example[np.newaxis]

				#compute the update for the weights between the H-I layers 
				deltaWeightIH = learningRate * (example.transpose() * self.weightIH) * hiddErr

				#gradient checking array of the updates since we don't want the learning rate as well
				gradCIH = (example.transpose() * self.weightIH) * hiddErr


				#make the gradient descent correctness check
				epsilon = math.pow(10,-4)

				#compute the error function with the slight variation to the right and left given by epsilon
				errorP = np.power(int(labels[iterator]) - float(self.gradientChecking(example, epsilon, labels[iterator])),2)
				errorM = np.power(int(labels[iterator]) - float(self.gradientChecking(example, -epsilon, labels[iterator])),2)
				
				#can print the difference between the original gradient and the approximated on to check their similarity, should be very small
				# print gradCHO - (errorP/2 - errorM/2)/(2*epsilon)


				#update the weights using the updates computed before
				self.weightIH += deltaWeightIH
				self.weightHO += deltaWeightHO 

				iterator += 1

    #testing function to determine the number of examples the model has classified incorrectly
	def test(self, testSet, labels):
		#get the number of examples in the dataset
		(classes, a) = labels.shape

		#create the prediction array that will be returned 
		predLabel = np.zeros(classes)
		iterator = 0
		error = 0

		#get a sorted list of the possible labels in the dataset
		labelsSet = np.unique(labels)

		#for each example in the testing set 
		for example in testSet:

			#do the forward propagation phase for the example, with the trained weights
			hiddFP = np.dot(np.transpose(self.weightIH), example) + self.biasIH
			hiddFP = self.activationF(hiddFP,1)

			outputFP = np.dot(np.transpose(hiddFP), self.weightHO) + self.biasHO

			#determine the node with the largest probability 
			outputs = np.argmax(self.activationF(outputFP,1))

			#get the label associated with that node, that had the largest probability
			predLabel[iterator] = labelsSet[outputs]

			#check if the predicted and the true label are the the same, if not increase the error for the example
			if int(labels[iterator]) != int(predLabel[iterator]):
				error += 1

			iterator += 1

		return (error, predLabel)	