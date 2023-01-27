import numpy as np
from scipy.special import expit
import matplotlib.pyplot as plt
def activation_function(activation) :
	if activation == "binary" :
		return lambda x: np.where(x<=0,0,1)
	elif activation == "sigmoid":
		return lambda x : expit(x)
	elif activation == "tanh":
		return lambda x : np.tanh(x)
	elif activation == "relu" :
		return lambda x: np.where(x<=0,0,x)
	elif activation == 'leakyRelu' :
		return lambda x: np.where(x<=0,0.01*x,x)


def activation_derivative(activation_function):
	if activation_function == "binary":
		return lambda x: np.where(x<=0,0,0)
	elif activation_function == "sigmoid":
		return lambda x: x*(1-x)
	elif activation_function == "tanh" :
		return lambda x : x * (1-x*x)
	elif activation_function == "relu" :
		return lambda x: np.where(x<=0,0,1)
	elif activation_function == "leakyRelu" :
		return lambda x: np.where(x<=0,0.01,1)

def weights_initializer(initializer,input_dim,output_dim):
	if initializer == 'gaussian' :
		return np.random.normal(0.0,pow(output_dim,-0.5),(input_dim,output_dim))
	if initializer == 'uniform' :
		return np.random.rand(input_dim,output_dim)-0.5

def confusion_matrix(y_test,y_pred):
	correct = 0
	incorrect = 0
	for i in range(len(y_test)):
		if y_test[i] == y_pred[i]:
			correct = correct+1
		else:
			incorrect = incorrect+1
	print("total correct = ",correct)
	print("total incorrect = ",incorrect)
	print("performence : ",(correct/(correct+incorrect))*100,"%")

def performence(y_test,y_pred):
	correct = 0
	incorrect = 0
	for i in range(len(y_test)):
		if y_test[i] == y_pred[i]:
			correct = correct+1
		else:
			incorrect = incorrect+1
	return round((correct/(correct+incorrect))*100,2)

def binary_confusion_matrix(y_test,y_pred):
	zbuto = 0
	obutz = 0
	zbutz = 0
	obuto = 0
	for i in range(len(y_test)):
		if y_test[i]==0 and y_pred[i] == 0 :
			zbutz = zbutz + 1
		if y_test[i]==0 and y_pred[i] == 1 :
			zbuto = zbuto + 1
		if y_test[i]==1 and y_pred[i]== 0 :
			obutz = obutz + 1
		if y_test[i]==1 and y_pred[i]== 1 :
			obuto = obuto + 1
	matrix = [[zbutz,zbuto],[obutz,obuto]]
	return np.array(matrix)

def accuracy_map(neural_network,costom,lw=1.0):
	accuracies = neural_network.accuracies
	plt.plot([i for i in range(len(accuracies))],accuracies, costom,lw =lw)
	plt.axis([0,len(accuracies), min(accuracies), max(accuracies)])
	plt.show()

