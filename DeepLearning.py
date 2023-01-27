import numpy as np
import SpecialFunction

# Neural network devloped using numpy librery 
class NeuralNetwork :
	# Initializing a neural network
	def __init__(self):
		self.layerGroup = []
		self.accuracies = []

	# Adding a neural layer to the neural network
	def add(self,layer):
		self.layerGroup.append(layer)

	# Forward propagation through all the layer
	def feed_forword_net(self,input_row):
		for layer in self.layerGroup :
			input_row = layer.feed_forward(input_row)
		return input_row

	# Back-propagation through all the layer
	def back_propagate_net(self,output_error):
		for i in reversed(range(len(self.layerGroup))) :
			output_error = self.layerGroup[i].back_propagate(output_error)
		return output_error

	# Weight rectification of all layer
	def update_weights_net(self,input_row):
		for layer in self.layerGroup :
			input_row = layer.update_weights(input_row)

	# Train the neural net with dataset includes forword propagation bacck-propagation and weight rectification
	# It also include the analization of all epoch with user-defined spilt
	def fit(self,X_train,y_train,epoch = 1,learning_rate = 0.01,eval_percent = 0.25):
		from sklearn.model_selection import train_test_split
		self.learning_rate = learning_rate
		for j in range(epoch):
			for i in range(len(X_train)):
				result = self.feed_forword_net(input_row = X_train[i])
				expected = [0 for i in range(self.layerGroup[-1].output_dim)]
				expected[y_train[i]] = 1
				output_error =expected - result
				output_error = output_error.reshape(len(output_error),1)
				self.back_propagate_net(output_error)
				input_data =np.array([X_train[i]])	
				input_data =input_data.reshape(len(input_data.ravel()),1)
				self.update_weights_net(input_row = input_data)
			X_tr, X_te, y_tr, y_te = train_test_split(X_train, y_train, test_size = eval_percent, random_state = np.random.randint(low=1,high = 2000))
			y_pr = self.predict(X_te)
			performence = SpecialFunction.performence(y_te,y_pr)
			self.accuracies.append(performence)
			print("epoch ",j," ======== > performence : ",performence,'%')

	# For predicting the testcase with trained network includes only forward propagation
	def predict(self,X_test):
		y_pred = []
		for row in X_test:
			result = self.feed_forword_net(input_row = row)
			max = -1
			max_i=-1
			for i in range(len(result)):
				if max <result[i] :
					max = result[i]
					max_i = i
			y_pred.append(max_i)
		return y_pred

# Layer are made of two dimensional matrix it is designed in such a way that user can add multiple layer each with uniq proporties
class Layer :
	# Initializing a neural layer
	def __init__(self,neural_network,output_dim,activation,initializer,input_dim="pre_output_dim",random_state = 1):
		self.neural_network = neural_network
		self.output_dim = output_dim
		self.initializer = initializer
		self.activation = activation
		self.input_dim =input_dim
		self.transfer = SpecialFunction.activation_function(activation = activation)
		self.transfer_derivative = SpecialFunction.activation_derivative(activation_function = activation)
		np.random.seed(random_state)
		if input_dim == "pre_output_dim" :
			self.input_dim =self.neural_network.layerGroup[-1].output_dim
		self.weights = SpecialFunction.weights_initializer(initializer =initializer,input_dim =self.input_dim,output_dim =self.output_dim)
	
	# Forward propagation	
	def feed_forward(self,input_row):
		self.output_row = self.transfer(np.dot(input_row,self.weights))
		return self.output_row
	# Backward propagation
	def back_propagate(self,output_error):
		self.deltas = output_error
		self.output_row = self.output_row.reshape(len(self.output_row.ravel()),1)
		return np.dot(self.weights,self.deltas)
	# Weight rectification
	def update_weights(self,input_row):
		self.weights += self.neural_network.learning_rate * np.dot(input_row,(self.deltas * self.transfer_derivative(self.output_row)).transpose())
		return self.output_row