### -- sentdex --
'''
input > weight > 
hidden layer 1 (activation function > weights > 
hidden layer 2 (activation function) > weights > 
output layer

compare output to indended output > cost function (using cross entropy)
optimization function (optimizer) > minimize cost (AdamOptimizer...SGD, AdaGrad)

backpropagation 
feed forward + backprop = epoch
'''

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/",one_hot=True)  ### reading data set

n_nodes_hl1 = 100 	## 500 nodes on layer 1
n_nodes_hl2 = 100 	## '' ... layer 2
n_nodes_hl3 = 100 	## '' ... layer 3 (output layer)
n_classes = 10		## - there are 10 classifiers: one to ten Numbers
batch_size = 100	## - batch of "100" data will be processed at a time, inside a sweep


x = tf.placeholder('float',[None,784])  ## each image is 28 x 28 pixel = 784/image - input
y = tf.placeholder('float') 			## output

### --- set up the model structure = flow graph ----------
def neural_network_model(data):
	
	## --- setting up and allocating the layers!
	hidden_1_layer = {'weights': tf.Variable( tf.random_normal( [784,n_nodes_hl1] ) ),  
					  'biases':  tf.Variable( tf.random_normal( [n_nodes_hl1] ) )
					  }
					  
	hidden_2_layer = {'weights': tf.Variable( tf.random_normal( [n_nodes_hl2,n_nodes_hl1] ) ),
					  'biases':  tf.Variable( tf.random_normal( [n_nodes_hl2] ) )
					  }
					  
	hidden_3_layer = {'weights': tf.Variable( tf.random_normal( [n_nodes_hl2,n_nodes_hl3] ) ),
					  'biases':  tf.Variable( tf.random_normal( [n_nodes_hl3] ) )
					  }
					  
	output_layer = {'weights': tf.Variable( tf.random_normal( [n_nodes_hl1,n_classes] ) ),
					'biases':  tf.Variable( tf.random_normal( [n_classes] ) )
					}
		
		
	## inputdata*weights + biases
	l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])
	l1 = tf.nn.relu(l1) ## rectified linear - threshold function
	
	l2= tf.add(tf.matmul(l1,hidden_2_layer['weights']), hidden_2_layer['biases'])
	l2 = tf.nn.relu(l2) ## rectified linear - threshold function 
	
	l3= tf.add(tf.matmul(l2,hidden_3_layer['weights']), hidden_3_layer['biases'])
	l3 = tf.nn.relu(l3) ## rectified linear - threshold function 
	
	output = tf.matmul(l3,output_layer['weights']) + output_layer['biases']
	return output
	
### --- Actual model trainer ---
def train_neural_network(x):
	
	prediction = neural_network_model(x) ## - set up the 
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction,y))
	optimizer = tf.train.AdamOptimizer().minimize(cost) ## learning rate = 0.001
	
	
	hm_epochs = 100  ## number of sweeps for optimizing
	with tf.Session() as sess: ## The true calculations start!
		sess.run(tf.initialize_all_variables())
		
		for epoch in range(hm_epochs):
			epoch_loss = 0
			for _ in range(int(mnist.train.num_examples/batch_size)):
				epoch_x,epoch_y =mnist.train.next_batch(batch_size)
				_,c = sess.run([optimizer, cost], feed_dict = {x: epoch_x, y: epoch_y})
				epoch_loss += c
			print 'Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss
			
		correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
		accuracy = tf.reduce_mean(tf.cast(correct,'float'))
		print 'Accuracy:', accuracy.eval({x:mnist.test.images, y:mnist.test.labels})



train_neural_network(x)






