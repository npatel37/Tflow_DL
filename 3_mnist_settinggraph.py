import tensorflow as tf

'''
input > 
weight > hidden layer 1 (activation function > 
weights > hidden layer 2 (activation function) >
weights > output layer

compare output to indended output > cost function (cross entropy)
optimization function (optimizer) > minimize cost (AdamOptimizer...SGD, AdaGrad)

backpropagation 

feed forward + backprop = epoch

'''

from tensorflow.examples.tutorials.mnist import input_data

## -- reading data set
mnist = input_data.read_data_sets("/tmp/data/",one_hot=True)
print " Reading data set mnist \n ", mnist


n_nodes_hl1 = 500 ## 500 nodes on layer 1
n_nodes_hl2 = 500 ## '' ... layer 2
n_nodes_hl3 = 500 ## '' ... layer 3 (output layer)

n_classes = 10
batch_size = 100

# height x width
x = tf.placeholder('float',[None,784])
y = tf.placeholder('float')

def neural_network_model(data):
	
	## --- setting up and allocating the layers!
	hidden_1_layer = {'weights': tf.Variable(tf.random_normal([784,n_node_hl1])),
					  'biases': tf.Variable(tf.random_normal([n_node_hl1]))}
					  
	hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_node_hl2,n_node_hl1])),
					  'biases': tf.Variable(tf.random_normal([n_node_hl2]))}
					  
	hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_node_hl2,n_node_hl3])),
					  'biases': tf.Variable(tf.random_normal([n_node_hl3]))}
					  
	output_layer = {
		'weights': tf.Variable(tf.random_normal([n_node_hl1,n_classes])),
		'biases': tf.Variable(tf.random_normal([n_classes]))
		}
		
		
	## inputdata*weights + biases
	l1 = tf.add(tf.mutmul(data,hidden_1_layer['weights']) + hidden_1_layer['biases'])
	l1 = tf.nn.relu(l1) ## rectified linear - threshold function
	
	l2= tf.add(tf.mutmul(l1,hidden_2_layer['weights']) + hidden_2_layer['biases'])
	l2 = tf.nn.relu(l2) ## rectified linear - threshold function 
	
	l3= tf.add(tf.mutmul(l2,hidden_3_layer['weights']) + hidden_3_layer['biases'])
	l3 = tf.nn.relu(l3) ## rectified linear - threshold function 
	
	output = tf.mutmul(l3,output_layer['weights']) + output_layer['biases']
	






