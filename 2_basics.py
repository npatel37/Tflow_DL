import tensorflow as tf

### -- this is called making a "graph" where you don't run anything!
x1 = tf.constant(5)
x2 = tf.constant(6)
result = tf.mul(x1,x2) ## multiplication
print(result)
## --- note: No computation has been done until now!


## -- To actually excecute the multiplication you have to run the session.
#~ sess = tf.Session()
#~ print (sess.run(result))

with tf.Session() as sess:
	output = sess.run(result)
	print(output)
	
print(output)
