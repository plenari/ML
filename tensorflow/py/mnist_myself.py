import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#read
data_dir=r'MNIST_data'
mnist=input_data.read_data_sets(data_dir,one_hot=True)
#divied

#weights
def weights(shape):
    weight=tf.Variable(tf.truncated_normal(shape,stddev=0.1))
    return weight 
#bias
def bias(shape):
    bias=tf.constant(0.1,shape=shape)
    return bias
#conv2d
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pool(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

x=tf.placeholder(tf.float32,[None,784])
x_image=tf.reshape(x,[-1,28,28,1])

#first conv1
w_conv1=weights([5,5,1,32])
b_conv1=bias([32])
h_conv1=tf.nn.relu(conv2d(x_image,w_conv1)+b_conv1)
h_pool1=max_pool(h_conv1)#[-1,14,14,32]
#second conv2
w_conv2=weights([5,5,32,64])
b_conv2=bias([64])
h_conv2=tf.nn.relu(conv2d(h_pool1,w_conv2)+b_conv2)
h_pool2=max_pool(h_conv2)

#third full connection
w_fc1=weights([7*7*64,1024])
b_fc1=bias([1024])
h_pool2_flat=tf.reshape(h_pool2,[-1,7*7*64])
h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,w_fc1)+b_fc1)#[-1,1024]

#delete tf.nn.dropout()
keep_prob=tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
#full connection
w_fc2=weights([1024,10])
b_fc2=bias([10])

#activate
y_conv=tf.nn.softmax(tf.matmul(h_fc1,w_fc2)+b_fc2)#[N,10]
#
y_=tf.placeholder(tf.float32,[None,10])

#cost
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y_conv),reduction_indices=[1]))
#AdamOptimizer
train_step=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
#y_conv y_ 个数要相同
correct_predicition=tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))

accuracy=tf.reduce_mean(tf.cast(correct_predicition,tf.float32))
init_op=tf.global_variables_initializer()
sess=tf.Session()
sess.run(init_op)
for i in range(100):        
    batch=mnist.train.next_batch(40)
    sess.run(train_step,feed_dict={x:batch[0],y_:batch[1],keep_prob: 1.0})
    if i%100==0:
        train_accuracy=accuracy.eval(feed_dict={x:batch[0],y_:batch[1],keep_prob: 1.0},session=sess)
    print(train_accuracy)

print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0},session=sess))

sess.close()