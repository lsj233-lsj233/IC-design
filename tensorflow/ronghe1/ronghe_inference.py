# _*_ coding: utf-8 _*_
import tensorflow as tf

# 配置神经网络的参数

IMAGE_SIZE = 48
NUM_CHANNELS = 11
NUM_LABELS = 1

# 第1个卷积层的尺寸和深度
CONV1_DEEP = 56
CONV1_SIZE = 3
# 第2个卷积层的尺寸和深度
CONV2_DEEP = 12
CONV2_SIZE = 1
# 第3个卷积层的尺寸和深度
CONV3_DEEP = 12
CONV3_SIZE = 3
# 第4个卷积层的尺寸和深度
CONV4_DEEP = 12
CONV4_SIZE = 3
# 第5个卷积层的尺寸和深度
CONV5_DEEP = 12
CONV5_SIZE = 3
# 第6个卷积层的尺寸和深度
CONV6_DEEP = 12
CONV6_SIZE = 3
# 第7个卷积层的尺寸和深度
CONV7_DEEP = 56
CONV7_SIZE = 1
# 第8个卷积层的尺寸和深度
CONV8_DEEP = 1
CONV8_SIZE = 3

def floatloss(data_vars):
	#return tf.cast(tf.cast(data_vars*127,dtype = tf.int32),dtype = tf.float32 )
	return data_vars*127
def tru2int8(data_vars):
	data_max = tf.constant(value = 127, dtype = tf.float32,shape = data_vars.get_shape())
	data_min = tf.constant(value = -128, dtype = tf.float32,shape = data_vars.get_shape())
	return tf.where(data_vars>127,data_max,tf.where(data_vars<-128,data_min,data_vars))   
def cvt(imgs):
	scale = tf.get_variable(name='scale',shape=[1],initializer=tf.constant_initializer(0.03),dtype = tf.float32)
	shift = tf.get_variable(name='shift',shape=[1],initializer=tf.constant_initializer(0.07),dtype = tf.float32)
	imgs = (imgs*scale*100)/(2**(shift*100))
	imgs = tru2int8(tf.cast(imgs,dtype=tf.float32))
	return imgs

def prelu(_x):
	alphas = tf.get_variable('alphas', _x.get_shape()[-1],initializer=tf.truncated_normal_initializer(stddev=0.0),dtype=tf.float32)
	pos = tf.nn.relu(_x)
	neg = alphas * (_x - abs(_x)) * 0.5
	return pos+neg

# 定义卷积神经网络的前向传播过程。这里添加了一个新的参数train，用于区别训练过程和测试过程。在这个程序中将用到dropout方法
# dropout可以进一步提升模型可靠性并防止过拟合（dropout过程只在训练时使用）
def inference(input_tensor, train):

	with tf.variable_scope('layer1-conv1'):
		conv1_weights_0 = tf.get_variable('weight', [CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP],initializer=tf.truncated_normal_initializer(stddev=0.1))
		conv1_biases_0 = tf.get_variable('bias', [CONV1_DEEP],
										initializer=tf.constant_initializer(0.0))
		conv1 = tf.nn.conv2d(input_tensor, conv1_weights_0, strides=[1, 1, 1, 1], padding='SAME')
		#relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases_0))
		prelu1 = prelu(tf.nn.bias_add(conv1, conv1_biases_0))
		#conv1_weights = floatloss(conv1_weights_0);
		#conv1_biases = floatloss(conv1_biases_0);
		#conv1 = tf.nn.conv2d(input_tensor-127, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
		#relu1_0 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))
		#conv1 = cvt(conv1)

	with tf.variable_scope('layer2-conv2'):
		conv2_weights_0 = tf.get_variable('weight', [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP],
										initializer=tf.truncated_normal_initializer(stddev=0.1))
		conv2_biases_0 = tf.get_variable('bias', [CONV2_DEEP],
										initializer=tf.constant_initializer(0.0))
		conv2 = tf.nn.conv2d(prelu1, conv2_weights_0, strides=[1, 1, 1, 1], padding='SAME')
		#relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases_0))
		prelu2 = prelu(tf.nn.bias_add(conv2, conv2_biases_0))
		#conv2_weights = floatloss(conv2_weights_0);
		#conv2_biases = floatloss(conv2_biases_0);
		#conv2 = tf.nn.conv2d(conv1,conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
		#relu2_0 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))
		#conv2 = cvt(conv2)
		
	with tf.variable_scope('layer3-conv3'):
		conv3_weights_0 = tf.get_variable('weight', [CONV3_SIZE, CONV3_SIZE, CONV2_DEEP, CONV3_DEEP],
								initializer=tf.truncated_normal_initializer(stddev=0.1))
		conv3_biases_0 = tf.get_variable('bias', [CONV3_DEEP],
								initializer=tf.constant_initializer(0.0))
		conv3 = tf.nn.conv2d(prelu2, conv3_weights_0, strides=[1, 1, 1, 1], padding='SAME')
		#relu3 = tf.nn.relu(tf.nn.bias_add(conv3, conv3_biases_0))
		prelu3 = prelu(tf.nn.bias_add(conv3, conv3_biases_0))
		#conv3_weights = floatloss(conv3_weights_0);
		#conv3_biases = floatloss(conv3_biases_0);
		#conv3 = tf.nn.conv2d(conv2,conv3_weights, strides=[1, 1, 1, 1], padding='SAME')
		#conv3 = cvt(conv3)
		
	with tf.variable_scope('layer4-conv4'):
		conv4_weights_0 = tf.get_variable('weight', [CONV4_SIZE, CONV4_SIZE, CONV3_DEEP, CONV4_DEEP],
							initializer=tf.truncated_normal_initializer(stddev=0.1))
		conv4_biases_0 = tf.get_variable('bias', [CONV4_DEEP],
							initializer=tf.constant_initializer(0.0))
		conv4 = tf.nn.conv2d(prelu3, conv4_weights_0, strides=[1, 1, 1, 1], padding='SAME')
		#relu4 = tf.nn.relu(tf.nn.bias_add(conv4, conv4_biases_0))
		prelu4 = prelu(tf.nn.bias_add(conv4, conv4_biases_0))
		#conv4_weights = floatloss(conv4_weights_0);
		#conv4_biases = floatloss(conv4_biases_0);
		#conv4 = tf.nn.conv2d(conv3,conv4_weights, strides=[1, 1, 1, 1], padding='SAME')
		#conv4 = cvt(conv4)

	with tf.variable_scope('layer5-conv5'):
		conv5_weights_0 = tf.get_variable('weight', [CONV5_SIZE, CONV5_SIZE, CONV4_DEEP, CONV5_DEEP],
							initializer=tf.truncated_normal_initializer(stddev=0.1))
		conv5_biases_0 = tf.get_variable('bias', [CONV5_DEEP],
							initializer=tf.constant_initializer(0.0))
		conv5 = tf.nn.conv2d(prelu4, conv5_weights_0, strides=[1, 1, 1, 1], padding='SAME')
		#relu5 = tf.nn.relu(tf.nn.bias_add(conv5, conv5_biases_0))
		prelu5 = prelu(tf.nn.bias_add(conv5, conv5_biases_0))
		#conv5_weights = floatloss(conv5_weights_0);
		#conv5_biases = floatloss(conv5_biases_0);
		#conv5 = tf.nn.conv2d(conv4,conv5_weights, strides=[1, 1, 1, 1], padding='SAME')
		#conv5 = cvt(conv5)
		
	with tf.variable_scope('layer6-conv6'):
		conv6_weights_0 = tf.get_variable('weight', [CONV6_SIZE, CONV6_SIZE, CONV5_DEEP, CONV6_DEEP],
							initializer=tf.truncated_normal_initializer(stddev=0.1))
		conv6_biases_0 = tf.get_variable('bias', [CONV6_DEEP],
							initializer=tf.constant_initializer(0.0))
		conv6 = tf.nn.conv2d(prelu5, conv6_weights_0, strides=[1, 1, 1, 1], padding='SAME')
		#relu6 = tf.nn.relu(tf.nn.bias_add(conv6, conv6_biases_0))
		prelu6 = prelu(tf.nn.bias_add(conv6, conv6_biases_0))
		#conv6_weights = floatloss(conv6_weights_0);
		#conv6_biases = floatloss(conv6_biases_0);
		#conv6 = tf.nn.conv2d(conv3,conv6_weights, strides=[1, 1, 1, 1], padding='SAME')
		#conv6 = cvt(conv6)
		
	with tf.variable_scope('layer7-conv7'):
		conv7_weights_0 = tf.get_variable('weight', [CONV7_SIZE, CONV7_SIZE, CONV6_DEEP, CONV7_DEEP],
							initializer=tf.truncated_normal_initializer(stddev=0.1))
		conv7_biases_0 = tf.get_variable('bias', [CONV7_DEEP],
							initializer=tf.constant_initializer(0.0))
		conv7 = tf.nn.conv2d(prelu6, conv7_weights_0, strides=[1, 1, 1, 1], padding='SAME')
		#relu7 = tf.nn.relu(tf.nn.bias_add(conv7, conv7_biases_0))
		prelu7 = prelu(tf.nn.bias_add(conv7, conv7_biases_0))
		#conv7_weights = floatloss(conv7_weights_0);
		#conv7_biases = floatloss(conv7_biases_0);
		#conv7 = tf.nn.conv2d(conv6,conv7_weights, strides=[1, 1, 1, 1], padding='SAME')
		#conv7 = cvt(conv7)
		if train:
			prelu7 = tf.nn.dropout(prelu7, 0.2)
		
	with tf.variable_scope('layer8-conv8'):
		conv8_weights_0 = tf.get_variable('weight', [CONV8_SIZE, CONV8_SIZE, CONV7_DEEP, CONV8_DEEP],
							initializer=tf.truncated_normal_initializer(stddev=0.1))
		conv8_biases_0 = tf.get_variable('bias', [CONV8_DEEP],
							initializer=tf.constant_initializer(0.0))
		conv8 = tf.nn.conv2d(prelu7 ,conv8_weights_0, strides=[1, 1, 1, 1], padding='SAME')
		#relu8 = tf.nn.relu(tf.nn.bias_add(conv8, conv8_biases_0))		
		#conv8_weights = floatloss(conv8_weights_0);
		#conv8_biases = floatloss(conv8_biases_0);
		#conv8 = tf.nn.conv2d(conv7,conv8_weights, strides=[1, 1, 1, 1], padding='SAME')
		#conv8 = cvt(conv8)		

	return conv8
	

#loss计算
    #传入参数：logits，网络计算输出值。labels，真实值，在这里是0或者1(还有2)
    #返回参数：loss，损失值
def losses(logits, labels):
	with tf.variable_scope('loss') as scope:
		#cross_entropy =tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='xentropy_per_example')
		cross_entropy = tf.square(logits - labels)
		loss = tf.reduce_mean(cross_entropy, name='loss')
		tf.summary.scalar(scope.name+'/loss', loss)
	return loss
 
#--------------------------------------------------------------------------
#loss损失值优化
    #输入参数：loss。learning_rate，学习速率。
    #返回参数：train_op，训练op，这个参数要输入sess.run中让模型去训练。
def trainning(loss, learning_rate):
	with tf.name_scope('optimizer'):
		optimizer = tf.train.AdamOptimizer(learning_rate= learning_rate)
		global_step = tf.Variable(0, name='global_step', trainable=False)
		train_op = optimizer.minimize(loss, global_step= global_step)

		#global_step = tf.Variable(0, name='global_step', trainable=False)
		#increment_global_step = tf.assign(global_step, global_step + 1)
		#learning_rate = tf.train.exponential_decay(learning_rate, global_step, 10*34,0.90, staircase=True)
		#optimizer = tf.train.AdamOptimizer(learning_rate= learning_rate)
		#train_op = optimizer.minimize(loss)
	return train_op
 
#-----------------------------------------------------------------------
#评价/准确率计算
    #输入参数：logits，网络计算值。labels，标签，也就是真实值，在这里是0或者1。
    #返回参数：accuracy，当前step的平均准确率，也就是在这些batch中多少张图片被正确分类了。
#def evaluation(logits, labels):
#    with tf.variable_scope('accuracy') as scope:
#        correct = tf.nn.in_top_k(logits, labels, 1)
#        correct = tf.cast(correct, tf.float16)
#        accuracy = tf.reduce_mean(correct)
#        tf.summary.scalar(scope.name+'/accuracy', accuracy)
#    return accuracy

#pool2 = tf.Variable([[[[1,2,3],[4,5,6],[7,8,9]],[[-1,-2,-3],[-4,-5,-6],[-7,-8,-9]]],[[[11,12,13],[14,15,16],[17,18,19]],[[-11,-12,-13],[-14,-15,-16],[-17,-18,-19]]]])



