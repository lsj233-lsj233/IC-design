import os
import numpy as np
import tensorflow as tf
import getfiles
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import ronghe_inference
from getfiles import get_files


#变量声明
N_CLASSES = 3  #husky,jiwawa,poodle,qiutian
IMG_W = 48   # resize图像，太大的话训练时间久
IMG_H = 48
BATCH_SIZE = 32
CAPACITY = 1593
MAX_STEP = 34 # 一般大于10K   训练集总共训练所有batch的次数
learning_rate = 0.001 # 一般小于0.0001
epoch = 1000

#获取批次batch
train_dir = '/data0/lishuaijun/tensorflow/ronghe1/wenjian/shujuji'     #训练样本的读入路径
logs_train_dir = '/data0/lishuaijun/tensorflow/ronghe1/model'    #logs存储路径
#logs_test_dir =  'E:/Re_train/image_data/test'        #logs存储路径
 
#train, train_label = input_data.get_files(train_dir)
train, train_label = getfiles.get_files(train_dir)
#训练数据及标签
train_batch,train_label_batch = getfiles.get_batch(train, train_label, BATCH_SIZE, CAPACITY)

#input_queue = tf.train.slice_input_producer([train, train_label])
 
#train_batch, train_label_batch = tf.train.batch([input_queue[0], input_queue[1]],  
#                                                batch_size= BATCH_SIZE,  
#                                                num_threads= 1  ,   
#                                                capacity = CAPACITY) 

#训练操作定义
train_logits = ronghe_inference.inference(train_batch,1)
train_loss = ronghe_inference.losses(train_logits, train_label_batch)        
train_op = ronghe_inference.trainning(train_loss, learning_rate)
#train_acc = ronghe_inference.evaluation(train_logits, train_label_batch)


#这个是log汇总记录
summary_op = tf.summary.merge_all() 
 
#产生一个会话

gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=True,
                                              gpu_options=gpu_options))  
#产生一个writer来写log文件
train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph) 
#val_writer = tf.summary.FileWriter(logs_test_dir, sess.graph) 
#产生一个saver来存储训练好的模型
saver = tf.train.Saver()
#所有节点初始化
sess.run(tf.global_variables_initializer())  
#队列监控
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)
 
graph = tf.get_default_graph()
#scale = graph.get_tensor_by_name("conv1/scale:0")
#shift = graph.get_tensor_by_name("conv1/shift:0") 
#print('scale:%4.5f: '%(sess.run(scale)),'\tshift:%4.5f: '%(sess.run(shift))) 
#进行batch的训练
try:
    for epochtime in range(epoch):
	#执行MAX_STEP步的训练，一步一个batch
        for step in np.arange(MAX_STEP):
            if coord.should_stop():
                break
            #启动以下操作节点，有个疑问，为什么train_logits在这里没有开启？
            _, tra_loss = sess.run([train_op, train_loss])

            #每隔1步打印一次当前的loss以及acc，同时记录log，写入writer   
            if step % 1  == 0:
                print('epoch %4d, Step %3d, train loss = %5.4f' %(epochtime+1, step, tra_loss))
                summary_str = sess.run(summary_op)
                train_writer.add_summary(summary_str, step)
            #每隔MAX_STEP步，保存一次训练好的模型
        if epochtime == epoch-1:
            checkpoint_path = os.path.join(logs_train_dir, 'net.ckpt')
            saver.save(sess, checkpoint_path, global_step=step)
    checkpoint_path = os.path.join(logs_train_dir, 'net.ckpt')
    saver.save(sess, checkpoint_path, global_step=step)     
    
#    scale = graph.get_tensor_by_name("layer1-conv1/scale:0")
#    shift = graph.get_tensor_by_name("layer1-conv1/shift:0")
#    fc1 = sess.run(graph.get_tensor_by_name('layer5-fc1/weight:0'))
#    print('scale:%4.5f: '%(sess.run(scale)),'\tshift:%4.5f: '%(sess.run(shift)))
except tf.errors.OutOfRangeError:
    print('Done training -- epoch limit reached')
 
finally:
    coord.request_stop()




