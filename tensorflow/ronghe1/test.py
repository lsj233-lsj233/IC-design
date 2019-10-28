import os
import numpy as np
import tensorflow as tf
import getfiles
from PIL import Image
import matplotlib.pyplot as plt
import ronghe_inference
from getfiles import get_files
from getfiles import read_image

#变量声明
IMG_W = 48   # resize图像，太大的话训练时间久
IMG_H = 48

logs_train_dir = '/data0/lishuaijun/tensorflow/ronghe1/model'    #logs存储路径
test_dir = '/data0/lishuaijun/tensorflow/ronghe1/wenjian/testimg2/3'
#testresult_dir = test_dir
testresult_dir = '/data0/lishuaijun/tensorflow/ronghe1/wenjian/testresultimg2'
val, val_label = get_files(test_dir)



gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=True,
                                              gpu_options=gpu_options)) 
#sess.close()

x = tf.placeholder(tf.float32, shape=[1,IMG_W, IMG_H, 11])
#img_dir = x
#img = Image.open(img_dir)
#image = np.array(x)
#image = tf.cast(image, tf.float32)
#image = tf.reshape(image, [1, IMG_W,IMG_H, 3])

logit1 = ronghe_inference.inference(x,0)
#logit = tf.nn.softmax(logit1)
#init = tf.global_variables_initializer()
#sess.run(init)

saver = tf.train.Saver()
#with tf.Session() as sess:
print("Reading checkpoints...")
ckpt = tf.train.get_checkpoint_state(logs_train_dir)
if ckpt and ckpt.model_checkpoint_path:
    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    saver.restore(sess, ckpt.model_checkpoint_path)
    print('Loading success, global_step is %s' % global_step)
else:
    print('No checkpoint file found')



#print('test_accuracy = %.6f' %(sess.run(test_acc)))
n = len(val)
print(n)
t = 0
sess1 = tf.InteractiveSession()
for i in range(n):
	img_dir = val[i]
	image_1 = tf.cast(read_image(img_dir[8]),dtype = tf.float32)
	image_2 = tf.cast(read_image(img_dir[9]),dtype = tf.float32)
	image_3 = tf.cast(read_image(img_dir[10]),dtype = tf.float32)
	image_4 = tf.cast(read_image(img_dir[0]),dtype = tf.float32)
	image_5 = tf.cast(read_image(img_dir[1]),dtype = tf.float32)
	image_6 = tf.cast(read_image(img_dir[2]),dtype = tf.float32)
	image_7 = tf.cast(read_image(img_dir[3]),dtype = tf.float32)
	image_8 = tf.cast(read_image(img_dir[4]),dtype = tf.float32)
	image_9 = tf.cast(read_image(img_dir[5]),dtype = tf.float32)
	image_10 = tf.cast(read_image(img_dir[6]),dtype = tf.float32)
	image_11 = tf.cast(read_image(img_dir[7]),dtype = tf.float32)
	#lable = np.multiply(image_1, 0.05)+np.multiply(image_2, 0.05)+np.multiply(image_3, 0.05)+np.multiply(image_4, 0.05)+np.multiply(image_5, 0.05)+np.multiply(image_6, 0.05)+np.multiply(image_7, 0.05)+np.multiply(image_8, 0.05)+np.multiply(image_9, 0.05)+np.multiply(image_10, 0.05)+np.multiply(image_11, 0.05)
	lable1 = (image_1+image_2+image_3+image_4+image_5+image_6+image_7+image_8+image_9+image_10+image_11)/20
	#lable1 = tf.add(tf.cast(image_1,dtype = tf.float32),tf.cast(image_2,dtype = tf.float32))/2
	lable = lable1.eval()
	#lable = lable1.mul(1).byte()
	#lable = lable.cpu().numpy().squeeze(0).transpose((1, 2, 0))
	lable = lable.astype(np.uint8)
	lable = lable.reshape( IMG_W,IMG_H)
	lable = Image.fromarray(lable)
	filename = testresult_dir + '/' + 'lable.jpg'
	lable.save(filename)
	
	
	test_image = tf.concat([image_1, image_2, image_3, image_4, image_5, image_6, image_7, image_8, image_9, image_10, image_11],2)

	test_image = tf.image.per_image_standardization(test_image)
	#test_image = tf.cast(test_image,dtype = tf.float32)
	test_image = tf.reshape(test_image, [1, IMG_W,IMG_H, 11])


	prediction = sess.run(logit1, feed_dict={x: sess.run(test_image)})
	
	
	
	
	image = np.multiply(prediction, 127)+127
	image = image.astype(np.uint8)
	image = image.reshape( IMG_W,IMG_H)
	im = Image.fromarray(image)
	filename = testresult_dir + '/' + str(i) + '.jpg'
	im.save(filename)
	print('Done: ',i/n)
    #max_index = np.argmax(prediction)
    ##print('befor:',befor,'after',after)
    #print('lable : ',val_label[i],'  logit:', max_index,'out: [%3.4f  %3.4f  %3.4f]'%(prediction[0][0],prediction[0][1],prediction[0][2]), 'Done: %3d%%' %(i/n*100))
    #if max_index == val_label[i]:
    #    t = t + 1
    #else:
    #    t = t
#print('test_accuracy = %.6f' %(t/n))


















