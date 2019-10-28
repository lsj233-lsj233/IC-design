import os  
import math  
import numpy as np  
import tensorflow as tf  
import matplotlib.pyplot as plt   
from tensorflow.python.framework import ops
#============================================================================  
#-----------------生成图片路径和标签的List------------------------------------  
image_W = 48
image_H = 48

train_dir = '/data0/lishuaijun/tensorflow/ronghe1/wenjian/shujuji'  
  
baixubao = []
label = []  

def read_image(input_queue):
	image_contents = tf.read_file(input_queue) #read img from a queue    
	image_d1 = tf.image.decode_jpeg(image_contents, channels=3)
	image_d2 = tf.image.resize_image_with_crop_or_pad(image_d1, image_W, image_H) 
	image = tf.image.rgb_to_grayscale(image_d2)
	return image
#step1：获取'/home/fengwenting/tensorflow/baixibaofenlei/baixibao_tfre'下所有的图片路径名，存放到  
#对应的列表中，同时贴上标签，存放到label列表中。  
#def get_files(file_dir):  
#	filelist = os.listdir(file_dir)
#	image1 = read_image(file_dir+'/'+filelist[0])	
#	image2 = read_image(file_dir+'/'+filelist[1])
#	image3 = read_image(file_dir+'/'+filelist[2])
#	image4 = read_image(file_dir+'/'+filelist[3])
#	image5 = read_image(file_dir+'/'+filelist[4])
#	image6 = read_image(file_dir+'/'+filelist[5])
#	image7 = read_image(file_dir+'/'+filelist[6])
#	image8 = read_image(file_dir+'/'+filelist[7])
#	image9 = read_image(file_dir+'/'+filelist[8])
#	image10 = read_image(file_dir+'/'+filelist[9])
#	for index,file in enumerate(filelist):  
#		if index >=10 : 
#			image11 = read_image(file_dir+'/'+ file)
#			#train_image = tf.concat(2,[image1, image2, image3, image4, image5, image6, image7, image8, image9, image10, image11])
#			train_image = tf.concat([image1, image2, image3, image4, image5, image6, image7, image8, image9, image10, image11],2)
#			#train_image = np.concatenate((image1, image2, image3, image4, image5, image6, image7, image8, image9, image10, image11),axis=2)
#			#train_image = np.concatenate(( image11, image11),axis=2)
#			lable_image = tf.image.per_image_standardization(image6)
#			lable_image = tf.cast(lable_image,dtype = tf.float32)
#			train_image = tf.image.per_image_standardization(train_image)
#			train_image = tf.cast(train_image,dtype = tf.float32)
#			baixubao.append(train_image)   
#			label.append(lable_image) 
#			image1 = image2;
#			image2 = image3;
#			image3 = image4;
#			image4 = image5;
#			image5 = image6;
#			image6 = image7;
#			image7 = image8;
#			image8 = image9;
#			image9 = image10;
#			image10 = image11;
#	return baixubao, label
      
i=0
b=1
def get_files(file_dir):  
	filelist = os.listdir(file_dir)
	image1 = file_dir+'/'+filelist[0]	
	image2 = file_dir+'/'+filelist[1]
	image3 = file_dir+'/'+filelist[2]
	image4 = file_dir+'/'+filelist[3]
	image5 = file_dir+'/'+filelist[4]
	image6 = file_dir+'/'+filelist[5]
	image7 = file_dir+'/'+filelist[6]
	image8 = file_dir+'/'+filelist[7]
	image9 = file_dir+'/'+filelist[8]
	image10 = file_dir+'/'+filelist[9]
	image11 = file_dir+'/'+filelist[10]
	train_image = [image1, image2, image3, image4, image5, image6, image7, image8, image9, image10, image11]
	lable_image = image6
	baixubao.append(train_image)
	label.append(lable_image) 
	for index,file in enumerate(filelist): 
		if  index== 11+i :
			image12 = file_dir+'/'+ file
			if  index==12+i :	
				image13 =file_dir+'/'+file
				if  index==13+i :	
					image14 =file_dir+'/'+file
					if  index==14+i :	
						image15 =file_dir+'/'+file
						if  index==15+i :	
							image16 =file_dir+'/'+file
							if  index==16+i :	
								image17 =file_dir+'/'+file
								if  index==17+i :	
									image18 =file_dir+'/'+file
									if  index>=18+i :	
										image19 =file_dir+'/'+file
										if  index>=19+i :	
											image20 =file_dir+'/'+file
											if  index>=20+i :	
												image21 =file_dir+'/'+file
												if  index>=21+i :	
													image22 =file_dir+'/'+file

													if index % 10==b:
														b=b+1          
														image1 = image12;
														image2 = image13;
														image3 = image14;
														image4 = image15;
														image5 = image16;
														image6 = image17;
														image7 = image18;
														image8 = image19;
														image9 = image20;
														image10 = image21;
														image11 = image22;
														train_image = [image1, image2, image3, image4, image5, image6, image7, image8, image9, image10, image11]
														lable_image = image6
														baixubao.append(train_image)
														label.append(lable_image) 
	return baixubao, label     
#---------------------------------------------------------------------------  
#--------------------生成Batch----------------------------------------------  
  
#step1：将上面生成的List传入get_batch() ，转换类型，产生一个输入队列queue，因为img和lab  
#是分开的，所以使用tf.train.slice_input_producer()，然后用tf.read_file()从队列中读取图像  
#   image_W, image_H, ：设置好固定的图像高度和宽度  
#   设置batch_size：每个batch要放多少张图片  
#   capacity：一个队列最大多少  
def get_batch(image, label, batch_size, capacity):  
 
#step4：生成batch
#image_batch: 4D tensor [batch_size, width, height, 11],dtype=tf.float32   
#label_batch: 1D tensor [batch_size], dtype=tf.int32  
	
	input_queue = tf.train.slice_input_producer([image, label]) 
	input_image = input_queue[0]

	image_1 = read_image(input_image[0])
	image_2 = read_image(input_image[1])
	image_3 = read_image(input_image[2])
	image_4 = read_image(input_image[3])
	image_5 = read_image(input_image[4])
	image_6 = read_image(input_image[5])
	image_7 = read_image(input_image[6])
	image_8 = read_image(input_image[7])
	image_9 = read_image(input_image[8])
	image_10 = read_image(input_image[9])
	image_11 = read_image(input_image[10])
	train_image = tf.concat([image_1, image_2, image_3, image_4, image_5, image_6, image_7, image_8, image_9, image_10, image_11],2)

	train_image = tf.image.per_image_standardization(train_image)
	train_image = tf.cast(train_image,dtype = tf.float32)
	
	lable_image = tf.image.per_image_standardization(read_image(input_queue[1]))
	lable_image = tf.cast(lable_image,dtype = tf.float32)
	image_batch, label_batch = tf.train.batch([train_image, lable_image],  
                                                batch_size= batch_size,  
                                                num_threads= 1  ,   
                                                capacity = capacity)  
	return image_batch, label_batch              
  
#========================================================================</span>  
