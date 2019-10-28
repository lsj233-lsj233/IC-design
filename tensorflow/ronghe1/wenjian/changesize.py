import os
import tensorflow as tf
from PIL import Image
import numpy as np

#image_dir = '/data0/tiandian/tensorflow/ronghe/wenjian/duozheng11bb/data/trainlabel' 
image_dir = '/data0/tiandian/tensorflow/ronghe/wenjian/duozheng11bb/data/test2' 
#out_dir = '/data0/tiandian/tensorflow/ronghe/wenjian/shujuji/'
out_dir = '/data0/tiandian/tensorflow/ronghe/wenjian/testimg2/'
size_h = 48
size_w = 48
for file in os.listdir(image_dir): 
	filename,type = os.path.splitext(file)
	if type == '.jpg':
		img=Image.open(image_dir + '/' + file)
		out = img.resize((size_w,size_h))
		out.save(out_dir + file)
