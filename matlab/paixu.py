# -*- coding: utf-8 -*-
"""
Created on Thu May 23 21:41:09 2019

@author: WYC
"""
'''
import os, sys
def MkDir():
    path = 'C:/Users/WYC/Desktop/'
    i = 0
    for i in range(10): #创建文件个数
        file_name = path + str(i)
        os.mkdir(file_name)
        i=i+1
MkDir()

import os, sys
path1 = 'C:/Users/WYC/Desktop/xieshanzao'  #指定名称文件夹所在路径
path2 = 'C:/Users/WYC/Desktop/'    #新建文件夹所在路径
 
def MkDir():
    dirs = os.listdir(path1)
 
    for dir in dirs:
        file_name = path2 + str(dir)
        os.mkdir(file_name)
 
MkDir()
'''
#单个文件夹的图片重命名
import os
 
class BatchRename():
    '''
    #批量重命名文件夹中的图片文件

    '''
    def __init__(self):
         self.path = '/home/lishuaijun1/matlab/xsz/'

    def rename(self):
        filelist = os.listdir(self.path)  # 获取文件路径
        total_num = len(filelist)  # 获取文件长度（个数）
        i = 0  # 表示文件的命名是从1开始的
        for item in filelist:
            if item.endswith('.jpg'):  # 初始的图片的格式为jpg格式的（或者源文件是png格式及其他格式，后面的转换格式就可以调整为自己需要的格式即可）
                src = os.path.join(os.path.abspath(self.path), item)
                dst = os.path.join(os.path.abspath(self.path), str(i)+ '.jpg')  # 处理后的格式也为jpg格式的，当然这里可以改成png格式
                # dst = os.path.join(os.path.abspath(self.path), '0000' + format(str(i), '0>3s') + '.jpg')    这种情况下的命名格式为0000000.jpg形式，可以自主定义想要的格式
                try:
                    os.rename(src, dst)
                    print('converting %s to %s ...' % (src, dst))
                    i = i + 1
                except:
                    continue
        print('total %d to rename & converted %d jpgs' % (total_num, i))

                   
if __name__ == '__main__':
    demo = BatchRename()
    demo.rename()

