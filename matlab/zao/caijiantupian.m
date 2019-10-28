%[filename, pathname] = uigetfile({'*.jpg'; '*.bmp'; '*.gif'; '*.png' }, 'xuanqu');



[filename,pathname] = uigetfile('/home/lishuaijun1/matlab/xsz/*.jpg',...
                        'Select an Image File');  %此处文件类型可自行定义
if filename == 0
    return;
end
str = [pathname,filename];
I=imread(str);
imshow(I);
[x,y] = ginput(1)
A=imcrop(I,[x,y,300,300]);

%[A,rect] = imcrop(I);
imshow(A);
a=107;  
imwrite(A, ['/home/lishuaijun1/matlab/1' '/' int2str(a) '.jpg']); 
   