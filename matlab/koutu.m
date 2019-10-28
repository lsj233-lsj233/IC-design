% [filename, pathname] = uigetfile({'*.jpg'; '*.bmp'; '*.gif'; '*.png' }, '选择图片');
% %没有图像
% if filename == 0
%     return;
% end
% str = [pathname,filename];
str='/home/lishuaijun1/matlab/xsz/';
%b=133;
for b=b
    b=b+1;
end
I=imread([str, '/' int2str(b) '.jpg']);

%I=imread(str);
[A,rect] = imcrop(I);
imshow(A);
%a=200;
for a=a
    a=a+11;
end
imwrite(A, ['/home/lishuaijun1/matlab/1' '/' int2str(a) '.jpg']); 


%I=imread(str);
[B,rect] = imcrop(I);
imshow(B);

imwrite(B, ['/home/lishuaijun1/matlab/1' '/' int2str(a+1) '.jpg']); 


%I=imread(str);
[C,rect] = imcrop(I);
imshow(C);

imwrite(C, ['/home/lishuaijun1/matlab/1' '/' int2str(a+2) '.jpg']); 


[D,rect] = imcrop(I);
imshow(D);

imwrite(D, ['/home/lishuaijun1/matlab/1' '/' int2str(a+3) '.jpg']); 


%I=imread(str);
[E,rect] = imcrop(I);
imshow(E);

imwrite(E, ['/home/lishuaijun1/matlab/1' '/' int2str(a+4) '.jpg']); 

[F,rect] = imcrop(I);
imshow(F);

imwrite(F, ['/home/lishuaijun1/matlab/1' '/' int2str(a+5) '.jpg']); 


%I=imread(str);
[G,rect] = imcrop(I);
imshow(G);

imwrite(G, ['/home/lishuaijun1/matlab/1' '/' int2str(a+6) '.jpg']); 

[H,rect] = imcrop(I);
imshow(H);

imwrite(H, ['/home/lishuaijun1/matlab/1' '/' int2str(a+7) '.jpg']); 


%I=imread(str);
[J,rect] = imcrop(I);
imshow(J);

imwrite(J, ['/home/lishuaijun1/matlab/1' '/' int2str(a+8) '.jpg']); 


[L,rect] = imcrop(I);
imshow(L);

imwrite(L, ['/home/lishuaijun1/matlab/1' '/' int2str(a+9) '.jpg']); 

[M,rect] = imcrop(I);
imshow(M);

imwrite(M, ['/home/lishuaijun1/matlab/1' '/' int2str(a+10) '.jpg']); 


[N,rect] = imcrop(I);
imshow(N);

imwrite(N, ['/home/lishuaijun1/matlab/1' '/' int2str(a+11) '.jpg']);