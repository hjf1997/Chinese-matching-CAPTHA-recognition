for i = 0:9999
    I=imread(strcat( 'E:\文档\验证码识别\数据\data-2\train\',strcat(num2str(i,'%04d'), '.jpg')));
    I=ImageBinarization(I,-1,1,0);
    add = strcat('E:\文档\验证码识别\处理后的数据\data-2\ ', num2str(i,'%04d'));
    imwrite(I, strcat(add, '.jpg'))   
    %imshow(I)
end