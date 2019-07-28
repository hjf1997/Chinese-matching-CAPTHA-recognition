for i = 0:9999
    I=imread(strcat(num2str(i,'%04d'),'.jpg'));
    I=ImageBinarization(I,1,3,1);
    add = '处理后的数据\data-3 ';
    %imshow(I);
    imwrite(I,strcat(add, strcat('\', strcat(num2str(i,'%04d'), '.jpg'))));
end