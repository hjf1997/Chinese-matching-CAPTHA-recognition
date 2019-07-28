for i = 0:9999
    add = strcat( 'E:\文档\验证码识别\数据\data-5\train\',strcat(num2str(i,'%04d'),'\'));
    add1 = strcat('E:\文档\验证码识别\处理后的数据\data-5_match\ ', num2str(i,'%04d'));
    mkdir(add1);
    for j = 0:8
        I=imread(strcat(add, strcat(num2str(j),'.jpg')));
        I=ImageBinarization(I,-1,2,0);
        imwrite(I, strcat(add1, strcat('\', strcat(num2str(j), '.jpg'))))
    end
end