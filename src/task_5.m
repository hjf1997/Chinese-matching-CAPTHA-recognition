for i = 1011:9999
    I=imread(strcat( 'E:\文档\验证码识别\数据\data-5\train\',strcat(num2str(i,'%04d'),strcat('\',...
    strcat(num2str(i,'%04d'),'.jpg')))));
    I=ImageBinarization(I,-1,2,0);
    add = strcat('E:\文档\验证码识别\处理后的数据\data-4\ ', num2str(i,'%04d'));
    mkdir(add);
    imwrite(I(:,1:37),strcat(add, strcat('\', strcat(num2str(1), '.jpg'))))
    imwrite(I(:,38:74),strcat(add, strcat('\', strcat(num2str(2), '.jpg'))))
    imwrite(I(:,75:111),strcat(add, strcat('\', strcat(num2str(3), '.jpg'))))
    imwrite(I(:,112:150),strcat(add, strcat('\', strcat(num2str(4), '.jpg'))))
    %figure(1);
    %imshow(I);
    %figure(10);
    %imshow(I(:,1:37))
    %figure(11);
    %imshow(I(:,38:74))
    %figure(12);
    %imshow(I(:,75:111))
    %figure(13);
    %imshow(I(:,112:150))    
    
end