for i = 9494
    I=imread(strcat(num2str(i,'%04d'),'.jpg'));
    add = strcat('处理后的数据\data-1\ ', num2str(i,'%04d'));
    mkdir(add);
    figure(1)
    imshow(I);    
    figure(2);
    I=ImageBinarization(I,1,3,1);
    imshow(I);
    X_sum = xfenge(I);
    figure(3);
    plot(X_sum);
    [w,h] = size(I);
    j = 1;
    count = 0;
    while (j<h)
        temp = [];
        while (X_sum(j)>4 && j<h)
            j = j + 1;
            temp = [temp, I(:,j)];
        end
        if (size(temp,2)>=20)
                count = count + 1;
                imwrite(temp,strcat(add, strcat('\', strcat(num2str(count), '.jpg'))));
            %imshow(temp);
        end
        j=j+1;
    end
   %pause(1.3);
end