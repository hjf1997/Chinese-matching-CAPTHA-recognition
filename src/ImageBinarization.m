function I3 = ImageBinarization(I, t, m,cleaning)
    %I = imread(strcat(strcat('数据\data-1\train\',int2str(i)), '.jpg'));
    %imshow(I);
    thresh = graythresh(I);
    I2 = im2bw(I, thresh);
    %imshow(I2)
    I3= medfilt2(I2,[m,m]);
    [w,h] = size(I3);
    black_point = 0;
    for x = 3:w-2
        for y = 3:h-2
            mid_pixel = I3(x,y);
            if mid_pixel == 0
                if (I3(x-2,y)==0)
                    black_point = black_point + 1;
                end
                if (I3(x+2,y)==0)
                    black_point = black_point + 1;
                end
                if (I3(x,y+2)==0)
                    black_point = black_point + 1;
                end
                if (I3(x,y-2)==0)
                    black_point = black_point + 1;
                end
                if black_point <= t
                    I3(x,y) = 1;
                end
                black_point = 0;
            end
        end
    %imwrite(I3, strcat(strcat('数据\降噪\',int2str(i)), '.jpg'));
    end
    if cleaning == 1
        num = sum(1-I3,1);
        for i = 1:h
            if num(i)<=3
            I3(:,i) = 1;
            end
        end
    end
end
