function [image,boo] = shortcout(I)
    boo = 1;
    j = 1;
    image = {};
    X_sum = xfenge(I);
    count = 0;
    while (j<=size(I,2))
        temp = [];
        while (j<=size(I,2) && X_sum(j)>2)
            temp = [temp, I(:,j)];
            j = j + 1;
        end
        if (size(temp,2)>=15)
            count = count + 1;
            if count >=5
                boo = 0;
            end
            figure(count+2);
            image{count} = temp;
            imshow(temp);
                %count = count + 1;
                %imwrite(temp,strcat(add, strcat('\', strcat(num2str(count), '.jpg'))));
            %imshow(temp);
        end
        j = j + 1;
    end
    if count<4
        boo = 0;
    end
end