for i = 0:9999
    add = strcat('处理后的数据\data-1\', num2str(i,'%04d'));
    D=dir(add);
    num = size(D,1) - 3;
    count = 0;
    for j = 1:8
        if (isnan(mappings2(i+1,j)) == false)
            count = count + 1;
        end
    end
    if num ~= count 
        num2str(i,'%04d')
    end
end