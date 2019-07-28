count = 0;
for i=0:9999
    add_a_basic = strcat( 'E:\文档\验证码识别\处理后的数据\data-5\',strcat(num2str(i,'%04d'),'\'));
    add_pn_basic = strcat('E:\文档\验证码识别\处理后的数据\data-5_match\ ',strcat(num2str(i,'%04d'),'\'));
    add_dis_basic = 'E:\文档\验证码识别\处理后的数据\data-5_triple\';
    %加载mappings先
    for j = 1:4
        label = mappings(i+1,j);
        add_pos = strcat(strcat(add_pn_basic, num2str(label)),'.jpg');
        I_pos = imread(add_pos);
        add_anchor = strcat(strcat(add_a_basic, num2str(j)),'.jpg');
        I_anc = imread(add_anchor);
        for k= 0:8
            if k==label
                continue;
            end
            add_neg = strcat(strcat(add_pn_basic, num2str(k)),'.jpg');
            I_neg = imread(add_neg);
            add_dis = strcat(add_dis_basic, num2str(count));
            mkdir(add_dis);
            add_dis = strcat(add_dis, '\');
            imwrite(I_anc, strcat(add_dis, strcat(num2str(1), '.jpg')));
            imwrite(I_pos, strcat(add_dis, strcat(num2str(2), '.jpg')));
            imwrite(I_neg, strcat(add_dis, strcat(num2str(3), '.jpg')));
            count = count + 1;
        end
    end     
end
