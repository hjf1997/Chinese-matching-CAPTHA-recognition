for i = 0:9999
    I=imread(strcat( 'E:\�ĵ�\��֤��ʶ��\����\data-4\train\',strcat(num2str(i,'%04d'), '.jpg')));
    I=ImageBinarization(I,-1,1,0);
    add = strcat('E:\�ĵ�\��֤��ʶ��\�����������\data-4_raw\ ', num2str(i,'%04d'));
    imwrite(I, strcat(add, '.jpg'))   
    %imshow(I)
end