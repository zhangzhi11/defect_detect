
data_dir = 'D:\zz\defect_detect\data\7953';
out_dir = 'D:\zz\defect_detect\output\7953';
% if ~exist(out_dir)
%     
% end
data_list = dir(data_dir);
for index = 3:size(data_list,1)
    subdata_dir = fullfile(data_dir,data_list(index).name);
    if ~exist(fullfile(out_dir,data_list(index).name))
        mkdir(fullfile(out_dir,data_list(index).name));
    end
    sub_data_list = dir(subdata_dir);
    for sub_index = 3:size(sub_data_list,1)
        name = sub_data_list(sub_index).name(1:end-4);
        fprintf(strcat(name,'\n'));
        im = imread(fullfile(subdata_dir,sub_data_list(sub_index).name));
        % im=rgb2gray(im);
        ims = 255-im;

        bw=im2bw(ims,0.4);
%         figure;
%         imshow(bw);
        [L, num] = bwlabel(bw,8);
        STATS = regionprops(L,'basic');
        % for i = 1:num
        %     
        % end
        stats=regionprops(L);
        a=zeros(size(stats,1),1);
        for i = 1:size(stats,1)
            a(i)=stats(i,1).Area;
        end
        [sorted, p] = sort(a,'descend');

        max_indexs = p(1:4);
        boundingBoxs=zeros(4,4);

        for i = 1:4
            boundingBoxs(i,:)=stats(max_indexs(i)).BoundingBox; 
            a = ceil(boundingBoxs(i,1));
            b = floor(boundingBoxs(i,1)+boundingBoxs(i,3));
            c = ceil(boundingBoxs(i,2));
            d = floor(boundingBoxs(i,2)+boundingBoxs(i,4));
            sub_im = im(c:d,a:b);
            outdata_name = strcat(name,'_',num2str(i),'.jpg');
            fulldir = fullfile(out_dir,data_list(index).name,outdata_name);
            imwrite(sub_im,fulldir);
        end         
    end
end
% im = imread('E:\program\defect_detect\7953\bad\7953-bad-(1).jpg');
% % im=rgb2gray(im);
% ims = 255-im;
% 
% bw=im2bw(ims,0.4);
% figure;
% imshow(bw);
% [L, num] = bwlabel(bw,8);
% STATS = regionprops(L,'basic')
% % for i = 1:num
% %     
% % end
% stats=regionprops(L);
% a=zeros(size(stats,1),1);
% for i = 1:size(stats,1)
%     a(i)=stats(i,1).Area;
% end
% [sorted, index] = sort(a,'descend');
% 
% max_indexs = index(1:4);
% boundingBoxs=zeros(4,4);
% 
% for i = 1:4
%     boundingBoxs(i,:)=stats(max_indexs(i)).BoundingBox; 
%     a = ceil(boundingBoxs(i,1));
%     b = floor(boundingBoxs(i,1)+boundingBoxs(i,3));
%     c = ceil(boundingBoxs(i,2));
%     d = floor(boundingBoxs(i,2)+boundingBoxs(i,4));
%     sub_im = im(c:d,a:b);
% end 
% 
% 
% % RGB = insertShape(im,'Rectangle',boundingBoxs, 'LineWidth', 10);
% % figure,imshow(RGB);
% 
% 
% 
