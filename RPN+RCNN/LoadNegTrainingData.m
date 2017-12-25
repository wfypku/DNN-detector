function neg_data = LoadNegTrainingData(imagelist_neg, rpn_model)

n = length(imagelist_neg);
max_img_size = rpn_model.param.max_img_size;
fprintf('Generate Neg Training Data. Memory cost: < %.2fGB\r\n', ...
    n*max_img_size*max_img_size*rpn_model.param.channel_num/1024/1024/1024);

neg_data = cell(n, 1);
tic;
for i = 1:n
    img = ReadColorImage(imagelist_neg{i});
    if (rpn_model.param.channel_num == 1)
        img = rgb2gray(img);
    end
    scale = min(max_img_size/size(img, 1), max_img_size/size(img, 2));
    neg_data{i} = imresize(img, scale);
    if (mod(i, 100) == 0)
        fprintf('load %d/%d... ', i, length(imagelist_neg));
        toc;
    end
end
