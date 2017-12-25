function train_batch = GetTrainBatch(select_index, imagelist_pos, rects_pos, points_pos, imagelist_neg, rpn_model)

param = rpn_model.param;
assert(param.pos_gpu_num + param.neg_gpu_num == length(param.gpu_id));
pos_gpu_num = param.pos_gpu_num;
neg_gpu_num = param.neg_gpu_num;
scale_num = param.scale_num;

%% load pos data
train_batch = cell(length(param.gpu_id), 1);
sa = select_index.sp;
for i = 1:pos_gpu_num
    id = sa(i);
    img = ReadColorImage(imagelist_pos{id});
    if (param.gray_augment_ratio > rand())
        img = repmat(rgb2gray(img), [1, 1, 3]);
    end
    if (param.channel_num == 1)
        img = rgb2gray(img);
    end
    [img_out, rects_out, points_out] = RandomResizePosImageMex(img, rects_pos(id, :), points_pos(id, :), rpn_model);
    [mask_pos, mask_points, label, delta] = GetMaskPosForTraining(img_out, rects_out, points_out, rpn_model);
    valid_points = reshape(single(repmat(points_pos(id, :) ~= -1, 1, scale_num)), [1, 1, scale_num*param.pos_points_num*2, 1]);
    mask_points = bsxfun(@times, mask_points, valid_points);
    data = single(DNN.ConvertImageFormatForCaffe(img_out)) - single(127);
    train_batch{i}{1} = data;
    train_batch{i}{2} = label;
    train_batch{i}{3} = delta;
    train_batch{i}{4} = mask_pos;
    train_batch{i}{5} = label;
    train_batch{i}{6} = mask_pos;
    train_batch{i}{7} = mask_points;
    train_batch{i}{8} = reshape(single(rpn_model.anchor_points)', [1, 1, param.pos_points_num*2, param.scale_num]);
    train_batch{i}{9} = reshape(single(param.face_size_class), [1, 1, 1, param.scale_num]);
    train_batch{i}{10} = reshape(single(points_out), [1 1 param.pos_points_num*2 1]);
    train_batch{i}{11} = ones([1, 1, 1, 1], 'single');
end

%% load neg data
sa = select_index.sn;
for i = 1:neg_gpu_num
    id = sa(i);
	img = ReadColorImage(imagelist_neg{id});
    if (param.gray_augment_ratio > rand())
        img = repmat(rgb2gray(img), [1, 1, 3]);
    end
    if (param.channel_num == 1)
        img = rgb2gray(img);
    end
    [img_out, dim_out] = RandomResizeNegImage(img, rpn_model);
    dim = dim_out(1)*dim_out(2);
    data = single(DNN.ConvertImageFormatForCaffe(img_out)) - single(127);
    train_batch{i+pos_gpu_num}{1} = data;
    train_batch{i+pos_gpu_num}{2} = zeros([1, 1, scale_num, dim], 'single');
    train_batch{i+pos_gpu_num}{3} = zeros([1, 1, param.pos_points_num*2*scale_num, 0], 'single');
    train_batch{i+pos_gpu_num}{4} = ones([dim_out(2), dim_out(1), 1, 1], 'single');
    train_batch{i+pos_gpu_num}{5} = ones([1, 1, scale_num, dim], 'single');
    train_batch{i+pos_gpu_num}{6} = zeros([dim_out(2), dim_out(1), 1, 1], 'single');
    train_batch{i+pos_gpu_num}{7} = zeros([1, 1, param.pos_points_num*2*scale_num, 0], 'single');
    train_batch{i+pos_gpu_num}{8} = reshape(single(rpn_model.anchor_points)', [1, 1, param.pos_points_num*2, param.scale_num]);
    train_batch{i+pos_gpu_num}{9} = reshape(single(param.face_size_class), [1, 1, 1, param.scale_num]);
    train_batch{i+pos_gpu_num}{10} = zeros([1, 1, param.pos_points_num*2, 0], 'single');
    train_batch{i+pos_gpu_num}{11} = zeros([1, 1, 1, 1], 'single');
end


