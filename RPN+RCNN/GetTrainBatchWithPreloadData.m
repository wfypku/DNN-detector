function train_batch = GetTrainBatchWithPreloadData(select_index, pos_data, neg_data, rpn_model)

param = rpn_model.param;
assert(param.pos_gpu_num + param.neg_gpu_num == length(param.gpu_id));
pos_gpu_num = param.pos_gpu_num;
neg_gpu_num = param.neg_gpu_num;
batch_size_per_gpu = param.batch_size_per_gpu;
scale_num = param.scale_num;

%% load pos data
train_batch = cell(length(param.gpu_id), 1);
sa = select_index.sp;
for i = 1:pos_gpu_num
    s = sa((i-1)*batch_size_per_gpu+1:i*batch_size_per_gpu);
    train_batch{i}{1} = single(pos_data.data(:, :, :, s))-single(127);
    train_batch{i}{2} = pos_data.label(:, :, :, s);
    train_batch{i}{3} = pos_data.delta_points(:, :, :, s);
    train_batch{i}{4} = ones([1, 1, 1, batch_size_per_gpu], 'single');
    train_batch{i}{5} = pos_data.label(:, :, :, s);
    train_batch{i}{6} = ones([1, 1, 1, batch_size_per_gpu], 'single');
    train_batch{i}{7} = pos_data.mask_points(:, :, :, s);
end

%% load neg data
sa = select_index.sn;
for i = 1:neg_gpu_num
    id = sa(i);
	img = neg_data{id};
    [img_out, dim_out] = RandomResizeNegImage(img, rpn_model);
    dim = dim_out(1)*dim_out(2);
    data = single(DNN.ConvertImageFormatForCaffe(img_out)) - single(127);
    train_batch{i+pos_gpu_num}{1} = data;
    train_batch{i+pos_gpu_num}{2} = zeros([1, 1, scale_num, dim], 'single');
    train_batch{i+pos_gpu_num}{3} = zeros([1, 1, param.pos_points_num*2*param.scale_num, 0], 'single');
    train_batch{i+pos_gpu_num}{4} = ones([dim_out(2), dim_out(1), 1, 1], 'single');
    train_batch{i+pos_gpu_num}{5} = ones([1, 1, scale_num, dim], 'single');
    train_batch{i+pos_gpu_num}{6} = zeros([dim_out(2), dim_out(1), 1, 1], 'single');
    train_batch{i+pos_gpu_num}{7} = zeros([1, 1, param.pos_points_num*2*param.scale_num, 0], 'single');
end
