function [dim_map, correct_receptive_field, stride] = GetDNNOutputDimMap(param)

root_folder = param.DNN_root_folder;
current_dir = pwd;
cd(root_folder);
if (DNN.caffe_mex('is_initialized') == 1)
    DNN.caffe_mex('release');
end
DNN.caffe_mex('init', 'net_noloss.prototxt', '', 0, [root_folder, '\log\']);
DNN.caffe_mex('set_mode_gpu');
DNN.caffe_mex('set_device', 0);
cd(current_dir);

dim_map = zeros(param.max_img_size, 1);
for dim = param.max_img_size:-1:1
    DNN.caffe_mex('set_input_size', [dim, dim, param.channel_num, 1]);
    feature = DNN.caffe_mex('get_response', 'feature');
    assert(size(feature, 1) == size(feature, 2));
    dim_map(dim) = size(feature, 1);
    if (size(feature, 1) == 1)
        break;
    end
end

correct_receptive_field = find(dim_map == 1, 1, 'last');
stride = sum(dim_map == 2);

DNN.caffe_mex('release');
