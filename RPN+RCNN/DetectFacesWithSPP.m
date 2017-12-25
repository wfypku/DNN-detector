function [all_faces, all_points, all_weights, all_scales, all_sub_scales] = DetectFacesWithSPP(imagelist, rpn_model, pyramid_scale, thr, gpu_id)

param = rpn_model.param;

%% init DNN solver
root_folder = param.DNN_root_folder;
if (nargin <= 4)
    gpu_id = param.gpu_id(1);
end
assert(length(gpu_id) == 1);
current_dir = pwd;
cd(root_folder);
DNN.caffe_mex('release_solver');
DNN.caffe_mex('set_device_solver', gpu_id);
DNN.caffe_mex('init_solver', 'solver.prototxt', rpn_model.dnn_model, [root_folder, '\log\']);
% DNN.caffe_mex('set_mode_cpu');
cd(current_dir);

%% detect
all_faces = cell(length(imagelist), 1);
all_weights = cell(length(imagelist), 1);
all_points = cell(length(imagelist), 1);
all_scales = cell(length(imagelist), 1);
all_sub_scales = cell(length(imagelist), 1);
tic;
for i = 1:length(imagelist)
    img = ReadColorImage(imagelist{i});
    if (param.channel_num == 1)
        img = rgb2gray(img);
    end
    rects = [];
    weights = [];
    points = [];
    pyramid_scales = [];
    sub_scales = [];
    for resize_scale = pyramid_scale
        pad = ceil(rpn_model.anchor_center(1));
        img_in = imresize(img, resize_scale);
        img_in = PaddingImage(img_in, pad);
        if (min(size(img_in, 1), size(img_in, 2)) < rpn_model.receptive_field_size || max(size(img_in, 1), size(img_in, 2)) > param.max_img_size)
            continue;
        end
        height_out = rpn_model.dim_map(size(img_in, 1));
        width_out = rpn_model.dim_map(size(img_in, 2));
        dim = height_out * width_out;
        data = single(DNN.ConvertImageFormatForCaffe(img_in)) - single(127);
        test_batch{1}{1} = data;
        test_batch{1}{2} = zeros([1, 1, param.scale_num, dim], 'single');
        test_batch{1}{3} = zeros([1, 1, param.pos_points_num*2*param.scale_num, dim], 'single');
        test_batch{1}{4} = ones([width_out, height_out, 1, 1], 'single');
        test_batch{1}{5} = zeros([1, 1, param.scale_num, dim], 'single');
        test_batch{1}{6} = ones([width_out, height_out, 1, 1], 'single');
        test_batch{1}{7} = zeros([1, 1, param.pos_points_num*2*param.scale_num, dim], 'single');
        test_batch{1}{8} = reshape(single(rpn_model.anchor_points)', [1, 1, param.pos_points_num*2, param.scale_num]);
        test_batch{1}{9} = reshape(single(param.face_size_class), [1, 1, 1, param.scale_num]);
        test_batch{1}{10} = zeros([1, 1, 5, 1], 'single');
        test_batch{1}{11} = zeros([1, 1, 1, 1], 'single');
        DNN.caffe_mex('test', test_batch);
        p = DNN.caffe_mex('get_response_solver', 'predict_facial_points');
        p = reshape(double(p{1}), param.pos_points_num*2+1, [])';
        p = p(:, 2:end);
        r = DNN.caffe_mex('get_response_solver', 'predict_face_rects');
        r = reshape(double(r{1}), size(r{1}, 3), [])';
        sub_scales = cat(1, sub_scales, r(:, [7, 7]));
        r = r(:, [2 4 3 5]);
        w = DNN.caffe_mex('get_response_solver', 'resp_spp');
        w = double(w{1}(:));
        f = w >= thr;
        w = w(f, :);
        rects = cat(1, rects, (r(f, :)-pad)/resize_scale);
        points = cat(1, points, (p(f, :)-pad)/resize_scale);
        weights = cat(1, weights, [w, w]);
        pyramid_scales = cat(1, pyramid_scales, ones(size(r, 1), 2)*resize_scale);
        
    end
    all_faces{i} = rects;
    all_weights{i} = weights;
    all_points{i} = points;
    all_scales{i} = pyramid_scales;
    all_sub_scales{i} = sub_scales;
    if (mod(i, 100) == 0)
        fprintf('%d/%d... ', i, length(imagelist));
        toc;
    end
end

