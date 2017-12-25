function [all_ft, all_ft_index, all_faces, all_points, all_weights, all_scales, all_sub_scales] = ExtractRPNRCNNFeature(imagelist, rpn_model, pyramid_scale, thr, gpu_id)

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
cd(current_dir);

%% init rcnn param
sim_trans_anchor = [18.5910 16.8455
    45.4307 16.8036
    32.0362 33.6163
    20.2993 48.0735
    43.8179 48.0359];
rects_anchor = GetRectFromPointsMex(reshape(sim_trans_anchor', 1, []));
width_anchor = rects_anchor(2) - rects_anchor(1);
height_anchor = rects_anchor(4) - rects_anchor(3);

%% detect
all_faces = cell(length(imagelist), 1);
all_weights = cell(length(imagelist), 1);
all_points = cell(length(imagelist), 1);
all_scales = cell(length(imagelist), 1);
all_sub_scales = cell(length(imagelist), 1);
all_ft_index = cell(length(imagelist), 1);
all_ft = cell(length(imagelist), 1);
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
    index = [];
    ft = [];
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
        test_batch{1}{10} = zeros([1, 1, param.pos_points_num*2, 0], 'single');
        test_batch{1}{11} = ones([1, 1, 1, 1], 'single')*2;
        DNN.caffe_mex('test', test_batch);
        if true
            select_facial_points = DNN.caffe_mex('get_response_solver', 'select_facial_points');
            select_facial_points = double(reshape(select_facial_points{1}, param.pos_points_num*2, [])');
            select_face_rects = DNN.caffe_mex('get_response_solver', 'select_face_rects');
            select_face_rects = double(reshape(select_face_rects{1}, 4, [])');
            feature_index = DNN.caffe_mex('get_response_solver', 'feature_index');
            feature_index = double(reshape(feature_index{1}, 3, [])');
            rpn_rcnn_feature = DNN.caffe_mex('get_response_solver', 'rpn_rcnn_feature');
            rpn_rcnn_feature = reshape(rpn_rcnn_feature{1}, 256, [])';
            resp_class_rcnn = DNN.caffe_mex('get_response_solver', 'resp_class_rcnn');
            resp_class_rcnn = double(resp_class_rcnn{1}(:));
            resp_reg_rcnn = DNN.caffe_mex('get_response_solver', 'resp_reg_rcnn');
            resp_reg_rcnn = double(reshape(resp_reg_rcnn{1}, param.pos_points_num*2, [])');
            f = resp_class_rcnn >= thr;
            weights = cat(1, weights, resp_class_rcnn(f));
            p = [];
            select_facial_points = select_facial_points(f, :);
            resp_reg_rcnn = resp_reg_rcnn(f, :);
            feature_index = feature_index(f, :);
            rpn_rcnn_feature = rpn_rcnn_feature(f, :);
            for j = 1:size(resp_reg_rcnn)
                trans = fitgeotrans(reshape(select_facial_points(j, 1:10), 2, 5)', sim_trans_anchor, 'nonreflectivesimilarity');
                p_rcnn = reshape(resp_reg_rcnn(j, :), 2, [])';
                p_rcnn(:, 1) = p_rcnn(:, 1) * width_anchor;
                p_rcnn(:, 2) = p_rcnn(:, 2) * height_anchor;
                p_rcnn_ori = trans.transformPointsInverse(p_rcnn);
                p(j, :) = reshape(p_rcnn_ori', 1, []);
            end
            sub_scales = cat(1, sub_scales, ones(sum(f), 1));
            rects = cat(1, rects, (GetRectFromPointsMex(p)-pad)/resize_scale);
            points = cat(1, points, (p-pad)/resize_scale);
            pyramid_scales = cat(1, pyramid_scales, ones(sum(f), 2)*resize_scale);
            index = cat(1, index, feature_index);
            ft = cat(1, ft, rpn_rcnn_feature);
        end
    end
    all_faces{i} = rects;
    all_weights{i} = weights;
    all_points{i} = points;
    all_scales{i} = pyramid_scales;
    all_sub_scales{i} = sub_scales;
    all_ft_index{i} = index;
    all_ft{i} = ft;
    if (mod(i, 100) == 0)
        fprintf('%d/%d... ', i, length(imagelist));
        toc;
    end
end