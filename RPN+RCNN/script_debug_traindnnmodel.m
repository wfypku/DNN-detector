clear all;
close all;
clc;

load('debug_spp');
train_batch = train_batch(3);
param = rpn_model.param;
root_folder = param.DNN_root_folder;
gpu_id = param.gpu_id;
recovery_model = param.recovery_model;
init_model = param.init_model;
current_dir = pwd;
cd(root_folder);
DNN.caffe_mex('release_solver');
DNN.caffe_mex('set_device_solver', 0);
if (~isempty(recovery_model) && ~strcmp(recovery_model, ''))
    DNN.caffe_mex('recovery_solver', 'solver.prototxt', recovery_model, [root_folder, '\log\']);
elseif (~isempty(init_model) && ~strcmp(init_model, ''))
    DNN.caffe_mex('init_solver', 'solver.prototxt', init_model, [root_folder, '\log\']);
else
    DNN.caffe_mex('init_solver', 'solver.prototxt', '', [root_folder, '\log\']);
end
DNN.caffe_mex('set_mode_cpu');
cd(current_dir);

cd(root_folder);
loss = DNN.caffe_mex('train', train_batch);
cd(current_dir);

for i = 1:length(train_batch)
    img = uint8(DNN.ConvertImageFormatForMatlab(train_batch{i}{1}) + 127);
    imshow(img);
    hold on;
    r_all = DNN.caffe_mex('get_response_solver', 'predict_face_rects');
    r_all = reshape(r_all{i}, size(r_all{i}, 3), [])';
%     DrawRects(r_all(:, [2 4 3 5]), 'b', 3);
    r_spp = DNN.caffe_mex('get_response_solver', 'rects_for_spp');
    r_spp = reshape(r_spp{i}, size(r_spp{i}, 3), [])';
    %     DrawRects(r_spp(:, [2, 4, 3, 5]), 'y', 1);
    label_spp = DNN.caffe_mex('get_response_solver', 'final_label');
    label_spp = double(label_spp{i}(:));
    DrawRects(r_spp(label_spp == 1, [2, 4, 3, 5]), 'y', 1);
    DrawRects(r_spp(label_spp == 0, [2, 4, 3, 5]), 'r', 1);
    r_label = DNN.caffe_mex('get_response_solver', 'label_face_rects');
    r_label = reshape(r_label{i}, size(r_label{i}, 3), [])';
    DrawRects(r_label(:, [2, 4, 3, 5]), 'g', 2);
    hold off;
    
    waitforbuttonpress;
end

resp_class = DNN.caffe_mex('get_response_solver', 'resp_class');
resp_class = DNN.ConvertMatForMatlab(resp_class{1});
imshow(TransformValue(resp_class, 0, 1));

spp = DNN.caffe_mex('get_response_solver', 'spp');
spp = reshape(spp{1}, [], size(spp{1}, 4))';
