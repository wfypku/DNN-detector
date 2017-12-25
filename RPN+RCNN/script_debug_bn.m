clear all;
close all;
clc;


root_folder = 'D:\DongChen\Matlab\DNNFaceDetectionOneStep\RPN+RCNN_output\GoogleNetHalf_GoogleNetHalf_s1_(clear_5p_neg_clear)_(36_72_3_640)_(8_10_50_5)';
load([root_folder, '\rpn_model.mat']);
load('train_batch.mat');
load('weights_init.mat');

param = rpn_model.param;
root_folder = param.DNN_root_folder;
gpu_id = param.gpu_id;
recovery_model = param.recovery_model;
init_model = param.init_model;
current_dir = pwd;
cd(root_folder);
DNNDebug.caffe_mex('release_solver');
DNNDebug.caffe_mex('set_device_solver', 0:7);
DNNDebug.caffe_mex('init_solver', 'solver.prototxt', '', [root_folder, '\log\']);
% DNNDebug.caffe_mex('set_mode_cpu');
cd(current_dir);

DNNDebug.caffe_mex('set_weights_solver', weights_init);
loss = DNNDebug.caffe_mex('train', train_batch);
weights = DNNDebug.caffe_mex('get_weights_solver');
for i = 1:length(weights) 
    for j = 1:length(weights(i).weights) 
        if (sum(sum(isnan(weights(i).weights{j}))) ~= 0) 
            fprintf('%d %d\n', i, j); 
        end
    end
end

%% get
conv1_t = DNNDebug.caffe_mex('get_response_solver', 'conv1_t');
v = [];
for i = 1:length(conv1_t)
    a = permute(conv1_t{i}, [3, 1, 2, 4]);
    a = reshape(a, size(a, 1), [])';
    v = [v; a];
end
mv = mean(v, 1);
mv2 = mean(v.^2, 1);
mv_bn = weights(2).weights{3}(:)';
mv2_bn = weights(2).weights{4}(:)';
plot(mv, 'r-');
hold on; 
plot(mv_bn, 'g-');

