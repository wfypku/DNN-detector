clear;
close all;
fclose all;
clc;

%% set training data
% param.pos_data = 'align_5p';
% param.pos_image_folder = 'D:\DongChen\Matlab\DNNFaceDetection\RawTrainingData\Positive';
% param.pos_image_list = 'D:\DongChen\Matlab\DNNFaceDetection\RawTrainingData\Positive\imagelist_100new_mpie_hardface_pubfig_lfw_aflw.txt';
% param.pos_rects_file = 'D:\DongChen\Matlab\DNNFaceDetection\RawTrainingData\Positive\rects_100new_mpie_hardface_pubfig_lfw_aflw.txt';
% param.pos_points_file = 'D:\DongChen\Matlab\DNNFaceDetection\RawTrainingData\Positive\points_100new_mpie_hardface_pubfig_lfw_aflw_5p.txt';
% param.pos_points_num = 5;
% param.pos_data = 'aflw_21p';
% param.pos_image_folder = 'D:\DongChen\FaceData\AFLW\Images';
% param.pos_image_list = 'D:\DongChen\FaceData\AFLW\imagelist_split.txt';
% param.pos_rects_file = 'D:\DongChen\FaceData\AFLW\face_rects_split.txt';
% param.pos_points_file = 'D:\DongChen\FaceData\AFLW\points_5p_split.txt';
% param.pos_points_num = 5;
% param.pos_data = 'all_5p';
% param.pos_image_folder = 'D:\DongChen\FaceData';
% param.pos_image_list = 'D:\DongChen\FaceData\All_LabelFace_5p\imagelist_all2.txt';
% param.pos_rects_file = 'D:\DongChen\FaceData\All_LabelFace_5p\rects_all.txt';
% param.pos_points_file = 'D:\DongChen\FaceData\All_LabelFace_5p\points_all.txt';
% param.pos_points_num = 5;
param.pos_data = 'clear_5p';
param.pos_image_folder = 'D:\fawe\DNNFaceDetection\Data\Positive';
param.pos_image_list = 'D:\fawe\DNNFaceDetection\Data\All_LabelFace_5p\imagelist_clear.txt';
param.pos_rects_file = 'D:\fawe\DNNFaceDetection\Data\All_LabelFace_5p\rects_clear.txt';
param.pos_points_file = 'D:\fawe\DNNFaceDetection\Data\All_LabelFace_5p\points_clear.txt';
param.pos_points_num = 5;

% param.neg_data = 'neg_all';
% param.neg_image_folder = 'D:\DongChen\FaceData';
% param.neg_image_list = 'D:\DongChen\FaceData\All_nonface\imagelist_all.txt';
param.neg_data = 'neg_clear';
param.neg_image_folder = 'D:\fawe\DNNFaceDetection\Data\Negative';
param.neg_image_list = 'D:\fawe\DNNFaceDetection\Data\All_nonface\imagelist_clear_20170208.txt';

%% set param
% input data setting
param.scale_num = 1;
param.min_face_size = 36;
param.max_face_size = 72;
param.channel_num = 3;
param.max_img_size = 1000;
param.max_rand_offset = 8;
param.pos_aug_num = 10;
param.pos_overlap_ratio = 0.5;
param.gray_augment_ratio = 0.5;
param.net_name = 'GoogleNetHalf_GoogleNetHalf_fintune_w0_lr10_pr128_nr128_t-3';
param.folder_name = sprintf('%s_s%d_(%s_%s)_(%d_%d_%d_%d)_(%d_%d_%d_%d)', ...
    param.net_name, param.scale_num, param.pos_data, param.neg_data, ...
    param.min_face_size, param.max_face_size, param.channel_num, param.max_img_size,...
    param.max_rand_offset, param.pos_aug_num, round(param.pos_overlap_ratio*100), round(param.gray_augment_ratio*10));

% dnn training setting
param.DNN_root_folder = ['D:\fawe\DNNFaceDetection\Output\', param.folder_name];
param.gpu_id = 0:7;
param.validation_rate = 0.1;
param.batch_size_per_gpu = 1;
param.pos_gpu_num = 6;
param.neg_gpu_num = 2;
param.anchor_is_field_center = false; % decide whether the anchor point is the center of receptive field or stride
param.preload_training_data = false;

% recovery or fine-tune
param.recovery_model = '';
param.init_model = 'model_init_rpn_rcnn';

% info output setting
param.test_interval = 5000;
param.test_iter = 500;
param.display_interval = 100;

%% calc face size of each scale
max_log_scale = log2(param.max_face_size/param.min_face_size);
log_scale_class = 0:max_log_scale/(2*param.scale_num):max_log_scale;
log_scale_class = log_scale_class(2:2:end);
assert(length(log_scale_class) == param.scale_num);
param.log_scale_class = log_scale_class;
param.face_size_class = param.min_face_size .* 2.^log_scale_class;
param.points_scale_value = reshape(repmat(param.face_size_class, param.pos_points_num*2, 1), [], 1);
if (~param.preload_training_data)
    assert(param.batch_size_per_gpu == 1);
end
assert(param.channel_num == 1 || param.channel_num == 3);

%% init output workspace
if (exist(param.DNN_root_folder, 'dir'))
    warning([param.DNN_root_folder, ' already exist!']);
end
CreateDirectory(param.DNN_root_folder);
a = param.scale_num;
net_folder = sprintf('%s\\..\\RPN+RCNN_net\\%s_s%d', pwd, param.net_name, param.scale_num);
copyfile([net_folder, '\\*'], param.DNN_root_folder);

%% begin training
disp('Training start...')
script_train_rpn_model;
