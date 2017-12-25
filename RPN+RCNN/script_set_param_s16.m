%% set training data
param.pos_image_folder = 'D:\DongChen';
param.pos_image_list = 'D:\DongChen\FaceData\All_LabelFace_5p\imagelist_all.txt';
param.pos_rects_file = 'D:\DongChen\FaceData\All_LabelFace_5p\rects_all.txt';
param.pos_points_file = 'D:\DongChen\FaceData\All_LabelFace_5p\points_all.txt';
param.pos_points_num = 5;

param.neg_image_folder = 'D:\DongChen';
param.neg_image_list = 'D:\DongChen\FaceData\All_nonface\imagelist.txt';

%% set param
% input data setting
param.min_face_size = 48;
param.max_face_size = 384;
param.scale_num = 16;
param.pos_aug_num = 10;
param.max_rand_offset = 8;
param.max_img_size = 1000;
param.pos_overlap_ratio = 0.5;
param.channel_num = 3;

% dnn training setting
param.DNN_root_folder = 'D:\DongChen\Matlab\DNNFaceDetection\train\googlenet_finetune';
param.gpu_id = 0:7;
param.validation_rate = 0.1;
param.batch_size_per_gpu = 1;
param.pos_gpu_num = 7;
param.neg_gpu_num = 1;
param.anchor_is_field_center = false; % decide whether the anchor point is the center of receptive field or stride
param.preload_training_data = false;

% recovery or fine-tune
param.recovery_model = '';
param.init_model = 'D:\DongChen\Matlab\DNNFaceDetection\train\googlenet_finetune\bn_v1.caffemodel';

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
