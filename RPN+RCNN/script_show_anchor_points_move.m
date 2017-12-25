clear all;
close all;
clc;

root_folder = 'D:\DongChen\Matlab\DNNFaceDetectionOneStep\RPN+RCNN_output\GoogleNetHalf_GoogleNetHalf_fintune_w0_lr100_pr128_nr128_t-3_offset_-20_s1_(clear_5p_neg_clear)_(36_72_3_1000)_(8_10_50_5)';

figure(1);
colors = hsv(100);
ct = 1;
for iter = 10000:10000:1000000
    dnn_model = [root_folder, '\model_iter_', int2str(iter)];
    if (~exist(dnn_model, 'file'))
        break;
    end
    current_dir = pwd;
    cd(root_folder);
    DNN.caffe_mex('release_solver');
    DNN.caffe_mex('set_device_solver', 0);
    DNN.caffe_mex('init_solver', 'solver.prototxt', dnn_model, [root_folder, '\log\']);
    cd(current_dir);
    
    weights = DNN.caffe_mex('get_weights_solver');
    assert(strcmp(weights(38).layer_names, 'similairty_transform'));
    pts = weights(38).weights{1}(:)';
    hold on;
    plot(pts(1:2:end), pts(2:2:end), '.', 'color', colors(ct, :));
    ct = ct + 1;
    xlim([0, 63]);
    ylim([0, 63]);
end
print('-dpng', [root_folder, '\anchor_move.png']);