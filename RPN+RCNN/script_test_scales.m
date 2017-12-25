clear all;
close all;
clc;

rpn_model.param.DNN_root_folder = 'D:\DongChen\Matlab\DNNFaceDetection\train\googlenet_aflw_s2';
rpn_model.dnn_model = 'D:\DongChen\Matlab\DNNFaceDetection\train\googlenet_aflw_s2\train_iter_340000';
dataset_name = 'fddb';
imagelist = ReadImageListFromFile('D:\DongChen\Matlab\DNNFaceDetection\RawTestData\FDDB\originalPics', 'D:\DongChen\Matlab\DNNFaceDetection\RawTestData\FDDB\imagelist.txt');
all_faces_label = ReadAllFacesFromFile('D:\DongChen\Matlab\DNNFaceDetection\RawTestData\FDDB\all_faces_label.txt');

all_faces_predict = ReadAllFacesFromFile([rpn_model.dnn_model, '.predict_faces_', dataset_name, '.txt']);
all_weights_predict = ReadAllFacialPointsFromFile([rpn_model.dnn_model, '.predict_weights_', dataset_name, '.txt']);
all_points_predict = ReadAllFacialPointsFromFile([rpn_model.dnn_model, '.predict_points_', dataset_name, '.txt']);
all_scales_predict = ReadAllFacialPointsFromFile([rpn_model.dnn_model, '.predict_scales_', dataset_name, '.txt']);
all_subscales_predict = ReadAllFacialPointsFromFile([rpn_model.dnn_model, '.predict_sub_scales_', dataset_name, '.txt']);

%% result of specific scale and subscale
interest_sub_scale = 0;
for i = 1:length(all_faces_predict)
    f = all_subscales_predict{i}(:, 1) == interest_sub_scale;
    if (isempty(f))
        continue;
    end
    all_faces_predict{i} = all_faces_predict{i}(f, :);
    all_weights_predict{i} = all_weights_predict{i}(f, :);
    all_points_predict{i} = all_points_predict{i}(f, :);
    all_scales_predict{i} = all_scales_predict{i}(f, :);
    all_subscales_predict{i} = all_subscales_predict{i}(f, :);
end

%% pr curve 
w = cell2mat(all_weights_predict);
wmax = max(w(:, 1));
wmin = max(3, min(w(:, 1)));
thrs = wmin : (wmax-wmin)/100 : wmax;
n = length(thrs);
recalls = zeros(n, 1);
precisions = zeros(n, 1);
err_nums = zeros(n, 1);
for i = 1:n
    [all_faces_merge, ~, all_weights_merge] = MergeAllDetectionResult(all_faces_predict, all_points_predict, all_weights_predict, thrs(i));
    [recalls(i), precisions(i), err_image, miss_image, err_detail, miss_detail, err_nums(i), match_info] = EvaluateFaceDetectionResult(all_faces_label, all_faces_merge, 0.2);
end
plot(err_nums, recalls, 'b-');
xlabel('false alarm number');
ylabel('recall');
xlim([0, 250]);
ylim([0.6, 1]);
grid on;

print('-dpng', [rpn_model.dnn_model, '.precision_recall_', dataset_name, '_subscale', int2str(interest_sub_scale), '.png']);

