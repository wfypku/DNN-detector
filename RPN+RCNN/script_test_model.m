clear all;
close all;
clc;

model_file = 'D:\DongChen\Matlab\DNNFaceDetectionOneStep\RPN+RCNN_output\GoogleNetHalf_GoogleNetHalf_fintune_w0_lr100_pr128_nr128_t-3_s1_(clear_5p_neg_clear)_(36_72_3_1000)_(8_10_50_5)\rpn_model.mat';
load(model_file);
rpn_model.param.DNN_root_folder = 'D:\DongChen\Matlab\DNNFaceDetectionOneStep\RPN+RCNN_output\GoogleNetHalf_GoogleNetHalf_fintune_w0_lr100_pr128_nr128_t-3_s1_(clear_5p_neg_clear)_(36_72_3_1000)_(8_10_50_5)';
rpn_model.param.max_img_size = 2000;

for iter = 730000:-10000:570000
    rpn_model.dnn_model = [rpn_model.param.DNN_root_folder, '\model_iter_', int2str(iter)];
    [rpn_model.dim_map, rpn_model.receptive_field_size, rpn_model.output_stride] = GetDNNOutputDimMap(rpn_model.param);
    
%     dataset_name = '';
%     imagelist = {'D:\DongChen\Matlab\DNNFaceDetection\RawTestData\FDDB\originalPics\2002\07\23\big\img_425.jpg'};
%     imagelist = {'D:\DongChen\Matlab\DNNFaceDetection\RawTestData\AFW\testimages\18489332.jpg'};
    % imagelist = {'D:\DongChen\Matlab\DNNFaceDetection\RawTestData\FDDB\originalPics\2002\08\02\big\img_1231.jpg'};
    % imagelist = {'D:\DongChen\Matlab\DNNFaceDetection\RawTestData\FDDB\originalPics\2002\09\01\big\img_16378.jpg'};
%     imagelist = {'\\dongdnn\D$\DongChen\photo\9e5389bbtw1ef1bx7jih8j20b20cijt1.jpg'};
    % imagelist = {'\\dongdnn\D$\DongChen\photo\Other_Photos\IMG_8974_small.jpg'};
%     imagelist = {'D:\DongChen\Matlab\DNNFaceDetection\RawTrainingData\Positive\AFLW\0\image13265.jpg'};
%     imagelist = {'D:\DongChen\FaceData\Person_noface\AFLW\0\image00070.jpg'};
%     imagelist = {'D:\DongChen\Matlab\DNNFaceDetection\RawTestData\FDDB\originalPics\2002\08\06\big\img_2717.jpg'};
    % fddb
    dataset_name = 'fddb_-9_2';
    imagelist = ReadImageListFromFile('D:\DongChen\Matlab\DNNFaceDetection\RawTestData\FDDB\originalPics', 'D:\DongChen\Matlab\DNNFaceDetection\RawTestData\FDDB\imagelist.txt');
    all_faces_label = ReadAllFacesFromFile('D:\DongChen\Matlab\DNNFaceDetection\RawTestData\FDDB\all_faces_label.txt');
    % AFW
    % dataset_name = 'afw_-9_2';
    % imagelist = ReadImageListFromFile('D:\DongChen\Matlab\DNNFaceDetection\RawTestData\AFW\testimages', 'D:\DongChen\Matlab\DNNFaceDetection\RawTestData\AFW\imagelist.txt');
    % all_faces_label = ReadAllFacesFromFile('D:\DongChen\Matlab\DNNFaceDetection\RawTestData\AFW\all_faces_label.txt');
    % coco
    % dataset_name = 'coco';
    % imagelist = ReadImageListFromFile('D:\DongChen\FaceData\coco\no_person', 'D:\DongChen\FaceData\coco\imagelist.txt');
    % all_faces_label = cell(length(imagelist), 1);
    % AFLW_neg
    % dataset_name = 'AFLW_neg';
    % imagelist = ReadImageListFromFile('D:\DongChen\FaceData\Person_noface', 'D:\DongChen\FaceData\Person_noface\imagelist_noface.txt');
    % all_faces_label = cell(length(imagelist), 1);
    % WiderFace
%     dataset_name = 'WiderFace_-9_0';
%     imagelist = ReadImageListFromFile('\\msra-vc15\E$\DongChen\FaceData\WiderFace\WiderFace', '\\msra-vc15\E$\DongChen\FaceData\WiderFace\imagelist_all.txt');
%     all_faces_label = ReadAllFacesFromFile('\\msra-vc15\E$\DongChen\FaceData\WiderFace\all_faces_all.txt');

    %% face detection
    [all_faces_predict, all_points_predict, all_weights_predict, all_scales_predict, all_sub_scales_predict] = ...
        DetectFacesParallel(imagelist, rpn_model, 2.^(-9:2), 0, 0:min(7, length(imagelist)-1));
    if (strcmp(dataset_name, ''))
        if (rpn_model.param.pos_points_num ~= 27)
            all_points_predict = cellfun(@(x)[x, zeros(size(x, 1), 54-rpn_model.param.pos_points_num*2)], all_points_predict, 'UniformOutput', false);
        end
        [all_faces_merge, all_points_merge, all_weights_merge] = MergeAllDetectionResult(all_faces_predict, all_points_predict, all_weights_predict, 15);
        ShowFacialPointsAndFaceRect(imagelist, all_points_merge, all_faces_merge, false);
        ShowFacialPointsAndFaceRect(imagelist, all_points_predict, all_faces_predict, false);
        return;
    end
    
    %% evaluate precision recall
    % all_faces_predict_ = cellfun(@GetRe, all_points_predict, 'UniformOutput', false);
    % all_faces_predict = all_faces_predict_;
    w = cell2mat(all_weights_predict);
    wmax = max(w(:, 1));
    wmin = max(3, min(w(:, 1)));
    thrs = wmin : (wmax-wmin)/100 : wmax;
    n = length(thrs);
    recalls = zeros(n, 1);
    precisions = zeros(n, 1);
    err_nums = zeros(n, 1);
    if (rpn_model.param.pos_points_num ~= 27)
        if (size(all_points_predict{1}, 2) < 54)
            all_points_predict = cellfun(@(x)[x, zeros(size(x, 1), 54-rpn_model.param.pos_points_num*2)], all_points_predict, 'UniformOutput', false);
        elseif (size(all_points_predict{1}, 2) > 54)
            all_points_predict = cellfun(@(x)x(:, 1:min(size(x, 2), 54)), all_points_predict, 'UniformOutput', false);
        end
    end
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
    
    %% save result
    SaveAllFacesToFile(all_faces_predict, [rpn_model.dnn_model, '.predict_faces_', dataset_name, '.txt']);
    SaveAllFacialPointsToFile(all_weights_predict, [rpn_model.dnn_model, '.predict_weights_', dataset_name, '.txt']);
    SaveAllFacialPointsToFile(all_points_predict, [rpn_model.dnn_model, '.predict_points_', dataset_name, '.txt']);
    SaveAllFacialPointsToFile(all_scales_predict, [rpn_model.dnn_model, '.predict_scales_', dataset_name, '.txt']);
    SaveAllFacialPointsToFile(all_sub_scales_predict, [rpn_model.dnn_model, '.predict_sub_scales_', dataset_name, '.txt']);
    save([rpn_model.dnn_model, '.precision_recall_', dataset_name, '.mat'], 'recalls', 'precisions', 'err_nums', 'thrs');
    print('-dpng', [rpn_model.dnn_model, '.precision_recall_', dataset_name, '.png']);
end

%% show result
all_faces_predict = ReadAllFacesFromFile([rpn_model.dnn_model, '.predict_faces_', dataset_name, '.txt']);
all_weights_predict = ReadAllFacialPointsFromFile([rpn_model.dnn_model, '.predict_weights_', dataset_name, '.txt']);
all_points_predict = ReadAllFacialPointsFromFile([rpn_model.dnn_model, '.predict_points_', dataset_name, '.txt']);
all_scales_predict = ReadAllFacialPointsFromFile([rpn_model.dnn_model, '.predict_scales_', dataset_name, '.txt']);
[all_faces_merge, all_points_merge, all_weights_merge] = MergeAllDetectionResult(all_faces_predict, all_points_predict, all_weights_predict, 8.87);%thrs(find(recalls > 0.16, 1, 'last')));
[recall, precision, err_image, miss_image, err_detail, miss_detail, err_num, match_info] = EvaluateFaceDetectionResult(all_faces_label, all_faces_merge, 0.2);
ShowFacialPointsAndFaceRect(imagelist, all_points_merge, all_faces_merge, false);
ShowFacialPointsAndFaceRect(imagelist, all_points_merge, err_detail, false, err_image);
ShowFacialPointsAndFaceRect(imagelist, all_points_merge, miss_detail, false, miss_image);
SaveAllFacesToFile(all_faces_merge, [rpn_model.dnn_model, '.merge_faces_', dataset_name, '.txt']);
SaveAllFacialPointsToFile(all_points_merge, [rpn_model.dnn_model, '.merge_points_', dataset_name, '.txt']);
SaveAllFacialPointsToFile(all_weights_merge, [rpn_model.dnn_model, '.merge_weights_', dataset_name, '.txt']);

%% show all curves
all_curves = GetFiles(rpn_model.param.DNN_root_folder, '*.mat', false);
figure();
hold on;
colors = hsv(length(all_curves));
for i = 1:length(all_curves)%[1  17]
    pr = load(all_curves{i});
    if (isfield(pr, 'err_nums') && isfield(pr, 'recalls'))
        plot(pr.err_nums, pr.recalls, 'b-', 'color', colors(i, :));
        index = find(pr.err_nums == 61);
        if (isempty(index))
            continue;
        end
        if (pr.recalls(index) > 0.897)
            fprintf('%s\n', all_curves{i});
        end
    end
end
xlabel('false alarm number');
ylabel('recall');
xlim([0, 250]);
ylim([0.6, 1]);
grid on;
hold off;
