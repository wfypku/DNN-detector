clear all;
close all;
fclose all;
clc;

%% compare with other method
baidu = load('D:\DongChen\Matlab\DNNFaceDetection\pr_curves\fddb\BAIDU-IDL-disc-v2.txt');
linkface = load('D:\DongChen\Matlab\DNNFaceDetection\pr_curves\fddb\Linkface-DiscROC_v3.txt');
ours = cell(0);
% ours{end+1, 1} = {'rcnn', 'D:\DongChen\Matlab\DNNFaceDetection\tmp\model_final.test_weights.mat'};
ours{end+1, 1} = {'no reg', '\\msra-facednn07\D$\DongChen\Matlab\DNNFaceDetection\train\6conv_nopad\model_best.precision_recall.mat'};
% ours{end+1, 1} = {'reg_iter240000', 'D:\DongChen\Matlab\DNNFaceDetection\train\6conv_reg\train_iter_240000.precision_recall.mat'};
ours{end+1, 1} = {'RPN scale 1 + dense pyramid', '\\msra-facednn07\D$\DongChen\Matlab\DNNFaceDetection\train\6conv_reg\train_iter_580000.precision_recall_fddb.mat'};
ours{end+1, 1} = {'V3 best', '\\msra-facednn07\D$\DongChen\Matlab\DNNFaceDetection\train\6conv_reg_v3\model_best.precision_recall_fddb.mat'};
% ours{end+1, 1} = {'V3 iter 340000', '\\msra-facednn07\D$\DongChen\Matlab\DNNFaceDetection\train\6conv_reg_v3\train_iter_340000.precision_recall_fddb.mat'};
ours{end+1, 1} = {'V2 scale 1', '\\msravc-facednn1\D$\DongChen\Matlab\DNNFaceDetection\train\6conv_s1\model_best.precision_recall_fddb.mat'};
ours{end+1, 1} = {'V2 scale 1 iter 1240000', '\\msravc-facednn1\D$\DongChen\Matlab\DNNFaceDetection\train\6conv_s1\train_iter_1240000.precision_recall_fddb.mat'};

colors = colormap(hsv(length(ours)+2));
figure(1);
hold on;
plot(baidu(:, 2), baidu(:, 1), '-', 'color', colors(1, :));
plot(linkface(:, 2), linkface(:, 1), '-', 'color', colors(2, :));
plot(14, 0.8410, 'kx');
for i = 1:length(ours)
    v = load(ours{i}{2});
    plot(v.err_nums, v.recalls, '-', 'color', colors(i+2, :));
end

leg = cell(length(ours) + 3, 1);
leg{1} = 'baidu';
leg{2} = 'linkface';
leg{3} = 'jda';
for i = 1:length(ours)
    leg{3+i} = ours{i}{1};
end
legend(leg, 'Location', 'SouthEast');
grid on;
xlim([0, 100]);
ylim([0.6, 0.90]);

