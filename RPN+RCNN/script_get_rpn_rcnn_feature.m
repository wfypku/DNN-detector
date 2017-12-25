clear all;
close all;
clc;

iter = 570000;
model_file = '\\msra-facednn12\D$\DongChen\Matlab\DNNFaceDetectionOneStep\RPN+RCNN_output\GoogleNetHalf_GoogleNetHalf_fintune_w0_lr10_pr128_nr128_t-3_s1_(clear_5p_neg_clear)_(36_72_3_1000)_(8_10_50_5)\rpn_model.mat';
load(model_file);
rpn_model.param.DNN_root_folder = '\\msra-facednn12\D$\DongChen\Matlab\DNNFaceDetectionOneStep\RPN+RCNN_output\GoogleNetHalf_GoogleNetHalf_fintune_w0_lr10_pr128_nr128_t-3_s1_(clear_5p_neg_clear)_(36_72_3_1000)_(8_10_50_5)';
rpn_model.param.max_img_size = 2000;
rpn_model.dnn_model = [rpn_model.param.DNN_root_folder, '\model_iter_', int2str(iter)];
[rpn_model.dim_map, rpn_model.receptive_field_size, rpn_model.output_stride] = GetDNNOutputDimMap(rpn_model.param);


imagelist = ReadImageListFromFile('D:\DongChen\Matlab\DNNFaceDetection\RawTestData\FDDB\originalPics', 'D:\DongChen\Matlab\DNNFaceDetection\RawTestData\FDDB\imagelist.txt');
[all_ft, all_ft_index, all_faces, all_points, all_weights, all_scales, all_sub_scales] = ExtractRPNRCNNFeatureParallel(imagelist, rpn_model, 2.^(-9:2), 8, 0:7);
save predict_results\fddb.mat all_ft all_ft_index all_faces all_points all_weights all_scales all_sub_scales -v7.3;

imagelist = ReadImageListFromFile('\\msra-vc15\E$\DongChen\FaceData\100new\images', '\\msra-vc15\E$\DongChen\FaceData\100new\100new_filp_imagelist.txt');
[all_ft, all_ft_index, all_faces, all_points, all_weights, all_scales, all_sub_scales] = ExtractRPNRCNNFeatureParallel(imagelist, rpn_model, 2.^(-9:1), 8, 0:7);
save predict_results\100new.mat all_ft all_ft_index all_faces all_points all_weights all_scales all_sub_scales -v7.3;

imagelist = ReadImageListFromFile('\\msra-vc15\E$\DongChen\FaceData\AFLW\Images', '\\msra-vc15\E$\DongChen\FaceData\AFLW\imagelist.txt');
[all_ft, all_ft_index, all_faces, all_points, all_weights, all_scales, all_sub_scales] = ExtractRPNRCNNFeatureParallel(imagelist, rpn_model, 2.^(-9:1), 8, 0:7);
save predict_results\aflw.mat all_ft all_ft_index all_faces all_points all_weights all_scales all_sub_scales -v7.3;

imagelist = ReadImageListFromFile('D:\DongChen\FaceData\PIPA\PersonIdentification', 'D:\DongChen\FaceData\PIPA\imagelist.txt');
[all_ft, all_ft_index, all_faces, all_points, all_weights, all_scales, all_sub_scales] = ExtractRPNRCNNFeatureParallel(imagelist, rpn_model, 2.^(-9:1), 8, 0:7);
save predict_results\pipa.mat all_ft all_ft_index all_faces all_points all_weights all_scales all_sub_scales -v7.3;

imagelist = ReadImageListFromFile('D:\DongChen\FaceData\HeadResult_our\image', 'D:\DongChen\FaceData\HeadResult_our\imagelist.txt');
[all_ft, all_ft_index, all_faces, all_points, all_weights, all_scales, all_sub_scales] = ExtractRPNRCNNFeatureParallel(imagelist, rpn_model, 2.^(-9:1), 8, 0:7);
save predict_results\head_our.mat all_ft all_ft_index all_faces all_points all_weights all_scales all_sub_scales -v7.3;
