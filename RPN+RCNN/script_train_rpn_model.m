diary([param.DNN_root_folder, '\diary.txt']);

%% get dim map and receptive size and stride
[rpn_model.dim_map, rpn_model.receptive_field_size, rpn_model.output_stride] = GetDNNOutputDimMap(param);
fprintf('Receptive field size: %d, DNN stride: %d\r\n', rpn_model.receptive_field_size, rpn_model.output_stride);
rpn_model.param = param;

%% read training data
disp('Loading data start...')
imagelist_pos = ReadImageListFromFile(param.pos_image_folder, param.pos_image_list);
rects_pos = ReadFaceRectFromFile(param.pos_rects_file);
points_pos = ReadFacialPointsFromFile(param.pos_points_file, param.pos_points_num);
imagelist_neg = ReadImageListFromFile(param.neg_image_folder, param.neg_image_list);
disp('Loading data end...')
%% calcuate mean face
rpn_model.mean_face = ComputeMeanFace(rects_pos, points_pos);
[rpn_model.anchor_rects, rpn_model.anchor_points, rpn_model.anchor_center] = ComputeAnchorFace(rpn_model.mean_face, rpn_model);

%% split train and valid
training_data = [];
validation_data = [];
if (param.validation_rate > 0 && param.test_interval > 0 && param.test_iter > 0)
    num_valid = round(length(imagelist_pos) * param.validation_rate);
    r = randperm(length(imagelist_pos));
    imagelist_pos_valid = imagelist_pos(r(1:num_valid));
    rects_pos_valid = rects_pos(r(1:num_valid), :);
    points_pos_valid = points_pos(r(1:num_valid), :);
    imagelist_pos = imagelist_pos(r(num_valid+1:end));
    rects_pos = rects_pos(r(num_valid+1:end), :);
    points_pos = points_pos(r(num_valid+1:end), :);
    num_valid = round(length(imagelist_neg)*param.validation_rate);
    r = randperm(length(imagelist_neg));
    imagelist_neg_valid = imagelist_neg(r(1:num_valid));
    imagelist_neg = imagelist_neg(r(num_valid+1:end));
    validation_data.imagelist_pos = imagelist_pos_valid;
    validation_data.rects_pos = rects_pos_valid;
    validation_data.points_pos = points_pos_valid;
    validation_data.imagelist_neg = imagelist_neg_valid;
    if (param.preload_training_data)
        % load and augment training data to speed up training
        validation_data.pos_data = LoadAndAugmentPosTrainingData(imagelist_pos_valid, rects_pos_valid, points_pos_valid, rpn_model);
        validation_data.neg_data = LoadNegTrainingData(imagelist_neg_valid, rpn_model);
    end
end
training_data.imagelist_pos = imagelist_pos;
training_data.rects_pos = rects_pos;
training_data.points_pos = points_pos;
training_data.imagelist_neg = imagelist_neg;
if (param.preload_training_data)
    % load and augment training data to speed up training
    training_data.pos_data = LoadAndAugmentPosTrainingData(imagelist_pos, rects_pos, points_pos, rpn_model);
    training_data.neg_data = LoadNegTrainingData(imagelist_neg, rpn_model);
end

%% RPN training
disp('RPN training start...')
rpn_model.dnn_model = [rpn_model.param.DNN_root_folder, '\model_best'];
save([param.DNN_root_folder, '/rpn_model.mat'], 'rpn_model');
rpn_model.dnn_model = TrainDNNModel(training_data, validation_data, rpn_model);
diary off;
