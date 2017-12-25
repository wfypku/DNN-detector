function pos_data = LoadAndAugmentPosTrainingData(imagelist_pos, rects_pos, points_pos, rpn_model)

param = rpn_model.param;
receptive_field_size = rpn_model.receptive_field_size;
aug_num = param.pos_aug_num;
scale_num = param.scale_num;
anchor_rects = rpn_model.anchor_rects;
anchor_points = rpn_model.anchor_points;
mem_cost = (receptive_field_size*receptive_field_size*param.channel_num + param.pos_points_num*2*scale_num*4*2)*length(imagelist_pos)*aug_num;
fprintf('Generate Pos Training Data. Memory cost: %.2fGB\r\n', mem_cost/1024/1024/1024);

total_num = length(imagelist_pos)*aug_num;
pos_data.data = zeros([receptive_field_size, receptive_field_size, param.channel_num, total_num], 'uint8');
pos_data.label = zeros([1, 1, scale_num, total_num], 'single');
pos_data.delta_points = zeros([1, 1, param.pos_points_num*2*scale_num, total_num], 'single');
pos_data.mask_points = zeros([1, 1, param.pos_points_num*2*scale_num, total_num], 'single');
ct = 1;
tic;
for i = 1:length(imagelist_pos)
    img = ReadColorImage(imagelist_pos{i});
    if (param.channel_num == 1)
        img = rgb2gray(img);
    end
    for j = 1:aug_num
        [face_patch, rects_crop, points_crop] = RandomCropPosPatch(img, rects_pos(i, :), points_pos(i, :), rpn_model);
        overlap = GetRectOverlappedRatio(anchor_rects, repmat(rects_crop, param.scale_num, 1));
        flag = overlap >= param.pos_overlap_ratio;
        [~, p] = max(overlap);
        flag(p) = true;
        pos_data.data(:, :, :, ct) = DNN.ConvertImageFormatForCaffe(face_patch);
        pos_data.label(1, 1, :, ct) = single(flag);
        for scale_class = find(flag)'
            pos_data.delta_points(1, 1, (scale_class-1)*param.pos_points_num*2+1:scale_class*param.pos_points_num*2, ct) = ...
                single(points_crop - anchor_points(scale_class, :))' / param.face_size_class(scale_class);
        end
        pos_data.mask_points(1, 1, :, ct) = reshape(repmat(single(flag), 1, param.pos_points_num*2)', [], 1);
        ct = ct + 1;
    end
    if (mod(i, 100) == 0)
        fprintf('load %d/%d... ', i, length(imagelist_pos));
        toc;
    end
end

