function [mask_pos, mask_points, label, delta] = GetMaskPosForTraining(img, rect, points, rpn_model)

param = rpn_model.param;
height_out = rpn_model.dim_map(size(img, 1));
width_out = rpn_model.dim_map(size(img, 2));
scale_num = param.scale_num;
points_num = param.pos_points_num;
stride = rpn_model.output_stride;

mask_pos = zeros([height_out, width_out], 'single');
label = zeros([scale_num, height_out*width_out], 'single');
delta = zeros([scale_num*points_num*2, height_out*width_out], 'single');
mask_points = zeros([scale_num*points_num*2, height_out*width_out], 'single');
ct = 1;
x_perfect = round(((rect(1) + rect(2))/2 - rpn_model.anchor_center(1))/stride) + 1;
y_perfect = round(((rect(3) + rect(4))/2 - rpn_model.anchor_center(2))/stride) + 1;
for y = 1:height_out
    for x = 1:width_out
        over_lap = GetRectOverlappedRatioMex(rect-[x-1, x-1, y-1, y-1]*stride, rpn_model.anchor_rects);
        if (x_perfect == x && y_perfect == y)
            over_lap(over_lap == max(over_lap)) = 1;
        end
        if (max(over_lap) < param.pos_overlap_ratio)
            continue;
        end
        mask_pos(y, x) = 1;
        flag = over_lap >= param.pos_overlap_ratio;
        label(:, ct) = flag;
        mask_points(:, ct) = reshape(repmat(flag, 1, points_num*2)', [], 1);
        p = points - repmat([x-1, y-1]*stride, 1, points_num);
        delta(:, ct) = repmat(p(:), scale_num, 1) - reshape(rpn_model.anchor_points', [], 1);
        delta(:, ct) = delta(:, ct) .* mask_points(:, ct) ./ param.points_scale_value;
        ct = ct + 1;
    end
end
ct = ct - 1;
mask_pos = mask_pos';
label = reshape(label(:, 1:ct), [1, 1, scale_num, ct]);
delta = reshape(delta(:, 1:ct), [1, 1, scale_num*points_num*2, ct]);
mask_points = reshape(mask_points(:,1:ct), [1, 1, scale_num*points_num*2, ct]);

