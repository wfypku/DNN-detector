function [img_out, rect_out, points_out] = RandomResizePosImage(img, rect, points, rpn_model)

param = rpn_model.param;
% random scale
rand_face_size = 2^(log2(param.min_face_size) + log2(param.max_face_size/param.min_face_size) * rand());
input_face_size = ((rect(2) - rect(1)) + (rect(4) - rect(3)))/2;
scale = rand_face_size / input_face_size;
height_out = round(size(img, 1) * scale);
width_out = round(size(img, 2) * scale);
img_out = imresize(img, [height_out, width_out]);
scale_y = (height_out - 1)/(size(img, 1) - 1);
scale_x = (width_out - 1)/(size(img, 2) - 1);
points_out = points;
points_out(1:2:end) = points(1:2:end) * scale_x;
points_out(2:2:end) = points(2:2:end) * scale_y;
rect_out = rect * scale;

% random offset
offset_x = randi(param.max_rand_offset) - 1;
offset_y = randi(param.max_rand_offset) - 1;
img_out = img_out(offset_y+1:end, offset_x+1:end, :);
rect_out = rect_out - [offset_x offset_x offset_y offset_y];
points_out(1:2:end) = points_out(1:2:end) - offset_x;
points_out(2:2:end) = points_out(2:2:end) - offset_y;
height_out = size(img_out, 1);
width_out = size(img_out, 2);

% if img_out is too small or too large
if (width_out < rpn_model.receptive_field_size)
    pad = ceil((rpn_model.receptive_field_size - width_out)/2);
    tmp = zeros([height_out, width_out+2*pad, size(img_out, 3)], 'like', img_out);
    tmp(:, pad+1:pad+width_out, :) = img_out;
    rect_out(1:2) = rect_out(1:2) + pad;
    points_out(1:2:end) = points_out(1:2:end) + pad;
    img_out = tmp;
elseif (width_out > param.max_img_size);
    crop = round(mean(rect_out(1:2)) - param.max_img_size/2);
    crop = max(crop, 1);
    crop = min(width_out - param.max_img_size + 1, crop);
    img_out = img_out(:, crop:crop+param.max_img_size-1, :);
    rect_out(1:2) = rect_out(1:2) - crop + 1;
    points_out(1:2:end) = points_out(1:2:end) - crop + 1;
end

height_out = size(img_out, 1);
width_out = size(img_out, 2);
if (height_out < rpn_model.receptive_field_size)
    pad = ceil((rpn_model.receptive_field_size - height_out)/2);
    tmp = zeros([height_out+2*pad, width_out, size(img_out, 3)], 'like', img_out);
    tmp(pad+1:pad+height_out, :, :) = img_out;
    rect_out(3:4) = rect_out(3:4) + pad;
    points_out(2:2:end) = points_out(2:2:end) + pad;
    img_out = tmp;
elseif (height_out > param.max_img_size)
    crop = round(mean(rect_out(3:4)) - param.max_img_size/2);
    crop = max(crop, 1);
    crop = min(height_out - param.max_img_size + 1, crop);
    img_out = img_out(crop:crop+param.max_img_size-1, :, :);
    rect_out(3:4) = rect_out(3:4) - crop + 1;
    points_out(2:2:end) = points_out(2:2:end) - crop + 1;
end
