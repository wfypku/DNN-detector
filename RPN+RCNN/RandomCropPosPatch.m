function [face_patch, rects_crop, points_crop] = RandomCropPosPatch(img, rect, points, rpn_model)

receptive_field_size = rpn_model.receptive_field_size;
param = rpn_model.param;
min_face_size = param.min_face_size;
max_face_size = param.max_face_size;
assert(max_face_size >= min_face_size);
max_rand_offset = param.max_rand_offset;

max_resize_log_ratio = log2(max_face_size/min_face_size);
resize_log_ratio = max_resize_log_ratio * rand();
resize_face_size = min_face_size*2^resize_log_ratio;
resize_face_left = (receptive_field_size - resize_face_size)/2 + max_rand_offset*(rand()-0.5)*2;
resize_face_top = (receptive_field_size - resize_face_size)/2 + max_rand_offset*(rand()-0.5)*2;
target_rect = [
    resize_face_left, resize_face_top
    resize_face_left + resize_face_size, resize_face_top
    resize_face_left + resize_face_size, resize_face_top + resize_face_size
    resize_face_left, resize_face_top + resize_face_size];
origin_rect = [
    rect(1), rect(3)
    rect(2), rect(3)
    rect(2), rect(4)
    rect(1), rect(4)
    ];
trans = cp2tform(origin_rect, target_rect, 'nonreflective similarity');
face_patch = imtransform(img, trans, 'XData', [1, receptive_field_size], 'YData', [1, receptive_field_size], 'XYScale', 1);
points_crop = reshape(tformfwd(trans, reshape(points, 2, [])')', 1, []);
rects_crop = [resize_face_left, resize_face_left + resize_face_size, resize_face_top, resize_face_top + resize_face_size];





