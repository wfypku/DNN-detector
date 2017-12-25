function [img_out, dim_out] = RandomResizeNegImage(img, rpn_model)

param = rpn_model.param;
min_size = rpn_model.receptive_field_size + (param.max_img_size - rpn_model.receptive_field_size) * rand();
resize_scale = min(min_size / size(img, 1), min_size / size(img, 2));
resize_dim = round([size(img, 1), size(img, 2)] * resize_scale);
resize_dim = max(resize_dim, rpn_model.receptive_field_size);
img_out = imresize(img, resize_dim);

dim_out = [rpn_model.dim_map(size(img_out, 1)), rpn_model.dim_map(size(img_out, 2))];
