function [anchor_rects, anchor_points, anchor_center] = ComputeAnchorFace(mean_face, rpn_model)

param = rpn_model.param;
if (param.anchor_is_field_center)
    anchor = (rpn_model.receptive_field_size - 1) / 2;
else
    anchor = (rpn_model.output_stride - 1) / 2;
end

anchor_rects = zeros(param.scale_num, 4);
anchor_points = zeros(param.scale_num, param.pos_points_num*2);
for i = 1:param.scale_num
    face_size = param.face_size_class(i);
    lt = anchor - face_size / 2;
    anchor_rects(i, :) = [lt, lt + face_size, lt, lt + face_size];
    anchor_points(i, :) = mean_face * face_size + lt;
end
anchor_center = [anchor anchor];