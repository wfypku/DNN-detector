function [all_faces, all_points, all_weights, all_scales, all_sub_scales] = DetectFacesParallel(imagelist, rpn_model, pyramid_scale, thr, gpu_id)

if (length(gpu_id) == 1)
    [all_faces, all_points, all_weights, all_scales, all_sub_scales] = DetectFaces(imagelist, rpn_model, pyramid_scale, thr, gpu_id);
else
    n = length(imagelist);
    all_faces = cell(n, 1);
    all_points = cell(n, 1);
    all_weights = cell(n, 1);
    all_scales = cell(n, 1);
    all_sub_scales = cell(n, 1);
    gpu_num = length(gpu_id);
    if (matlabpool('size') ~= gpu_num)
        matlabpool('open', gpu_num);
    end
    all_faces_parts = cell(gpu_num, 1);
    all_points_parts = cell(gpu_num, 1);
    all_weights_parts = cell(gpu_num, 1);
    all_scales_parts = cell(gpu_num, 1);
    all_sub_scales_parts = cell(gpu_num, 1);
    parfor i = 1:gpu_num
        [all_faces_parts{i}, all_points_parts{i}, all_weights_parts{i}, all_scales_parts{i}, all_sub_scales_parts{i}]...
            = DetectFaces(imagelist(i:gpu_num:n), rpn_model, pyramid_scale, thr, gpu_id(i));
    end
    for i = 1:gpu_num
        all_faces(i:gpu_num:n) = all_faces_parts{i};
        all_points(i:gpu_num:n) = all_points_parts{i};
        all_weights(i:gpu_num:n) = all_weights_parts{i};
        all_scales(i:gpu_num:n) = all_scales_parts{i};
        all_sub_scales(i:gpu_num:n) = all_sub_scales_parts{i};
    end
end
