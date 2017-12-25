function mean_face = ComputeMeanFace(rects_pos, points_pos)

face_size = ((rects_pos(:, 2) - rects_pos(:, 1)) + (rects_pos(:, 4) - rects_pos(:, 3))) / 2;
points_norm = points_pos - repmat([rects_pos(:, 1), rects_pos(:, 3)], 1, size(points_pos, 2)/2);
points_norm = bsxfun(@times, points_norm, 1./face_size);
mean_face = median(points_norm, 1);
