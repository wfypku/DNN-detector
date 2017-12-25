clear all;
close all;
fclose all;
clc;

load('debug_data');

for i = 1:size(pos_data.data, 4)
    imshow(pos_data.data(:, :, :, i)');
    hold on;
    l = pos_data.label(:, :, :, i);
    delta = pos_data.delta_points(:, :, :, i);
    delta = reshape(delta, 54, [])';
    for j = find(l)'
        DrawRects(rpn_model.anchor_rects(j, :), 'r', 1);
        pt = rpn_model.anchor_points(j, :) + delta(j, :);
        plot(pt(1:2:end), pt(2:2:end), 'y+');
    end
    hold off;
    waitforbuttonpress;
end