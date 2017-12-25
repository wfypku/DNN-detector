function SetInputSizeSolver(input)

n = length(input);
input_size = cell(n, 1);
for i = 1:n
    v = input{i};
    s = zeros(length(v)*4, 1);
    for j = 1:length(v)
        s(4*j-3) = size(v{j}, 1);
        s(4*j-2) = size(v{j}, 2);
        s(4*j-1) = size(v{j}, 3);
        s(4*j) = size(v{j}, 4);
    end
    input_size{i} = s;
end

DNN.caffe_mex('set_input_size_solver', input_size);
