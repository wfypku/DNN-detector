function dnn_model = TrainDNNModel(training_data, validation_data, rpn_model)

%% init DNN solver
param = rpn_model.param;
root_folder = param.DNN_root_folder;
gpu_id = param.gpu_id;
recovery_model = param.recovery_model;
init_model = param.init_model;
current_dir = pwd;
cd(root_folder);
DNN.caffe_mex('release_solver');
DNN.caffe_mex('set_device_solver', gpu_id);
if (~isempty(recovery_model) && ~strcmp(recovery_model, ''))
    DNN.caffe_mex('recovery_solver', 'solver.prototxt', recovery_model, [root_folder, '\log\']);
elseif (~isempty(init_model) && ~strcmp(init_model, ''))
    DNN.caffe_mex('init_solver', 'solver.prototxt', init_model, [root_folder, '\log\']);
else
    DNN.caffe_mex('init_solver', 'solver.prototxt', '', [root_folder, '\log\']);
end
% DNN.caffe_mex('set_mode_cpu');
cd(current_dir);

%% begin train
test_interval = param.test_interval;
test_iter = param.test_iter;
begin_iter = DNN.caffe_mex('get_solver_iter');
max_iter = DNN.caffe_mex('get_solver_max_iter');
tic;
mean_loss = 0;
mean_loss_part = [];
all_train_loss = [];
all_test_loss = [];
all_iter = [];
if (param.preload_training_data)
	n_pos = size(training_data.pos_data.data, 4);
	n_neg = length(training_data.neg_data);
    n_pos_valid = size(validation_data.pos_data.data, 4);
    n_neg_valid = length(validation_data.neg_data);
else
    n_pos = length(training_data.imagelist_pos);
    n_neg = length(training_data.imagelist_neg);
    n_pos_valid = length(validation_data.imagelist_pos);
    n_neg_valid = length(validation_data.imagelist_neg);
end
pos_order = randperm(n_pos);
neg_order = randperm(n_neg);
for iter = begin_iter+1:max_iter
    sb = iter * param.pos_gpu_num * param.batch_size_per_gpu;
    se = (iter + 1) * param.pos_gpu_num * param.batch_size_per_gpu - 1;
    select_index.sp = pos_order(mod(sb:se, n_pos)+1);
    if (mod(sb, n_pos) > mod(se, n_pos) || mod(sb, n_pos) == 0)
        fprintf('randperm pos data...\n');
        pos_order = randperm(n_pos);
    end
    sb = iter * param.neg_gpu_num;
    se = (iter + 1) * param.neg_gpu_num - 1;
    select_index.sn = neg_order(mod(sb:se, n_neg)+1);
    if (mod(sb, n_neg) > mod(se, n_neg) || mod(sb, n_neg) == 0)
        fprintf('randperm neg data...\n');
        neg_order = randperm(n_neg);
    end
    if (param.preload_training_data)
    	train_batch = GetTrainBatchWithPreloadData(select_index, training_data.pos_data, training_data.neg_data, rpn_model);
    else
    	train_batch = GetTrainBatch(select_index, training_data.imagelist_pos, training_data.rects_pos, training_data.points_pos, training_data.imagelist_neg, rpn_model);
    end
    cd(root_folder);
%     weight_init = DNN.caffe_mex('get_weights_solver');
    loss = DNN.caffe_mex('train', train_batch);
    for i =  1:length(loss)
        if(isnan(loss(i).results))
            disp('NaN in loss..');
            break;
        end
    end

%     weight = DNN.caffe_mex('get_weights_solver');
%     for i = 1:length(weight)
%         for j = 1:length(weight(i).weights)
%             if (sum(sum(isnan(weight(i).weights{j}))) ~= 0)
%                 error('nan');
%             end
%         end
%     end
%     figure(3);
%     hold on;
%     tmp = weight(38).weights{1}(:);
%     plot(tmp(1:2:end), tmp(2:2:end), 'b.');
%     xlim([1, 64]);
%     ylim([1, 64]);
    cd(current_dir);
    if (isempty(mean_loss_part))
        mean_loss_part = zeros(length(loss), 1);
    end
    for i = 1:length(loss)
        mean_loss = mean_loss + loss(i).results * loss(i).weight;
        mean_loss_part(i) = mean_loss_part(i) + loss(i).results;
    end
    if (mod(iter, param.display_interval) == 0)
        fprintf('iter: %d -- loss: %f. (', iter, mean_loss/param.display_interval);
        for i = 1:length(loss)
            fprintf('%.5f*%g ', mean_loss_part(i)/param.display_interval, loss(i).weight);
        end
        fprintf(') ');
        all_train_loss = cat(1, all_train_loss, mean_loss/param.display_interval);
        mean_loss = 0;
        mean_loss_part = [];
        figure(1);
        plot(all_train_loss, 'b-');
        title('train loss');
        toc;
    end
    
    if (param.validation_rate > 0 && test_interval > 0 && test_iter > 0 && mod(iter, test_interval) == 0)
        test_loss = 0;
        for r = 1:test_iter
            select_index.sp = randperm(n_pos_valid, param.pos_gpu_num * param.batch_size_per_gpu);
            select_index.sn = randperm(n_neg_valid, param.neg_gpu_num);
            if (param.preload_training_data)
                test_batch = GetTrainBatchWithPreloadData(select_index, validation_data.pos_data, validation_data.neg_data, rpn_model);
            else
                test_batch = GetTrainBatch(select_index, validation_data.imagelist_pos, validation_data.rects_pos, validation_data.points_pos, validation_data.imagelist_neg, rpn_model);
            end
            loss = DNN.caffe_mex('test', test_batch);
            for i = 1:length(loss)
                test_loss = test_loss + loss(i).results * loss(i).weight;
            end
        end
        test_loss = test_loss / test_iter;
        all_test_loss = cat(1, all_test_loss, test_loss);
        all_iter = cat(1, all_iter, DNN.caffe_mex('get_solver_iter'));
        figure(2);        
        plot(all_iter, all_test_loss, 'r-');
        title('test loss');
        if (min(all_test_loss) == test_loss)
            DNN.caffe_mex('snapshot', [root_folder, '\model_best']);
        end
    end
    drawnow;
end

dnn_model = [root_folder, '\model_best'];


