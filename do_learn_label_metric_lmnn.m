function do_learn_label_metric_lmnn(config_file)
% Function that learn label specific distance metric using LMNN model

clc;
%% Initial some configurations
eval(config_file);

%load train_feature_full
load(fullfile(RUN_DIR, Global.Train_Feature_Dir, 'train_features_full.mat'));
%load multiple train feature
% load(fullfile(RUN_DIR, Global.Train_Feature_Dir, 'train_multifeature_corel5k.mat'));

% load(fullfile(DATA_DIR,'corel5k_train_pairs'));

addpath(genpath('D:\workspace-limu\image annotation\iciap2013\KISSME\toolbox\helper'));
addpath(genpath('D:\software\matlab-toolbox\L1GeneralExamples'));

X_full = train_features_full;
% [P D] = size(X);


%train mahalabios matrix for each label
load(fullfile(RUN_DIR, Global.Train_Feature_Dir, 'seman_group_subset_corel5k.mat'));
L = length(seman_group_subset.label_img_index);

learn_method = Global.Learn_method;  %here mmlmnn

model_dir = [];
learned_model_name = [];
MULTIPLE_FEATURE_DIM = 13900;
Max_iter = Global.Max_Iteration;

MODEL_LMNN_DIR = 'D:\workspace-limu\cloud disk\Dropbox\limu\lmnn_model';

%% now learn metric for each label
for i = 1:L
    fprintf('learn metric for label %d\n',i);
    %load sample pairs in i-th label directory
    label_dir = sprintf('label_%d', i);
    
    %initial eloss and geloss
    
    eta = 1;
    tol = 1e-3;
    prev_Eloss = Inf;
    best_Eloss = Inf;
    W = ones(1, MULTIPLE_FEATURE_DIM);
    best_W = W;
    % Perform main learning iterations
    iter = 1;
%     prev_Eloss - Eloss > tol || 

    tstart = tic;
    while (iter < Max_iter)
        
        N = 0;
        gEloss = zeros(1, MULTIPLE_FEATURE_DIM); 
        
        label_pairs_dir = LABEL_PAIRS_DIR;
        if exist(fullfile(LABEL_PAIRS_DIR, 'train', label_dir),'dir')
            sample_index = seman_group_subset.label_img_index{i};
            num_samples = length(sample_index);
            %pre allocate 
            sample_gEloss = zeros(length(sample_index),MULTIPLE_FEATURE_DIM);
            sample_eloss = zeros(length(sample_index), 1);
            %% loop for each sample to update the loss and gradient
            STEP = floor(num_samples / 5);
            
            parfor s = 1 : num_samples
                %first get the target and imposter for s-th sample
                [inxa, inxb, matches] = sample_target_imposters(label_pairs_dir, label_dir, sample_index(s));

                %update eloss geloss for each sample
                [sample_eloss(s), sample_gEloss(s,:)] = sample_update_gradient(X_full, inxa, inxb, matches, gEloss, W);
                % sum each sample's eloss

                N = N + length(inxa);
                
                if(mod(s, STEP) == 0)
                    fprintf('...... finished %d-th sample in total %d samples ....... \n', s, num_samples);
                end
            end
            Eloss = sum(sample_eloss,1);
            gEloss = sum(sample_gEloss,1);
            %update W
            W = W + (eta ./ N) .* gEloss;
            
            if prev_Eloss > Eloss
                eta = eta * 1.1;
            else
                eta = eta * .8;
            end
            
            prev_Eloss = Eloss;
            
            if Eloss < best_Eloss
                best_W = W;
                best_Eloss = Eloss;
            else
                fprintf('break point happend at %d-th iteration, stop iteration, return best metric! \n', iter);
                break;
            end
            
            fprintf('the %d-th iteration, Eloss: %f \n', iter, Eloss);
        else
            fprintf('there is no samples for label %d, quit!', i);
%             label_model(i).M = [];
%             label_model(i).t = 0;
            break;
        end
    
        iter = iter + 1;
    end
    ellipse_time = toc(tstart);
    fprintf('Iteration finished! Using time: %f \n' ,ellipse_time);
    % Return best metric
    label_model(i).M = best_W;
    label_model(i).t = ellipse_time;
    
    % save learned parameters for each label in the public dropbox dir
    model_dir = 'label_basedLMNN';
    if ~exist(fullfile(MODEL_LMNN_DIR, model_dir), 'dir')
        mkdir(fullfile(MODEL_LMNN_DIR, model_dir));
    end
    label_i_model = sprintf('label_%d_W.mat',i);
    save(fullfile(MODEL_LMNN_DIR, model_dir, label_i_model), 'best_W', 'ellipse_time');
end

%% set model name, save in local dir
model_dir = 'label_basedLMNN';
learned_model_name = 'label_basedLMNN.mat';
    
if ~exist(fullfile(MODEL_DIR, model_dir), 'dir')
    mkdir(fullfile(MODEL_DIR, model_dir));
end

save(fullfile(MODEL_DIR, model_dir, learned_model_name), 'label_model');

end




%% this function is to get targets and imposters for indexed sample
function [ixda, ixdb, matches] = sample_target_imposters(label_pairs_dir, label_dir, sample_index)
    pairs_all = [];
    overlap_a = []; %test img id
    overlap_b = []; %overlap img id
    overlap_match = [];
        
    %load sample pairs for i-th sample
    try
        label_mat_name = sprintf('sample_%d.mat', sample_index);
        load(fullfile(label_pairs_dir, 'train', label_dir, label_mat_name)); 

        index_sim = find([label_pairs.match] == 1);
        sample_id = label_pairs(1).img1.id;

        %here consider the overlapped samples to be positve samples
        num_overlap = length(label_pairs_overlap);
        overlap_a = [overlap_a, ones(1, num_overlap) * sample_id];
        overlap_b = [overlap_b, label_pairs_overlap'];
        overlap_match = [overlap_match, logical(label_pairs_overlap')];

        index_dissim = find([label_pairs.match] == 0);

        preserved_index = [index_sim, index_dissim];
        pairs_all = label_pairs(preserved_index);
 
    catch
        lasterr
    end
    
    temp_a = [pairs_all.img1];
    temp_b = [pairs_all.img2];     
    ixda = [temp_a.id, overlap_a];
    ixdb = [temp_b.id, overlap_b];
    matches = [logical([pairs_all.match]), overlap_match];
        
end


function [eloss, geloss] = sample_update_gradient(X_full, idxa, idxb, matches, old_geloss, w)

MULTIPLE_FEATURE_DIM = 13900;
if (length(old_geloss) ~= MULTIPLE_FEATURE_DIM)
    error('dimension mismatched!');
end

%create the X contains both distance between sample with its targets and imposters
% N = length(idxa);
% X = zeros(N, MULTIPLE_FEATURE_DIM);
% for d = 1 : N
%     Xa = extract_set_samples(idxa(d),train_samples);
%     Xb = extract_set_samples(idxb(d),train_samples);
%     
%     %calculate basis distance
%     X(d,:) = base_distance_vector(Xa,Xb);
% end    

sample_vector = X_full(idxa(1),:);
set_vector = X_full(idxb,:);

Dist = distance_multiple_features(sample_vector, set_vector, w);

V_target = Dist(matches == 1, :);
D_target = sum(V_target, 2);
V_imposter = Dist(matches ~= 1, :);
D_imposter = sum(V_imposter, 2);

mu = 0.5;
% eloss = 0;
% geloss = zeros(1,MULTIPLE_FEATURE_DIM);

[eloss, geloss] = gEloss(old_geloss, D_target, D_imposter, V_target, V_imposter, mu);


end


%% this function is to extract multifeature structures for indexed imgs
function set_samples = extract_set_samples(sample_index, train_samples)

    set_samples.denseHUE = train_samples.denseHUE(sample_index,:);
    set_samples.denseSIFT = train_samples.denseSIFT(sample_index,:);
    set_samples.GIST = train_samples.GIST(sample_index,:);
    set_samples.HSV = train_samples.HSV(sample_index,:);
    set_samples.LAB = train_samples.LAB(sample_index,:);
    set_samples.RGB = train_samples.RGB(sample_index,:);

end