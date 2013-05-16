function do_learn_label_metric_lmnn(config_file)
% Function that learn label specific distance metric using LMNN model

%% Initial some configurations
eval(config_file);

%load train_feature_full
load(fullfile(RUN_DIR, Global.Train_Feature_Dir, 'train_features_full.mat'));
%load multiple train feature
load(fullfile(RUN_DIR, Global.Train_Feature_Dir, 'train_multifeature_corel5k.mat'));

% load(fullfile(DATA_DIR,'corel5k_train_pairs'));

addpath(genpath('D:\workspace-limu\image annotation\iciap2013\KISSME\toolbox\helper'));
addpath(genpath('D:\software\matlab-toolbox\L1GeneralExamples'));

X = train_features_full;
[P D] = size(X);


%train mahalabios matrix for each label
load(fullfile(RUN_DIR, Global.Train_Feature_Dir, 'seman_group_subset_corel5k.mat'));
L = length(seman_group_subset.label_img_index);

learn_method = Global.Learn_method;  %here mmlmnn

model_dir = [];
learned_model_name = [];
MULTIPLE_FEATURE_DIM = 13800;

%% now learn metric for each label
for i = 1:L
    fprintf('learn metric for label %d\n',i);
    %load sample pairs in i-th label directory
    label_dir = sprintf('label_%d', i);
    
    %initial eloss and geloss
    Eloss = 0;
    gEloss = ones(1, MULTIPLE_FEATURE_DIM); 
    
    if exist(fullfile(LABEL_PAIRS_DIR, 'train', label_dir),'dir')
        sample_index = seman_group_subset.label_img_index{i};
        
        %% loop for each sample to update the loss and gradient
        for s = 1 : length(sample_index)
            %first get the target and imposter for s-th sample
            [inxa, inxb, matches] = sample_target_imposters(LABEL_PAIRS_DIR, label_dir, sample_index(s));
            
            %update eloss geloss for each sample
            [eloss, gEloss] = sample_update_gradient(train_samples, inxa, inxb, matches, gEloss);
            % sum each sample's eloss
            Eloss = Eloss + eloss;
        end
    else
        label_model(i).M = [];
        label_model(i).t = 0;
    end
    
end


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


function [eloss, geloss] = sample_update_gradient(train_samples, idxa, idxb, matches, old_geloss)

if (length(old_geloss) ~= MULTIPLE_FEATURE_DIM)
    error('dimension mismatched!');
end

MULTIPLE_FEATURE_DIM = 13800;
N = length(idxa);
X = zeros(N, MULTIPLE_FEATURE_DIM);

eloss = 0;
geloss = zeros(1,MULTIPLE_FEATURE_DIM);
%create the X contains both distance between sample with its targets and imposters
for d = 1 : N
    Xa = extract_set_samples(idxa(d),train_samples);
    Xb = extract_set_samples(idxb(d),train_samples);
    
    %calculate basis distance
    X(d,:) = base_distance_vector(Xa,Xb);
end    

V_target = X(matches == 1);
D_target = sum(V_target, 2);
V_imposter = X(matches ~= 1);
D_imposter = sum(V_imposter);

mu = 0.5;

[eloss, geloss] = gEloss(old_geloss, D_target, D_imposter, V_target, V_imposter, mu);




end