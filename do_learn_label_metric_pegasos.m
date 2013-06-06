function do_learn_label_metric_lmnn(config_file)
% Function that learn label specific distance metric using LMNN model

clc;
%% Initial some configurations
eval(config_file);

%load train_feature_full
% load(fullfile(RUN_DIR, Global.Train_Feature_Dir, 'train_features_full.mat'));
%load multiple train feature
load(fullfile(RUN_DIR, Global.Train_Feature_Dir, 'train_multifeature_corel5k.mat'));

% load(fullfile(DATA_DIR,'corel5k_train_pairs'));

addpath(genpath('D:\workspace-limu\image annotation\iciap2013\KISSME\toolbox\helper'));
addpath(genpath('D:\software\matlab-toolbox\L1GeneralExamples'));

% global MULTIPLE_FEATURE_DIM;
% X_full = train_features_full;
% [N MULTIPLE_FEATURE_DIM] = size(X_full);

idFea = 1;
Fea{idFea} = train_samples.denseHUE; Fea_Alpha{idFea} = train_samples.alpha_denseHUE; idFea=idFea+1;
Fea{idFea} = train_samples.denseSIFT; Fea_Alpha{idFea} = train_samples.alpha_denseSIFT; idFea=idFea+1;
Fea{idFea} = train_samples.GIST; Fea_Alpha{idFea} = train_samples.alpha_GIST;
Fea_Type = Global.Feature_Type;


%train mahalabios matrix for each label
load(fullfile(RUN_DIR, Global.Train_Feature_Dir, 'seman_group_subset_corel5k.mat'));
L = length(seman_group_subset.label_img_index);

MODEL_LMNN_DIR = 'D:\workspace-limu\cloud disk\Dropbox\limu\lmnn_model';


%% now learn metric for each label
for i = 1:L
    fprintf('learn metric for label %d\n',i);
    %load sample pairs in i-th label directory
    label_dir = sprintf('label_%d', i);
     
    if exist(fullfile(LABEL_PAIRS_DIR, 'train', label_dir),'dir')
        sample_index = seman_group_subset.label_img_index{i};
        [inxa, inxb, matches] = ...
            merge_sample_pairs_in_label(LABEL_PAIRS_DIR, label_dir, sample_index);
    else
        label_model(i).W = [];
        label_model(i).V = [];
        label_model(i).t = 0;
        continue;
    end
       
    % learn metric for i-th label
    %initial w and v for each feature
    for j = 1 : 3
        weights.w(j) = 1;
        weights.v{j} = ones(1, Global.Feature_Dim(j)) / Global.Feature_Dim(j); 
    end
    
    label_model(i) = learn_metric_pegasos(Fea, Fea_Alpha, inxa', inxb', matches', weights, Fea_Type);
    
    % save learned parameters for each label in the public dropbox dir
%     model_dir = 'label_basedLMNN';
%     if ~exist(fullfile(MODEL_LMNN_DIR, model_dir), 'dir')
%         mkdir(fullfile(MODEL_LMNN_DIR, model_dir));
%     end
%     label_i_model = sprintf('label_%d_W.mat',i);
%     save(fullfile(MODEL_LMNN_DIR, model_dir, label_i_model), 'best_W', 'ellipse_time');
    
    if(mod(i,10) == 0)
        fprintf(sprintf('caculate %d-th label model finished!\n',i));
    end
end

%% set model name, save in local dir
model_dir = 'label_basedLMNN_pegasos';
learned_model_name = 'label_basedLMNN_pegasos.mat';
    
if ~exist(fullfile(MODEL_DIR, model_dir), 'dir')
    mkdir(fullfile(MODEL_DIR, model_dir));
end

save(fullfile(MODEL_DIR, model_dir, learned_model_name), 'label_model');
fprintf('\nfinshed learn multiple metrics and save metrics! \n');
end




%% this function is to merge all the sample pairs in one label directory
function [ixda, ixdb, matches] = merge_sample_pairs_in_label(label_pairs_dir, label_dir, sample_index)
    pairs_all = [];
    overlap_a = []; %test img id
    overlap_b = []; %overlap img id
    overlap_match = [];
    
    N_samples = length(sample_index);
%     N_preserved = 10;
    balance_factor = 1;
        
    for i = 1 : N_samples
        %load sample pairs for i-th sample
        try
            label_mat_name = sprintf('sample_%d.mat', sample_index(i));
            load(fullfile(label_pairs_dir, 'train', label_dir, label_mat_name)); 

            index_sim = find([label_pairs.match] == 1);
            sample_id = label_pairs(1).img1.id;
            
            %here consider the overlapped samples to be positve samples
            num_overlap = length(label_pairs_overlap);
            overlap_a = [overlap_a, ones(1, num_overlap) * sample_id];
            overlap_b = [overlap_b, label_pairs_overlap'];
            overlap_match = [overlap_match, logical(label_pairs_overlap')];
            
            index_dissim = find([label_pairs.match] == 0);

            preserved_index = [index_sim, index_dissim(1: (length(index_sim)+num_overlap)*balance_factor)];
            temp = label_pairs(preserved_index);

            if isempty(pairs_all)
                pairs_all = temp;
            else
                pairs_all = [pairs_all, temp];
            end    
        catch
            lasterr
        end
    end
    
    temp_a = [pairs_all.img1];
    temp_b = [pairs_all.img2];     
    ixda = [temp_a.id, overlap_a];
    ixdb = [temp_b.id, overlap_b];
    matches = [logical([pairs_all.match]), overlap_match];
        
end



%% function learn_metric_pegasos learns metric in distance space
function metric = learn_metric_pegasos(Features, Alpha, I_q, I_pn, matches, weights, type)
%   Features is cell of multiple features
%   I_q is query sample image index
%   I_pn is targets / imposters indexes of each query image
%   matches is a logical vector with 1 target 0 imposter
%   weights contains inter weights w, intra weights v.

if(size(matches,1)~= size(I_q,1) || size(matches,1)~= size(I_q,1) || size(I_pn,1)~= size(I_q,1))
    fprintf('\nError: Number of elements in X and Y must same\nSee pegasos usage for further help\n');
    lasterr;
end

% inital some parameters
maxIter = 300;
lambda = 1; k = ceil(0.5*size(matches,1));
Tolerance=10^-4;

%% optimize inter feature w and v alternatively
fprintf('first optimize inter feature weights w: \n');
w(1,:) = weights.w';
v(1,:) = cell2mat(weights.v);
tstart = tic;

for t = 1 : maxIter
    % rand indexes for At
    idx = randi([1, size(I_pn, 1)], k, 1);
    % compute distance where D(q, n) - D(q, p) < 1
    At_q = I_q(idx);
    idx_q_uni = unique(At_q);
    idx_pn = I_pn(idx);
    matches_pn = matches(idx);
    % formulate idx_p / idx_n
    idx_q = [];
    idx_p = [];
    idx_n = [];
    
    idx_q_u = [];
    idx_p_u = [];
    
    for s = 1 : length(idx_q_uni);
        %get the q, p, n indexes for each sample
        iq = idx_q_uni(s);
        idx_iq = find(At_q == iq);        
        
        ip = idx_pn(idx_iq(matches_pn(At_q == iq) == 1));
        in = idx_pn(idx_iq(matches_pn(At_q == iq) == 0));
        %first part p->q
        idx_p_u = [idx_p_u; ip];
        idx_q_u = [idx_q_u;  kron(iq, ones(size(ip)))];
        %second part p->q, n
        idx_q = [idx_q; kron(iq, ones(length(ip)*length(in), 1))];
        idx_p = [idx_p ; kron(ip, ones(size(in)))];
        idx_n = [idx_n; kron(in, ones(size(ip)))];
    end
    
    %first part p->q
    for j = 1 : length(weights.w)
        [Dist_qp_u{j}, Vect_qp_u{j}] = Calculate_pairwise_distance(Features{j}(idx_q_u,:), Features{j}(idx_p_u,:),Alpha{j},weights.v{j},type{j});
        [Dist_qp{j}, Vect_qp{j}] = Calculate_pairwise_distance(Features{j}(idx_q,:), Features{j}(idx_p,:),Alpha{j},weights.v{j},type{j});
        [Dist_qn{j}, Vect_qn{j}] = Calculate_pairwise_distance(Features{j}(idx_q,:), Features{j}(idx_n,:),Alpha{j},weights.v{j},type{j});
    end
    % calulate hinge loss index
    D_qp = cell2mat(Dist_qp) * w(t,:)';
    D_qn = cell2mat(Dist_qn) * w(t,:)';
    idx1 = (D_qn - D_qp) < 0;
    
    eta_t_w = 1e-2 / (lambda * t);
    m1 = length(idx_q_u);
    m2 = length(find(idx1 == 1));
    %updata w
    for j = 1 : length(weights.w)
%         w1(j) = weights.w(j) - (eta_t/Alpha{j})*sum(Dist_qp_u{j},1) - (eta_t/Alpha{j})*sum(Dist_qp{j}(idx1,:) - Dist_qn{j}(idx1,:), 1);
          w1(j) = weights.w(j) - (eta_t_w)*sum(Dist_qp_u{j},1) - (eta_t_w)*sum(Dist_qp{j}(idx1,:) - Dist_qn{j}(idx1,:), 1);
    end
    w1(w1<=0) = 0;
    w(t+1,:) = normalize(w1, 2)*3;
    norm_value_w = norm(w(t+1,:)-w(t,:));
    
    % update v
    wT = w(end,:);
    weights.w = wT';
    % calulate hinge loss index
    D_qp = cell2mat(Dist_qp) * wT';
    D_qn = cell2mat(Dist_qn) * wT';
    idx1 = (D_qn - D_qp) < 0;
    
    eta_t_v = 1e-3 / (lambda * t);
    m1 = length(idx_q_u);
    m2 = length(find(idx1 == 1));
    %updata w
    for j = 1 : length(weights.w)
        v1{j} = weights.v{j} - wT(j)*(eta_t_v)*sum(Vect_qp_u{j},1) - wT(j)*(eta_t_v)*sum(Vect_qp{j}(idx1,:) - Vect_qn{j}(idx1,:), 1);
        v1{j}(v1{j} <= 0) = 0;
        v1{j} = normalize(v1{j},2);
    end
    weights.v = v1;
    v(t+1,:) = cell2mat(v1);
    
    
    norm_value_v = norm(v(t+1,:)-v(t,:)); 
    if(norm_value_w < Tolerance && norm_value_v < Tolerance)
        fprintf('   calculate w and v converged in %d iteration! \n', t);
        break;
    end
    if(mod(t, 100) == 0)
        fprintf('\n  iteration # %d/%d, norm value w: %f, v: %f \n',t,maxIter, norm_value_w, norm_value_v);
        fprintf('    constrains: %d in total %d pairs. \n', m2, length(idx1));
    end
    
end


wT = w(end,:)
vT = weights.v;
if(t<maxIter) 
    fprintf('v converged in %d iterations. \n',t);
else
    fprintf('v not converged in %d iterations. \n',maxIter);
end

%% save learned w and v
metric.W = wT;
metric.V = vT;
metric.t = toc(tstart);

end


function [dist,vect] = Calculate_pairwise_distance(hist1, hist2, alpha, weight, type)
% return gradient value of w or v, depending on flag 0 / 1
 if(strcmp(type, 'hue') || strcmp(type, 'sift')) %chi
%      [dist,vect] = GetchiSquareDist_Vect(hist1, hist2, weight);
     [dist,vect] = GetL1Dist_Vect(hist1, hist2, alpha, weight);
 elseif(strcmp(type, 'gist'))
     [dist,vect] = GetL2Dist_Vect(hist1, hist2, alpha, weight);
 else
     display('error: error feature type input! \n');
 end
end


function [dist,vect] = GetL1Dist_Vect(hist1, hist2, alpha, weights)
% hist1 and hist2 a nxk matrix, weights is a 1xk vector 
% return: dist nx1, vect nxk
vect = bsxfun(@times, abs(hist1 - hist2), weights);

dist = sum(vect, 2) / alpha.alpha_sum;
vect = bsxfun(@rdivide, vect, alpha.alpha_v);

end

function [dist,vect] = GetL2Dist_Vect(hist1, hist2, alpha, weights)
% hist1 and hist2 are nxk matrix, weights is a 1xk vector 
vect = abs(hist1 - hist2);
vect = vect.^2;
vect = bsxfun(@times, vect, weights);
% normalize, assume that each feature contributes equally

dist = sum(vect, 2) / alpha.alpha_sum;
vect = bsxfun(@rdivide, vect, alpha.alpha_v);
end

function [dist,vect] = GetchiSquareDist_Vect(his1,his2, weights)
% hist1 and hist2 are nxk matrix
temp_matrix = ((his1==0).*(his2==0));
his1_all = his1 + temp_matrix;
his2_all = his2 + temp_matrix;
vect = (his1_all-his2_all).^2./(his1_all+his2_all);
vect = bsxfun(@times, vect, weights);

dist = sum(vect, 2);
end