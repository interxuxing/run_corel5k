function do_learn_label_based_distance_metric(config_file)
%% Fuction that learn different types of model accorind to target / imposter
%%  pairs selected from previous steps


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

%% different learn method 
%% 1, label-based, one M for each label;  2, global-based one M for all labels
%% 3, label-based, LR+schmidt; 4, label-based , LR+liblinear
%% 5, global-based one W for all labels, LR+liblinear
learn_method = Global.Learn_method; 

model_dir = [];
learned_model_name = [];

if(learn_method == 1) % label-based, one M for each label;
    for i = 1:L
        %load pairs for this label i
        label_mat_name = sprintf('label_%d_pairs.mat', i);
        load(fullfile(LABEL_PAIRS_DIR,'train',label_mat_name));
        
        pairs_i = label_pairs;
        [pairs_i_s, inxa, inxb] = cell_2_struct(pairs_i);

        if(isempty(pairs_i_s))
            label_model(i).M = diag(ones(1,P));
            label_model(i).t = 0;
            continue;
        end

%         label_pairs.inxa = inxa;
%         label_pairs.inxb = inxb;

        matches = logical([pairs_i_s.match]);
        un = unique([pairs_i_s.fold]);

        for c = un
            trainMask = ([pairs_i_s.fold] == c);

            %Train%
            label_model(i) = learn_label_Maha(X, inxa(trainMask), inxb(trainMask),matches);
        end

        if(mod(i,20) == 0)
            fprintf(sprintf('caculate %d-th labels finished!\n',i));
        end
    end
    %set model name
    model_dir = 'label_basedM';
    learned_model_name = 'label_basedM_model.mat';
    
elseif(learn_method == 2) % global-based one M for all labels
    inxa = []; inxb = [];
    matches = []; trainMask = [];
    %merge sim / dissim label pairs in all labels to one matrix
     for i = 1:L
        %load pairs for this label i
        label_mat_name = sprintf('label_%d_pairs.mat', i);
        load(fullfile(LABEL_PAIRS_DIR,'train',label_mat_name));
        
        pairs_i = label_pairs.pairs;
        [pairs_i_s, tempa, tempb] = cell_2_struct(pairs_i);
        
        if(isempty(pairs_i_s))
            continue;
        end
        
        inxa = [inxa, tempa];
        inxb = [inxb, tempb];
        
        matches = logical([matches, logical([pairs_i_s.match])]);
        trainMask = logical([trainMask, ([pairs_i_s.fold] == 1)]);
     end
     
     %now train a global M matrix
     label_model = learn_label_Maha(X, inxa(trainMask), inxb(trainMask),matches);
     
     %set model name
    model_dir = 'global_basedM';
    learned_model_name = 'global_basedM_model.mat';
     
elseif(learn_method == 3) % label-based, LR; use schmidt lib to train L1+LR
    for i = 1:L
        %load pairs for this label i
        label_mat_name = sprintf('label_%d_pairs.mat', i);
        load(fullfile(LABEL_PAIRS_DIR,'train',label_mat_name));
        
        pairs_i = label_pairs.pairs;
        [pairs_i_s, inxa, inxb] = cell_2_struct(pairs_i);

        if(isempty(pairs_i_s))
            label_model(i).M = [];
            label_model(i).t = 0;
            continue;
        end

        matches = logical([pairs_i_s.match]);
        un = unique([pairs_i_s.fold]);

        for c = un
            trainMask = ([pairs_i_s.fold] == c);

            %Train%
            label_model(i) = learn_label_LRL1(train_samples, inxa(trainMask), inxb(trainMask),matches);
        end

        if(mod(i,20) == 0)
            fprintf(sprintf('caculate %d-th labels finished!\n',i));
        end
    end
    
    %set model name
    model_dir = 'label_basedLRL1SM';
    learned_model_name = 'label_basedLRL1SM.mat';
    
elseif(learn_method == 4) % label-based one M for all labels, LR
    %add liblinear path
    addpath(genpath('D:\workspace-limu\image annotation\iciap2013\KISSME\toolbox\lib\liblinear-1.92'));
    
    %make label_based model dir, to store one mat file for one label
%     sub_model_dir = 'label_based';
%     if ~exist(fullfile(MODEL_DIR, sub_model_dir), 'dir')
%         mkdir(fullfile(MODEL_DIR, sub_model_dir));
%     end
     
    for i = 1:L
        fprintf('label %d\n',i);
        %load sample pairs in i-th label directory
        label_dir = sprintf('label_%d', i);
        
        if exist(fullfile(LABEL_PAIRS_DIR, 'train', label_dir),'dir')
            sample_index = seman_group_subset.label_img_index{i};
            [inxa, inxb, matches] = merge_sample_pairs_in_label(LABEL_PAIRS_DIR, label_dir, sample_index);
        else
            label_model(i).M = [];
            label_model(i).t = 0;
            continue;
        end

        %Train%
        label_model(i) = learn_label_LRL1_LIBLINEAR(train_samples, inxa, inxb, matches);

%         label_model_name = sprintf('label_d%_model.mat',i);
%         save(fullfile(MODEL_DIR, sub_model_dir, label_model_name), 'label_model');
        
        if(mod(i,20) == 0)
            fprintf(sprintf('caculate %d-th label model finished!\n',i));
        end
    end
    
    %set model name
    model_dir = 'label_basedLRLINEAR';
    learned_model_name = 'label_basedLRLINEAR.mat';
   
elseif(learn_method == 5) %global-based, liblinear  
    %add liblinear path
    addpath(genpath('D:\workspace-limu\image annotation\iciap2013\KISSME\toolbox\lib\liblinear-1.92\matlab'));
    
    inxa = []; inxb = [];
    matches = []; trainMask = [];
    %merge sim / dissim label pairs in all labels to one matrix
     for i = 1:L
        %load pairs for this label i
        label_mat_name = sprintf('label_%d_pairs.mat', i);
        load(fullfile(LABEL_PAIRS_DIR,'train',label_mat_name));
         
        pairs_i = label_pairs.pairs;
        [pairs_i_s, tempa, tempb] = cell_2_struct(pairs_i);
        
        if(isempty(pairs_i_s))
            continue;
        end
        
        inxa = [inxa, tempa];
        inxb = [inxb, tempb];
        
        matches = logical([matches, logical([pairs_i_s.match])]);
        trainMask = logical([trainMask, ([pairs_i_s.fold] == 1)]);
     end
     
    %train weight parameters
    label_model = learn_label_LRL1_LIBLINEAR_global(train_samples, inxa(trainMask), inxb(trainMask),matches);
    
end

%%save learned model mat
if ~exist(fullfile(MODEL_DIR, model_dir), 'dir')
    mkdir(fullfile(MODEL_DIR, model_dir));
end

save(fullfile(MODEL_DIR, model_dir, learned_model_name), 'label_model');

end


function s = learn_label_Maha(X, idxa, idxb, matches)
    % s = learn_label_Maha(obj,X,idxa,idxb,matches)
            % Learn from pairwise equivalence labels
            %
            % parameters:  
            %   X         - input matrix, each column is an input vector 
            %   [DxN*2]. N is the number of pairs. D is the feature 
            %   dimensionality
            %   idxa      - index of image A in X [1xN]
            %   idxb      - index of image B in X [1xN]
            %   matches   - matches defines if a pair is similar (1) or 
            %   dissimilar (0)
            %
            % return:
            %   s         - Result data struct
            %   s.M       - Trained quadratic distance metric
            %   s.t       - Training time in seconds
            %   s.p       - Used parameters, see LearnAlgoKISSME properties for details.s
            %   s.learnAlgo - class handle to obj
            %   s.roccolor  - line color for ROC curve, default 'g'       
            tic;
            % Eqn. (12) - sum of outer products of pairwise differences (similar pairs)
            % normalized by the number of similar pairs.
            % Here we have to average(normalize) the SOPD for all sim / dissim pairs
            covMatches    = SOPD(X,idxa(matches),idxb(matches)) / sum(matches);
            % Eqn. (13) - sum of outer products of pairwise differences (dissimilar pairs)
            % normalized by the number of dissimilar pairs.
            covNonMatches = SOPD(X,idxa(~matches),idxb(~matches)) / sum(~matches);
            t = toc;
            
            tic;
            % Eqn. (15-16)
            lambda = 1; pmetric = 1;
            
            s.M = pseudoinverse(covMatches)*eye(length(covMatches)) - lambda * pseudoinverse(covNonMatches)*eye(length(covNonMatches));   
            if pmetric
                % to induce a valid pseudo metric we enforce that  M is p.s.d.
                % by clipping the spectrum
                s.M = validateCovMatrix(s.M);
            end
            s.t = toc + t;                 
end


function s = learn_label_LRL1(train_samples, idxa, idxb, matches)
%train_samples is a struct of all feature types

%formulate the orginal distance matirx X
D = length(idxa);
step = floor(D / 10);

parfor d = 1 : D
    Xa = extract_set_samples(idxa(d),train_samples);
    Xb = extract_set_samples(idxb(d),train_samples);
    
    %calculate basis distance
    X(d,:) = base_distance_vector(Xa,Xb);
    
    if(mod(d,step) == 0)
        fprintf('formulate %d-th pairs distance finished!\n',d);
    end
end

%use LR+L1 to train weight parameters
[D,N] = size(X);
X_bias = [ones(D,1) X]; %D+1
Y = ones(D,1);
Y(~matches) = -1;
sparsityFactor = .2;
nVars = N+1;
w = zeros(nVars,1);
w_init = zeros(nVars,1);
% w = randn(nVars,1).*(rand(nVars,1) < sparsityFactor);

w(1) =  log(sum(Y==1)/sum(Y==-1));

[f,g] = LogisticLoss(w,X_bias,Y);
lambdaMax = max(abs(g));

lambdaValues = lambdaMax*[1:-.1:0];
nLambdaValues = length(lambdaValues);

W = zeros(nVars,nLambdaValues);
lambdaVect = [0;ones(nVars-1,1)]; % Pattern of the lambda vector

tic;
for i = 1:nLambdaValues
    lambda = lambdaValues(i);
    fprintf('lambda = %f\n',lambda);
    
    % Compute free variables (uses w and g from above)
    free = lambdaVect == 0 | w ~= 0 | (w == 0 & (abs(g) > lambda));
        
    while 1
        % Solve with respect to free variables
        funObj = @(w)LogisticLoss(w,X_bias(:,free),Y);
        w(free) = L1General2_PSSgb(funObj,w(free),lambda*lambdaVect(free));
        
        % Compute new set of free variables
        [f,g] = LogisticLoss(w,X_bias,Y);
        free_new = lambdaVect == 0 | w ~= 0 | (w == 0 & (abs(g) > lambda));
        
        if any(free_new == 1 & free == 0)
            % We added a new free variable, re-optimize
            free = free_new;
        else
            % Optimal solution found
            break;
        end
    end
    
    % Record solution
    W(:,i) = w;
end

s.M = W(:,end);
s.t = toc;

end

function s = learn_label_LRL1_LIBLINEAR(train_samples, idxa, idxb, matches)
%train_samples is a struct of all feature types
%formulate the orginal distance matirx X
D = length(idxa);
STEP_SIZE = 10;
step = floor(D / STEP_SIZE);

parfor d = 1 : D
    Xa = extract_set_samples(idxa(d),train_samples);
    Xb = extract_set_samples(idxb(d),train_samples);
    
    %calculate basis distance
    X(d,:) = base_distance_vector(Xa,Xb);
    
    if(mod(d,step) == 0)
        fprintf('formulate %d-th pairs distance finished!\n',d);
    end
end

%use liblinear L1+LR to trian weight for each label
Y = ones(D,1);
Y(~matches) = -1;

liblinear_opt = '-s 2 -B 1 -c 10 -e 0.0001';

tic;
X_sparse = sparse(X);
s.M = train(Y,X_sparse,liblinear_opt);
s.t = toc;

predict(Y, X_sparse, s.M);
end

function s = learn_label_LRL1_LIBLINEAR_global(train_samples, idxa, idxb, matches)
%train_samples is a struct of all feature types
%formulate the orginal distance matirx X
D = length(idxa);
STEP_SIZE = 20;
step = floor(D / STEP_SIZE);


%save each step data
for i = 1 : STEP_SIZE
    for d = ((i-1)*step+1) :  i*step
        Xa = extract_set_samples(idxa(d),train_samples);
        Xb = extract_set_samples(idxb(d),train_samples);

        %calculate basis distance
        base_index = (i-1)*step;
        X(d-base_index,:) = base_distance_vector(Xa,Xb);
        
        if(mod(d-base_index, 500) == 0)
            fprintf('X distance for %d-th sample-pair finished!\n', d);
        end
    end
    %save this clip file
    step_file_name = sprintf('clip_step%d.mat',i);
    X = sparse(X);
    save(fullfile(TEMP_DATA_DIR, step_file_name), 'X');
    clear X;
end
% 
% %for residual step
for d = (STEP_SIZE*step+1) :  D
    Xa = extract_set_samples(idxa(d),train_samples);
    Xb = extract_set_samples(idxb(d),train_samples);

    %calculate basis distance
    base_index = STEP_SIZE*step;
    X(d-base_index,:) = base_distance_vector(Xa,Xb);
end
    %save this clip file
    step_file_name = sprintf('clip_step%d.mat',STEP_SIZE+1);
    X = sparse(X);
    save(fullfile(TEMP_DATA_DIR, step_file_name), 'X');
    clear X;

%load all the X data, and merge together
X_sparse = [];
for index = 1 : STEP_SIZE+1
    step_file_name = sprintf('clip_step%d.mat',index);
    load(fullfile(TEMP_DATA_DIR, step_file_name));
    
    if(isempty(X_sparse))
        X_sparse = X;
    else
        X_sparse = [X_sparse; X];
    end
end


%use liblinear L1+LR to train weight for each label
Y = ones(D,1);
Y(~matches) = -1;

liblinear_opt = '-c 4 -e 0.1';

tic;
% X_sparse = sparse(X);
s.M = train(Y,X_sparse,liblinear_opt);
s.t = toc;

end


function [Y, ixda,ixdb] = cell_2_struct(X)
    %input X is X{i}.pairs.id
    %output Y is Y(i).id

    if(isempty(X))
         display('empty label pairs!');
         Y = [];
         ixda = [];
         ixdb = [];
    end
        C = length(X);


    for c = 1 : C
        Y(c).pairId = X{c}.pairId;
        Y(c).img1 = X{c}.img1;
        Y(c).img2 = X{c}.img2;
        Y(c).match = X{c}.match;
        Y(c).training = X{c}.training;
        Y(c).fold = X{c}.fold;

        ixda(c) = X{c}.img1.id;
        ixdb(c) = X{c}.img2.id;
    end
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


%% this function is to merge all the sample pairs in one label directory
function [ixda, ixdb, matches] = merge_sample_pairs_in_label(label_pairs_dir, label_dir, sample_index)
    pairs_all = [];
    overlap_a = []; %test img id
    overlap_b = []; %overlap img id
    overlap_match = [];
    
    N_samples = length(sample_index);
    N_preserved = 10;
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
            
            %formulate balanced 
%             if(length(index_sim) >= N_preserved)
%                 preserved_index = [index_sim(1:N_preserved), index_dissim(1: N_preserved*balance_factor)];
%             else
%                 preserved_index = [index_sim, index_dissim(1: length(index_sim)*balance_factor)];
%             end
%             preserved_index = [index_sim, index_dissim(1: (length(index_sim))*balance_factor)];
%             rand_dissim = randperm(length(index_dissim));

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