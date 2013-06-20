%This function is to find semantic neighbors (targets and imposters) for
%   each traning sample based on its multi labels

function do_generate_label_based_pairs_test(config_file)
    % The procedures for the semantic neighbors finding are following steps:
    %   1, first formulate semantic clusters for each label prioriely
    %   2, for one training sample in label cluster i
    %       (1)first find N1 nearest neighbors in label cluster i
    %       (2)then select N2 nearest neighbors in the other lable cluster
    %               j ~= i;
    %       (3)record these similar / dissimilar pairs


    eval(config_file);

    global Global;
    
    %get image annotation matrix  mat_img_anno DxW
    %   D is test samples, W is word number
    img_anno_fname = 'corel5k_test_annot.hvecs';
    mat_img_anno = vec_read(fullfile(IMAGE_ANNOTATION_DIR,img_anno_fname));

    [D, W] = size(mat_img_anno);

    %load train subset semantic group
    load(fullfile(RUN_DIR, Global.Train_Feature_Dir, 'seman_group_subset_corel5k.mat'));
   
    %load train / test feature mat
    load(fullfile(RUN_DIR, Global.Train_Feature_Dir, 'train_multifeature_corel5k.mat'));
    load(fullfile(RUN_DIR, Global.Test_Dir, 'test_multifeature_corel5k.mat'));
    
    N1 = 4;
    
    tstart = tic;
    
    %select semantic neighbors for each test image
    if strcmp(Global.Metric, 'label-basedLRLINEAR')
            addpath(genpath('D:\workspace-limu\image annotation\iciap2013\KISSME\toolbox\lib\liblinear-1.92'));
            model_dir = 'label_basedLRLINEAR';
            learned_model_name = 'label_basedLRLINEAR.mat';
            load(fullfile(MODEL_DIR, model_dir, learned_model_name));
    elseif strcmp(Global.Metric, 'label_basedLMNN')
            model_dir = 'label_basedLMNN';
            learned_model_name = 'label_basedLMNN.mat';
            load(fullfile(MODEL_DIR, model_dir, learned_model_name));
    elseif strcmp(Global.Metric, 'label_basedLMNN_pegasos')
            model_dir = 'label_basedLMNN_pegasos';
            learned_model_name = 'label_basedLMNN_pegasos.mat';
            load(fullfile(MODEL_DIR, model_dir, learned_model_name));
    end
    
    for d = 1 : D
        %get d-th sample vector from test set
        sample_vector = extract_set_samples(d, test_samples);      
        test_sample_pairs{d} = [];
         
        if (strcmp(Global.Metric, 'label-basedLRLINEAR'))
        %select N1 inter simimilar pairs in each label
            [sim_index, dist_score] = select_all_neighbors_metric(sample_vector, train_samples, seman_group_subset, N1, [1:W], 0, label_model, 1);  
        elseif (strcmp(Global.Metric, 'label-basedLMNN'))
            [sim_index, dist_score] = select_all_neighbors_metric(sample_vector, train_samples, seman_group_subset, N1, [1:W], 0, label_model, 2); 
        elseif (strcmp(Global.Metric, 'label_basedLMNN_pegasos'))
            [sim_index, dist_score] = select_all_neighbors_metric(sample_vector, train_samples, seman_group_subset, N1, [1:W], 0, label_model, 3); 
        elseif strcmp(Global.Metric, 'base')
            [sim_index, dist_score] = select_all_neighbors(sample_vector, train_samples, seman_group_subset, N1, [1:W], 0); 
        else
            lasterr
            return;
        end
        %formulate test pairs in each label, 
        %here d is in test set, sim_index is in train set 
        test_sample_pairs{d} = generate_label_pairs(test_sample_pairs{d}, d, sim_index);                
        test_sample_dist_score{d} = dist_score;
        
        if (mod(d,50) == 0)
            fprintf(sprintf('selecting pairs for test image %d  finished!\n', d));
        end
    end
    
    toc(tstart);
    fprintf('finished generate semantic neighborhood for test samples, taking time %f \n', toc(tstart));
    %% save test sample pairs
    if ~exist(fullfile(RUN_DIR, Global.Test_Dir), 'dir')
        mkdir(fullfile(RUN_DIR, Global.Test_Dir));
    end
     
    save(fullfile(RUN_DIR, Global.Test_Dir, 'corel5k_test_pairs.mat'), 'test_sample_pairs' , 'test_sample_dist_score');
    fprintf('finshed save neighbor informations for test samples! \n');
end

%%
function sim_index = select_intra_neighbors(sample_vector, set_vector, semantic_group, N1, label_index)
    % this function is for step 2.1
    distance_score = base_distance(sample_vector, set_vector); % 1xD1 vector
    
    %sort to select top-N1 minimal values
    [v, v_index]= sort(distance_score, 2, 'ascend');

    %remove itself
    sim_index = semantic_group.label_img_index{label_index}(v_index(2:(N1+1)));

end

%% this function is to select N1 neighbors in each label_index, and preserved final N2 neighbors
function [sim_index, dist_score] = select_all_neighbors(sample_vector, train_samples, semantic_group, N1, label_index, N2)
    distance_intra_label_index = [];
    distance_intra_label_score = [];
    sim_index = [];
    
    for w = label_index %label_index is sampel image's labels
        %all samples in label w cluster
        w_index = semantic_group.label_img_index{w};
        
        if(length(w_index) > N1)
            %compare distance in w-th label cluster
            set_vector = extract_set_samples(w_index, train_samples);
            distance_score = base_distance(sample_vector, set_vector);

            [v, v_index]= sort(distance_score, 2, 'ascend');
            if v(1) == 0 % preserve from 2-N1+1
                v_index_N1 = v_index(2:N1+1);
                temp_index = w_index(v_index_N1);
                temp_score = v(2:N1+1);
            else
                v_index_N1 = v_index(1:N1);
                temp_index = w_index(v_index_N1);
                temp_score = v(1:N1);
            end

            if(isempty(distance_intra_label_index))
                distance_intra_label_index = temp_index;
                distance_intra_label_score = temp_score;
            else
                distance_intra_label_index = [distance_intra_label_index; temp_index];
                distance_intra_label_score = [distance_intra_label_score, temp_score];
            end        
        end    
    end
    
    if isempty(distance_intra_label_index)
        sim_index = [];
        return;
    end
    
    % return similar index and remove repeat index
    % ensure that the sim_index is sorted by distance ascend
    [unique_value, unique_index] = unique(distance_intra_label_index);
    [v_u, v_u_index] = sort(distance_intra_label_score(unique_index), 2, 'ascend');
    
    unique_temp = distance_intra_label_index(unique_index);
    unique_distance_temp = distance_intra_label_score(unique_index);
    
    if N2 == 0 %preserve all
        sim_index = unique_temp(v_u_index);
        dist_score = unique_distance_temp(v_u_index);
    else
        sim_index = unique_temp(v_u_index(1:N2));
        dist_score = unique_distance_temp(v_u_index(1:N2));
    end
end


%% this function is to select N1 neighbors in each label_index, and preserved final N2 neighbors
function [sim_index, dist_score] = select_all_neighbors_metric(sample_vector, train_samples, semantic_group, N1, label_index, N2, metric_models, metric_type)
    global Global;
    
    distance_intra_label_index = [];
    distance_intra_label_score = [];
    sim_index = [];
    
    for w = label_index %label_index is sampel image's labels
        %all samples in label w cluster
        w_index = semantic_group.label_img_index{w};
        
        if(length(w_index) > N1)
            %compare distance in w-th label cluster
            set_vector = extract_set_samples(w_index, train_samples);
            
            
            if(metric_type == 1)
                distance_vectors = base_distance_vector(sample_vector, set_vector);
                [predict_labels, accuracy, dec_value] = predict(ones(length(w_index),1), sparse(distance_vectors), metric_models(w).M, '-q');
                %sort the neighbors according to ascend order
                [v, v_index]= sort(dec_value, 1, 'descend');
            elseif(metric_type == 2)
                distance_vectors = base_distance_vector(sample_vector, set_vector);
                dec_value = sparse(distance_vectors) * metric_models(w).M';
                dec_value(dec_value < 0) = Inf;
                [v, v_index]= sort(dec_value, 1, 'ascend');
            elseif(metric_type == 3) %for mmlmnn pegasos
                    idFea = 1;
                    Fea_Alpha{idFea} = train_samples.alpha_denseHUE; idFea=idFea+1;
                    Fea_Alpha{idFea} = train_samples.alpha_denseSIFT; idFea=idFea+1;
                    Fea_Alpha{idFea} = train_samples.alpha_GIST;
                    Fea_Type = Global.Feature_Type;
    
                dec_value = CalculateMultipleDistance(sample_vector, set_vector,Fea_Alpha,...
                        metric_models(w).W,metric_models(w).V, Fea_Type);
                [v, v_index]= sort(dec_value, 1, 'descend');
            end
            %here use test sample, so no zero distance.
%             if v(1) == 0 % preserve from 2-N1+1
%                 v_index_N1 = v_index(2:N1+1);
%                 temp_index = w_index(v_index_N1);
%                 temp_score = v(2:N1+1);
%             else
                v_index_N1 = v_index(1:N1);
                temp_index = w_index(v_index_N1);
                temp_score = v(1:N1);
%             end

            if(isempty(distance_intra_label_index))
                distance_intra_label_index = temp_index;
                distance_intra_label_score = temp_score;
            else
                distance_intra_label_index = [distance_intra_label_index; temp_index];
                distance_intra_label_score = [distance_intra_label_score; temp_score];
            end        
        end    
    end
    
    if isempty(distance_intra_label_index)
        sim_index = [];
        return;
    end
    
    % return similar index and remove repeat index
    % ensure that the sim_index is sorted by distance ascend
    [unique_value, unique_index] = unique(distance_intra_label_index);
    if(metric_type == 1)
        [v_u, v_u_index] = sort(distance_intra_label_score(unique_index), 1, 'descend');
    elseif (metric_type == 2 || metric_type == 3)
        [v_u, v_u_index] = sort(distance_intra_label_score(unique_index), 1, 'ascend');
    end
    unique_temp = distance_intra_label_index(unique_index);
    unique_distance_temp = distance_intra_label_score(unique_index);
    
    if N2 == 0 %preserve all
        sim_index = unique_temp(v_u_index);
        dist_score = unique_distance_temp(v_u_index);
    else
        sim_index = unique_temp(v_u_index(1:N2));
        dist_score = unique_distance_temp(v_u_index(1:N2));
    end
end

%%
function set_samples = extract_set_samples(sample_index, train_samples)
%     set_samples.denseHUE = train_samples.denseHUE(sample_index,:);
%     set_samples.denseSIFT = train_samples.denseSIFT(sample_index,:);
%     set_samples.GIST = train_samples.GIST(sample_index,:);
%     set_samples.HSV = train_samples.HSV(sample_index,:);
%     set_samples.LAB = train_samples.LAB(sample_index,:);
%     set_samples.RGB = train_samples.RGB(sample_index,:);

    set_samples.denseHUE = train_samples.denseHUE(sample_index,:);
    set_samples.alpha_denseHUE = train_samples.alpha_denseHUE;
    
    set_samples.denseSIFT = train_samples.denseSIFT(sample_index,:);
    set_samples.alpha_denseSIFT = train_samples.alpha_denseSIFT;
    
    set_samples.GIST = train_samples.GIST(sample_index,:);
    set_samples.alpha_GIST = train_samples.alpha_GIST;
end

%%
function pairs = generate_label_pairs(pairs, img_index, sim_pairs)
    n_sim = length(sim_pairs);
  
    if(isempty(pairs))
        for i = 1 : n_sim
            pairs(i).img1.id = img_index;
            pairs(i).img2.id = sim_pairs(i);

            pairs(i).match = 1;
            pairs(i).training = 0;
        end
    
    else
        n_count = length(pairs);

        for i = (n_count+1) : (n_count+n_sim)
            ind = i - n_count;
            
            pairs(i).img1.id = img_index;
            pairs(i).img2.id = sim_pairs(ind);

            pairs(i).match = 1;
            pairs(i).training = 0;
        end

    end
        
end

function dist_value = CalculateMultipleDistance(test_vector, neigh_vector, alpha, w, v,type)
% return nx1 dist value: test to neighbors    
if(length(w) ~= length(v))
        fprintf('error: w and v should be identical dimension! \n');
        return;
    end
    
    %sum each dist
    for i = 1 : length(w)
        if(strcmp(type{i}, 'hue')) %L1
            dist(:,i) = GetL1Dist(test_vector.denseHUE, neigh_vector.denseHUE, alpha{i},v{i});
        elseif(strcmp(type{i}, 'sift'))
            dist(:,i) = GetL1Dist(test_vector.denseSIFT, neigh_vector.denseSIFT, alpha{i},v{i});
        elseif(strcmp(type{i}, 'gist'))
            dist(:,i) = GetL2Dist(test_vector.GIST, neigh_vector.GIST, alpha{i},v{i});
        else
            display('error: error feature type input! \n');
        end
    end
    
    dist_value = dist * w'; %nx1 dist value
end

function dist = GetL1Dist(hist1, hist2, alpha,weights)
% hist1 and hist2 a nxk matrix, weights is a 1xk vector 
% return: dist nx1, vect nxk
hist1_vec = repmat(hist1, size(hist2,1),1);
vect = bsxfun(@times, abs(hist1_vec - hist2), weights);

% dist = sum(vect, 2) / alpha.alpha_sum;
dist = sum(vect, 2);


end

function [dist] = GetL2Dist(hist1, hist2, alpha,weights)
% hist1 and hist2 are nxk matrix, weights is a 1xk vector 
hist1_vec = repmat(hist1, size(hist2,1),1);
vect = abs(hist1_vec - hist2);
vect = vect.^2;
vect = bsxfun(@times, vect, weights);
% normalize, assume that each feature contributes equally
dist = sum(vect, 2);

end