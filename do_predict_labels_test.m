%This function is to predict annotation score given the label-based metric
function do_predict_labels_test(config_file)

eval(config_file);

%load semantic neighbors for each test image
load(fullfile(RUN_DIR, Global.Test_Dir, 'corel5k_test_pairs.mat'));
D = length(test_sample_pairs);

%load train annotation matrix
train_anno = double(vec_read(fullfile(IMAGE_ANNOTATION_DIR, 'corel5k_train_annot.hvecs')));

[D_TRAIN, W] = size(train_anno);


anno_score = zeros(D, W);
anno_score_decvalue = zeros(D, W);

%learn method 1, label-based, one M for each label;  2, global-based one M
%for all labels
learn_method = Global.Learn_method;

if(learn_method == 1)
    %set model name
    model_dir = 'label_basedM';
    learned_model_name = 'label_basedM_model.mat';
    load(fullfile(MODEL_DIR, model_dir, learned_model_name), 'label_model');
    
    for d = 1 : D
        test_vector = test_features_full(d,:); %1xN

        for i = 1 : length(test_sample_pairs{d})
            %caculate distance between current test image and its i neighbor
            neighbor_index = test_sample_pairs{d}(i).img2.id;
            neighbor_vector = train_features_full(neighbor_index,:);

            %find anno in neighbor image
            anno_index = find(train_anno(neighbor_index,:) ~= 0);
            for j = 1 : length(anno_index)
                dist = cMdistance(label_model(anno_index(j)).M, test_vector', neighbor_vector');
                if(dist < 0)
                    display('Error: distance less than 0! ');
                end
                anno_score(d, anno_index(j)) = anno_score(d, anno_index(j)) + exp(-dist);
            end
        end

        if(mod(d,10) == 0)
            fprintf(sprintf('predict for %d-th images finished!\n',d));
        end
    end
    
elseif(learn_method == 2)
    %set model name
    model_dir = 'global_basedM';
    learned_model_name = 'global_basedM_model.mat';
    load(fullfile(MODEL_DIR, model_dir, learned_model_name), 'label_model');
    
    for d = 1 : D
        test_vector = test_features_full(d,:); %1xN

        for i = 1 : length(test_sample_pairs{d})
            %caculate distance between current test image and its i neighbor
            neighbor_index = test_sample_pairs{d}(i).img2.id;
            neighbor_vector = train_features_full(neighbor_index,:);

            %find anno in neighbor image
            anno_index = find(train_anno(neighbor_index,:) ~= 0);
            for j = 1 : length(anno_index)
                dist = cMdistance(label_model.M, test_vector', neighbor_vector');
                if(dist < 0)
                    display('Error: distance less than 0! ');
                end
                anno_score(d, anno_index(j)) = anno_score(d, anno_index(j)) + exp(-dist);
            end
        end

        if(mod(d,10) == 0)
            fprintf(sprintf('predict for %d-th images finished!\n',d));
        end
    end
    
elseif(learn_method == 3)   
    %set model name
    model_dir = 'label_basedLRL1SM';
    learned_model_name = 'label_basedLRL1SM.mat';
    load(fullfile(MODEL_DIR, model_dir, learned_model_name), 'label_model');
    
    %load original multi-features of train and test set
    load(fullfile(RUN_DIR, Global.Train_Feature_Dir, 'train_multifeature_corel5k.mat'));
    load(fullfile(RUN_DIR, Global.Test_Dir, 'test_multifeature_corel5k.mat'));
    
    for d = 1:D   
        %get d-th test vector
        test_vector = extract_index_samples(d, test_samples); %a struct
        
        for i = 1 : length(test_sample_pairs{d})
            %caculate distance between current test image and its i neighbor
            neighbor_index = test_sample_pairs{d}(i).img2.id;
            neighbor_vector = extract_index_samples(neighbor_index,train_samples);

            %find anno in neighbor image
            anno_index = find(train_anno(neighbor_index,:) ~= 0);
            for j = 1 : length(anno_index)
                %calculate similarity prob through basic distance
                base_dis = base_distance_vector(test_vector, neighbor_vector);
                base_dis = [1 base_dis];
                if(~isempty(label_model(anno_index(j)).M))
                    try
                        dist = base_dis * label_model(anno_index(j)).M;
                    catch
                        display('error');
                    end
                    anno_score(d, anno_index(j)) = anno_score(d, anno_index(j)) + 1 / (1 + exp(-dist));
                end
            end
        end
    end
    
elseif(learn_method == 4) 
    %add liblinear path
    addpath(genpath('D:\workspace-limu\image annotation\iciap2013\KISSME\toolbox\lib\liblinear-1.92\matlab'));
    
    %set model name
    model_dir = 'label_basedLRLINEAR';
    learned_model_name = 'label_basedLRLINEAR.mat';
    load(fullfile(MODEL_DIR, model_dir, learned_model_name));
    
    %load original multi-features of train and test set
    load(fullfile(RUN_DIR, Global.Train_Feature_Dir, 'train_multifeature_corel5k.mat'));
    load(fullfile(RUN_DIR, Global.Test_Dir, 'test_multifeature_corel5k.mat'));
    
    for d = 1:D   
        %get d-th test vector
        test_vector = extract_index_samples(d, test_samples); %a struct
        %select top nn neighbors to predict
        if(Global.Test_Neighbors == 0)
            nn = length(test_sample_pairs{d});
        else
            nn = Global.Test_Neighbors;
        end
        
        for i = 1 : nn
            %caculate distance between current test image and its i neighbor
            neighbor_index = test_sample_pairs{d}(i).img2.id;
            neighbor_vector = extract_index_samples(neighbor_index,train_samples);

            %find anno in neighbor image
            anno_index = find(train_anno(neighbor_index,:) ~= 0);
            for j = 1 : length(anno_index)
                %calculate similarity prob through basic distance
                base_dis = base_distance_vector(test_vector, neighbor_vector);
                
                if(~isempty(label_model(anno_index(j)).M))
                    %similar pair, calculate prob
                    try
                        [predict_label, accuracy, dec_value] = ...
                            predict([1], sparse(base_dis),label_model(anno_index(j)).M,'-q');
                    catch
                        display('error');
                    end
                    if(predict_label > 0)
                        anno_score(d, anno_index(j)) = anno_score(d, anno_index(j)) + 1;      
                    end
                    anno_score_decvalue(d, anno_index(j)) = anno_score(d, anno_index(j)) + exp(dec_value);
                end
            end
        end
        
        if mod(d, 50) == 0
            fprintf('predict %d test samples finished!\n', d);
        end
    end
    
elseif(learn_method == 6) %label-based LMNN 
    %set model name
    model_dir = 'label_basedLMNN';
    learned_model_name = 'label_basedLMNN.mat';
    load(fullfile(MODEL_DIR, model_dir, learned_model_name));
    
    %load original multi-features of train and test set
    load(fullfile(RUN_DIR, Global.Train_Feature_Dir, 'train_multifeature_corel5k.mat'));
    load(fullfile(RUN_DIR, Global.Test_Dir, 'test_multifeature_corel5k.mat'));
    
    for d = 1:D   
        %get d-th test vector
        test_vector = extract_index_samples(d, test_samples); %a struct
        %select top nn neighbors to predict
        if(Global.Test_Neighbors == 0)
            nn = length(test_sample_pairs{d});
        else
            nn = Global.Test_Neighbors;
        end
        
        for i = 1 : nn
            %caculate distance between current test image and its i neighbor
            neighbor_index = test_sample_pairs{d}(i).img2.id;
            neighbor_vector = extract_index_samples(neighbor_index,train_samples);

            %find anno in neighbor image
            anno_index = find(train_anno(neighbor_index,:) ~= 0);
            for j = 1 : length(anno_index)
                %calculate similarity prob through basic distance
                base_dis = base_distance_vector(test_vector, neighbor_vector);
                
                if(~isempty(label_model(anno_index(j)).M))
                    %similar pair, calculate prob
                    dec_value = label_model(anno_index(j)).M * base_dis';
                    if (dec_value < 0)
                        fprintf('sample %d, label %d, distance value is negative: %f \n', d, anno_index(j), dec_value);        
                        continue;
                    end
                    anno_score(d, anno_index(j)) = anno_score(d, anno_index(j)) - dec_value;      
                    anno_score_decvalue(d, anno_index(j)) = anno_score_decvalue(d, anno_index(j)) + exp(-dec_value);
                end
            end
        end
        
        if mod(d, 50) == 0
            fprintf('predict %d test samples finished!\n', d);
        end
    end

elseif(learn_method == 7) % label-based LMNN pegasos    
    %set model name
    model_dir = 'label_basedLMNN_pegasos';
    learned_model_name = 'label_basedLMNN_pegasos.mat';
    load(fullfile(MODEL_DIR, model_dir, learned_model_name));
    
    %load original multi-features of train/test set, save as cell format
    load(fullfile(RUN_DIR, Global.Train_Feature_Dir, 'train_multifeature_corel5k.mat'));
    load(fullfile(RUN_DIR, Global.Test_Dir, 'test_multifeature_corel5k.mat'));
    
    idFea = 1;
    Fea_Alpha{idFea} = train_samples.alpha_denseHUE; idFea=idFea+1;
    Fea_Alpha{idFea} = train_samples.alpha_denseSIFT; idFea=idFea+1;
    Fea_Alpha{idFea} = train_samples.alpha_GIST;
    Fea_Type = Global.Feature_Type;

    for d = 1:D   
        %get d-th test vector
        test_vector = extract_index_samples(d, test_samples); %a struct
        %select top nn neighbors to predict
        if(Global.Test_Neighbors == 0)
            nn = length(test_sample_pairs{d});
        else
            nn = Global.Test_Neighbors;
        end
        
        for i = 1 : nn
            %caculate distance between current test image and its i neighbor
            neighbor_index = test_sample_pairs{d}(i).img2.id;
            neighbor_vector = extract_index_samples(neighbor_index,train_samples);

            %find anno in neighbor image
            anno_index = find(train_anno(neighbor_index,:) ~= 0);
            for j = 1 : length(anno_index)
                %calculate similarity prob through basic distance
%                 base_dis = base_distance_vector(test_vector, neighbor_vector);
                
                if(~isempty(label_model(anno_index(j)).W) && ~isempty(label_model(anno_index(j)).V))
                    %similar pair, calculate prob
                    dec_value = CalculateMultipleDistance(test_vector, neighbor_vector,Fea_Alpha,...
                        label_model(anno_index(j)).W,label_model(anno_index(j)).V, Fea_Type);
                    if (dec_value < 0)
                        fprintf('sample %d, label %d, distance value is negative: %f \n', d, anno_index(j), dec_value);        
                        continue;
                    end
                    anno_score(d, anno_index(j)) = anno_score(d, anno_index(j)) - dec_value;      
                    anno_score_decvalue(d, anno_index(j)) = anno_score_decvalue(d, anno_index(j)) + exp(-dec_value);
                end
            end
        end
        
        if mod(d, 50) == 0
            fprintf('predict %d test samples finished!\n', d);
        end
        anno_score(anno_score == 0) = -Inf;
    end
end


save(fullfile(MODEL_DIR, model_dir, 'predict_test.mat'),'anno_score','anno_score_decvalue');
fprintf('\n finished predict labels for test samples, save predicted results finished! \n');

end


function index_samples = extract_index_samples(sample_index, set_samples)
% this function is to extract multifeature structures for indexed imgs


    index_samples.denseHUE = set_samples.denseHUE(sample_index,:);
    index_samples.denseSIFT = set_samples.denseSIFT(sample_index,:);
    index_samples.GIST = set_samples.GIST(sample_index,:);
%     index_samples.HSV = set_samples.HSV(sample_index,:);
%     index_samples.LAB = set_samples.LAB(sample_index,:);
%     index_samples.RGB = set_samples.RGB(sample_index,:);

end

function dist_value = CalculateMultipleDistance(test_vector, neigh_vector, alpha, w, v,type)
    if(length(w) ~= length(v))
        fprintf('error: w and v should be identical dimension! \n');
        return;
    end
    
    dist_value = 0; dist = zeros(length(w),1);
    %sum each dist
    for i = 1 : length(w)
        if(strcmp(type{i}, 'hue')) %L1
            dist(i) = GetL1Dist(test_vector.denseHUE, neigh_vector.denseHUE, alpha{i},v{i});
        elseif(strcmp(type{i}, 'sift'))
            dist(i) = GetL1Dist(test_vector.denseSIFT, neigh_vector.denseSIFT, alpha{i},v{i});
        elseif(strcmp(type{i}, 'gist'))
            dist(i) = GetL2Dist(test_vector.GIST, neigh_vector.GIST, alpha{i},v{i});
        else
            display('error: error feature type input! \n');
        end
    end
    
    dist_value = w * dist;
end

function [varargout] = GetL1Dist(hist1, hist2, alpha,weights)
% hist1 and hist2 a nxk matrix, weights is a 1xk vector 
% return: dist nx1, vect nxk
vect = bsxfun(@times, abs(hist1 - hist2), weights);

% dist = sum(vect, 2) / alpha.alpha_sum;
dist = sum(vect, 2);
if nargout == 1 %only output dist
    varargout{1} = dist;
elseif nargout == 2 % output dist and vect
    varargout{1} = dist;
    varargout{2} = vect;
else
    display('error output format!');
end

end

function [varargout] = GetL2Dist(hist1, hist2, alpha,weights)
% hist1 and hist2 are nxk matrix, weights is a 1xk vector 
vect = abs(hist1 - hist2);
vect = vect.^2;
vect = bsxfun(@times, vect, weights);
% normalize, assume that each feature contributes equally
% dist = sum(vect, 2) / alpha.alpha_sum;
dist = sum(vect, 2);
if nargout == 1 %only output dist
    varargout{1} = dist;
elseif nargout == 2 % output dist and vect
    varargout{1} = dist;
    varargout{2} = vect;
else
    display('error output format!');
end

end