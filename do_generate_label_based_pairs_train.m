function do_generate_label_based_pairs_train(config_file)
%%This function is to find semantic neighbors (targets and imposters) for
%each traning sample based on its multi labels from 2PKNN method
%
% The procedures for the semantic neighbors finding are following steps:
%   1, for each sample image with m labels
%       (1) find N1xm target neighbors in all m labels
%       (2) find N1x(V-m) imposter neighbors in other (V-m) labels
%       (3) remove overlaped neighbors in both target and imposter
%   2, mark the similar (target) / dissimilar (imposter) for each
%   sample
%   example: for a sample img with 5 labels, total 100 labels, N1 =4;
%   similar pairs are 5x4, dissimilar pairs are (100-5)x4

    
    eval(config_file);

    %% load original train set annotation
    img_anno_fname = 'corel5k_train_annot.hvecs';
    mat_img_anno = vec_read(fullfile(IMAGE_ANNOTATION_DIR, img_anno_fname));
    [D, W] = size(mat_img_anno);
    
    %% load multifeature train_full, subset semantic_group
    load(fullfile(RUN_DIR, Global.Train_Feature_Dir, 'train_multifeature_corel5k.mat'));
    load(fullfile(RUN_DIR, Global.Train_Feature_Dir, 'seman_group_subset_corel5k.mat'));
    
    N1 = Global.Semantic_Neighbor;
    
    tstart = tic;
    
    %% find the target / imposter for all the unique samples in subset
    sample_dir = 'unique_samples';
    if ~exist(fullfile(LABEL_PAIRS_DIR, 'train', sample_dir),'dir')
            mkdir(fullfile(LABEL_PAIRS_DIR, 'train', sample_dir));
    end
     
    %% rename all file
%     for s = 1 : length(subset_unique_index)
%         sample_index_i = subset_unique_index(s);
%         
%         temp1 = sprintf('sample_%d_rename.mat', sample_index_i);
%         ori_name = fullfile(LABEL_PAIRS_DIR, 'train', sample_dir, temp1);
%         
%         temp2 = sprintf('sample_%d.mat', sample_index_i);
%         dst_name = fullfile(LABEL_PAIRS_DIR, 'train', sample_dir, temp2);
%         
%         try
%             movefile(ori_name, dst_name);
%         catch
%             lasterr
%         end
%     end
    if 0
    for s = 1 : length(subset_unique_index)
        label_pairs = [];
        
        sample_index_i = subset_unique_index(s);
        %find other tags for current sample
        label_i = (mat_img_anno(sample_index_i,:));
        label_i_index = find(label_i == 1);
        
        %get current sample vector
        sample_vector = extract_set_samples(sample_index_i, train_samples);

        %select N1 intra simimilar pairs
        sim_index = select_intra_neighbors(sample_vector, train_samples, seman_group_subset, N1, label_i_index, sample_index_i);

         %select N1 inter dissimlar pairs
        dissim_index = select_inter_neighbors(sample_vector, train_samples, seman_group_subset, N1, W, label_i_index, sample_index_i);

        %remove overlapped index in both target / imposter neighbors
        [c, ia, ib] = intersect(sim_index, dissim_index);
        sim_index(ia) = [];
        dissim_index(ib) = [];

        label_pairs = generate_label_pairs(label_pairs, sample_index_i, sim_index, dissim_index);
        label_pairs_overlap = c;
        % set label mat name one by one
        label_mat_name = sprintf('sample_%d.mat', sample_index_i);
        save(fullfile(LABEL_PAIRS_DIR, 'train', sample_dir, label_mat_name),'label_pairs', 'label_pairs_overlap');
        
        if mod(s, 100) == 0
            fprintf(' generate %d-th sample pairs finished!\n', s);
        end
    end
    end
    
    if 1
    %% loop for each label group, then copy each mat for sample_dir to each label dir
    for w = 1 : W
        seman_group_w = seman_group_subset.label_img_index{w}; %column vector
        count_w = length(seman_group_w);
        
        %save pairs for each sample under the label directory
        label_dir = sprintf('label_%d', w);
        
        if ~exist(fullfile(LABEL_PAIRS_DIR, 'train', label_dir),'dir') && count_w > N1
            mkdir(fullfile(LABEL_PAIRS_DIR, 'train', label_dir));
        end
        
        if(count_w > N1)
            for i = 1 : count_w
                sample_index_i = seman_group_w(i);     
                           
                % copy this index file to label_dir
                label_mat_name = sprintf('sample_%d.mat', sample_index_i);
                src_file = fullfile(LABEL_PAIRS_DIR, 'train', sample_dir, label_mat_name);
                dst_file = fullfile(LABEL_PAIRS_DIR, 'train', label_dir, label_mat_name);
                
                try
                    [info_1, info_2, info_3] = copyfile(src_file, dst_file);
                catch
                    lasterr
                end
            end
        end
       
        if mod(w, 10) == 0
            fprintf(' copy all sample pairs for %d -th label finished!\n', w);
        end  
    end
    
    toc(tstart);
    end
end

%% this function is to select N1 intra neighbors (share tags with sample_vector)
function sim_index = select_intra_neighbors(sample_vector, train_samples, semantic_group, N1, label_index,sample_index)
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
%             v_index_N1 = v_index(1:N1);
%             %remove the same label_index in inter neighbors
%             temp = w_index(v_index_N1);
%             temp(temp == sample_index) = [];
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
    sim_index = unique_temp(v_u_index);
end

%% this function is to select N1 inter neighbors(no tags shared in sample_vector)
function dissim_index = select_inter_neighbors(sample_vector, train_samples, semantic_group, N1, W, label_index,sample_index)
    % this function is for step 2.2
    distance_inter_label_index = [];
    distance_inter_label_score = [];
    
    temp = [1:1:W];
    temp(label_index) = [];
    imposter_label_index = temp;
    
    %find top N1 in each semantic group for sample_vector
    for w = imposter_label_index
        %all samples in label w cluster
         w_index = semantic_group.label_img_index{w};
         
        if(length(w_index) > N1)
           %calculate base distance of sample and w-th semantic group
            set_vector_w = extract_set_samples(w_index, train_samples);
            distance_score = base_distance(sample_vector,set_vector_w);
           
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
            
%             v_index_N1 = v_index(1:N1);
%             %remove the same label_index in inter neighbors
%             temp = w_index(v_index_N1);
%             temp(temp == sample_index) = [];
            if(isempty(distance_inter_label_index))
                distance_inter_label_index = temp_index;
                distance_inter_label_score = temp_score;
            else
                distance_inter_label_index = [distance_inter_label_index; temp_index];
                distance_inter_label_score = [distance_inter_label_score, temp_score];
            end        
        end
    end
    % return dissimilar index and remove repeat index
    % ensure that the dissim_index is sorted by distance ascend
    [unique_value, unique_index] = unique(distance_inter_label_index);
    [v_u, v_u_index] = sort(distance_inter_label_score(unique_index), 2, 'ascend');
    
    unique_temp = distance_inter_label_index(unique_index);
    dissim_index = unique_temp(v_u_index);
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

%% this function is to merge all samples'pairs in one label group
function pairs = generate_label_pairs(pairs, img_index, sim_pairs, dissim_pairs)
    n_sim = length(sim_pairs);
    n_dissim = length(dissim_pairs);
    
    if(isempty(pairs))
        for i = 1 : n_sim
            pairs(i).img1.id = img_index;
            pairs(i).img2.id = sim_pairs(i);

            pairs(i).match = 1;
            pairs(i).training = 1;
        end

        for i = (n_sim+1) : (n_sim + n_dissim)
            ind = i - n_sim;
            pairs(i).img1.id = img_index;
            pairs(i).img2.id = dissim_pairs(ind);

            pairs(i).match = 0;
            pairs(i).training = 1;
        end
    
    else
        n_count = length(pairs);

        for i = (n_count+1) : (n_count+n_sim)
            ind = i - n_count;
            pairs(i).img1.id = img_index;
            pairs(i).img2.id = sim_pairs(ind);

            pairs(i).match = 1;
            pairs(i).training = 1;
        end

        for i = (n_count+n_sim+1) : (n_count + n_sim + n_dissim)
            ind = i - n_count - n_sim;
            pairs(i).img1.id = img_index;
            pairs(i).img2.id = dissim_pairs(ind);

            pairs(i).match = 0;
            pairs(i).training = 1;
        end
    end
        
end