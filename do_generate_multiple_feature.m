function do_generate_multiple_feature(config_file)
%%Function that generate multiple feature for:
%% 1, entire original training set
%% 2, subset tranining set
%% 3, entire test set

eval(config_file);

% feature scale type
if strcmp(Global.Feature_Scale, 'normalize')
    feature_scale = 1;
else strcmp(Global.Feature_Scale, 'scale')
    feature_scale = 2;
end


if strcmp(Global.Multiple_Feature,'train')
%% 1 step   
    display('1, entire original training set');
    %% load all types of feature: denseHue, denseSIFT, GIST, HSV, LAB, RGB
	%% ensure that all 6 types of features are already normalized
    feature_file = fullfile(IMAGE_ANNOTATION_DIR, 'corel5k_train_DenseHue.hvecs');
    if feature_scale == 1
        train_samples.denseHUE = normalize(double(vec_read(feature_file)), 2);
    else
        [train_samples.denseHUE, train_samples.alpha_denseHUE] = scalefeature(double(vec_read(feature_file)));
    end

    feature_file = fullfile(IMAGE_ANNOTATION_DIR, 'corel5k_train_DenseSift.hvecs');
    if feature_scale == 1
        train_samples.denseSIFT = normalize(double(vec_read(feature_file)), 2);
    else
        [train_samples.denseSIFT, train_samples.alpha_denseSIFT] = scalefeature(double(vec_read(feature_file)));
    end

    feature_file = fullfile(IMAGE_ANNOTATION_DIR, 'corel5k_train_Gist.fvec');
    if feature_scale == 1
        train_samples.GIST = normalize(double(vec_read(feature_file)), 2);
    else
        [train_samples.GIST, train_samples.alpha_GIST] = scalefeature(double(vec_read(feature_file)));
    end

%     feature_file = fullfile(IMAGE_ANNOTATION_DIR, 'corel5k_train_Hsv.hvecs32');
%     if feature_scale == 1
%         train_samples.HSV = normalize(double(vec_read(feature_file)), 2);
%     else
%         [train_samples.HSV, train_samples.alpha_HSV] = scalefeature(double(vec_read(feature_file)));
%     end
% 
%     feature_file = fullfile(IMAGE_ANNOTATION_DIR, 'corel5k_train_Lab.hvecs32');
%     if feature_scale == 1
%         train_samples.LAB = normalize(double(vec_read(feature_file)), 2);
%     else
%         [train_samples.LAB, train_samples.alpha_LAB] = scalefeature(double(vec_read(feature_file)));
%     end
% 
%     feature_file = fullfile(IMAGE_ANNOTATION_DIR, 'corel5k_train_Rgb.hvecs32');
%     if feature_scale == 1
%         train_samples.RGB = normalize(double(vec_read(feature_file)), 2);
%     else
%         [train_samples.RGB, train_samples.alpha_RGB] = scalefeature(double(vec_read(feature_file)));
%     end

    %merge multiple features to one long matrix
    train_features_full = [train_samples.denseHUE, train_samples.denseSIFT, train_samples.GIST];
%     train_features_full = [train_samples.denseHUE, train_samples.denseSIFT, train_samples.GIST, ...
%         train_samples.HSV, train_samples.LAB, train_samples.RGB];
    %save trainning data
    save(fullfile(RUN_DIR, Global.Train_Feature_Dir, 'train_multifeature_corel5k.mat'), 'train_samples');
    save(fullfile(RUN_DIR, Global.Train_Feature_Dir, 'train_features_full.mat'), 'train_features_full');
    display('save train_multifeature_corel5k.mat.');
    
elseif strcmp(Global.Multiple_Feature, 'train_subset')
%% 2 step
    display('2, subset tranining set');
    
    if ~exist(fullfile(RUN_DIR, Global.Train_Feature_Dir, 'train_multifeature_corel5k.mat'), 'file')
        %%first do 1 step
		feature_file = fullfile(IMAGE_ANNOTATION_DIR, 'corel5k_train_DenseHue.hvecs');
        if feature_scale == 1
            train_samples.denseHUE = normalize(double(vec_read(feature_file)), 2);
        else
            [train_samples.denseHUE, train_samples.alpha_denseHUE] = scalefeature(double(vec_read(feature_file)));
        end

        feature_file = fullfile(IMAGE_ANNOTATION_DIR, 'corel5k_train_DenseSift.hvecs');
        if feature_scale == 1
            train_samples.denseSIFT = normalize(double(vec_read(feature_file)), 2);
        else
            [train_samples.denseSIFT, train_samples.alpha_denseSIFT] = scalefeature(double(vec_read(feature_file)));
        end

        feature_file = fullfile(IMAGE_ANNOTATION_DIR, 'corel5k_train_Gist.fvec');
        if feature_scale == 1
            train_samples.GIST = normalize(double(vec_read(feature_file)), 2);
        else
            [train_samples.GIST, train_samples.alpha_GIST] = scalefeature(double(vec_read(feature_file)));
        end

        feature_file = fullfile(IMAGE_ANNOTATION_DIR, 'corel5k_train_Hsv.hvecs32');
        if feature_scale == 1
            train_samples.HSV = normalize(double(vec_read(feature_file)), 2);
        else
            [train_samples.HSV, train_samples.alpha_HSV] = scalefeature(double(vec_read(feature_file)));
        end

        feature_file = fullfile(IMAGE_ANNOTATION_DIR, 'corel5k_train_Lab.hvecs32');
        if feature_scale == 1
            train_samples.LAB = normalize(double(vec_read(feature_file)), 2);
        else
            [train_samples.LAB, train_samples.alpha_LAB] = scalefeature(double(vec_read(feature_file)));
        end

        feature_file = fullfile(IMAGE_ANNOTATION_DIR, 'corel5k_train_Rgb.hvecs32');
        if feature_scale == 1
            train_samples.RGB = normalize(double(vec_read(feature_file)), 2);
        else
            [train_samples.RGB, train_samples.alpha_RGB] = scalefeature(double(vec_read(feature_file)));
        end
    else
        load(fullfile(RUN_DIR, Global.Train_Feature_Dir, 'train_multifeature_corel5k.mat'));   
    end
    
    %%statistic unique train indinces from subset semantic group
        if ~exist(fullfile(RUN_DIR, Global.Train_Feature_Dir, 'seman_group_subset_corel5k.mat'), 'file')
            error('seman_group_subset_corel5k not exist! first do_random_train_indices');
        else
            load(fullfile(RUN_DIR, Global.Train_Feature_Dir, 'seman_group_subset_corel5k.mat'));
            
            train_subset_samples.denseHue = train_samples.denseHue(subset_unique_index,:);
            train_subset_samples.denseSIFT = train_samples.denseSIFT(subset_unique_index,:);
            train_subset_samples.GIST = train_samples.GIST(subset_unique_index,:);
            train_subset_samples.HSV = train_samples.HSV(subset_unique_index,:);
            train_subset_samples.LAB = train_samples.LAB(subset_unique_index,:);
            train_subset_samples.RGB = train_samples.RGB(subset_unique_index,:);
        end
        
        %save trainning data
        save(fullfile(RUN_DIR, Global.Train_Feature_Dir, 'train_multifeature_subset_corel5k.mat'), 'train_subset_samples');
        display('save train_multifeature_subset_corel5k.mat.'); 

elseif strcmp(Global.Multiple_Feature, 'test')
%% 3 step
    display('3, entire test set');
    %load all types of feature: denseHue, denseSIFT, GIST, HSV, LAB, RGB
    feature_file = fullfile(IMAGE_ANNOTATION_DIR, 'corel5k_test_DenseHue.hvecs');
    if feature_scale == 1
        test_samples.denseHUE = normalize(double(vec_read(feature_file)), 2);
    else
        [test_samples.denseHUE, test_samples.alpha_denseHUE] = scalefeature(double(vec_read(feature_file)));
    end

    feature_file = fullfile(IMAGE_ANNOTATION_DIR, 'corel5k_test_DenseSift.hvecs');
    if feature_scale == 1
        test_samples.denseSIFT = normalize(double(vec_read(feature_file)), 2);
    else
        [test_samples.denseSIFT, test_samples.alpha_denseSIFT] = scalefeature(double(vec_read(feature_file)));
    end
    
    feature_file = fullfile(IMAGE_ANNOTATION_DIR, 'corel5k_test_Gist.fvec');
    if feature_scale == 1
        test_samples.GIST = normalize(double(vec_read(feature_file)), 2);
    else
        [test_samples.GIST, test_samples.alpha_GIST] = scalefeature(double(vec_read(feature_file)));
    end
    
%     feature_file = fullfile(IMAGE_ANNOTATION_DIR, 'corel5k_test_Hsv.hvecs32');
%     if feature_scale == 1
%         test_samples.HSV = normalize(double(vec_read(feature_file)), 2);
%     else
%         [test_samples.HSV, test_samples.alpha_HSV] = scalefeature(double(vec_read(feature_file)));
%     end
%     
%     feature_file = fullfile(IMAGE_ANNOTATION_DIR, 'corel5k_test_Lab.hvecs32');
%     if feature_scale == 1
%         test_samples.LAB = normalize(double(vec_read(feature_file)), 2);
%     else
%         [test_samples.LAB, test_samples.alpha_LAB] = scalefeature(double(vec_read(feature_file)));
%     end
%     
%     feature_file = fullfile(IMAGE_ANNOTATION_DIR, 'corel5k_test_Rgb.hvecs32');
%     if feature_scale == 1
%         test_samples.RGB = normalize(double(vec_read(feature_file)), 2);
%     else
%         [test_samples.RGB, test_samples.alpha_RGB] = scalefeature(double(vec_read(feature_file)));
%     end
%     
%     test_features_full = [test_samples.denseHUE, test_samples.denseSIFT, test_samples.GIST, ...
%         test_samples.HSV, test_samples.LAB, test_samples.RGB];
    
    test_features_full = [test_samples.denseHUE, test_samples.denseSIFT, test_samples.GIST];
    %save trainning data
    if ~exist(fullfile(RUN_DIR, Global.Test_Dir), 'dir')
        mkdir(fullfile(RUN_DIR, Global.Test_Dir));
    end
    save(fullfile(RUN_DIR, Global.Test_Dir, 'test_multifeature_corel5k.mat'), 'test_samples');
    save(fullfile(RUN_DIR, Global.Test_Dir, 'test_features_full.mat'), 'test_features_full');
    display('save test_multifeature_corel5k.mat.');

end