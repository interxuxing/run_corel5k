function do_generate_multiple_feature_shaoting(config_file)
%%Function that generate multiple feature for train / test set simenteniously:
%% new_fea = old_fea / \sum_{|max_e - min_e|}

eval(config_file);

%% load feature data of train / test
feature_file = fullfile(IMAGE_ANNOTATION_DIR, 'corel5k_train_DenseHue.hvecs');
denseHUE_train = double(vec_read(feature_file));
[N_train, m] = size(denseHUE_train);
feature_file = fullfile(IMAGE_ANNOTATION_DIR, 'corel5k_test_DenseHue.hvecs');
denseHUE_test = double(vec_read(feature_file));
[N_test, m] = size(denseHUE_test);

feature_file = fullfile(IMAGE_ANNOTATION_DIR, 'corel5k_train_DenseSift.hvecs');
denseSIFT_train = double(vec_read(feature_file));
feature_file = fullfile(IMAGE_ANNOTATION_DIR, 'corel5k_test_DenseSift.hvecs');
denseSIFT_test = double(vec_read(feature_file));

feature_file = fullfile(IMAGE_ANNOTATION_DIR, 'corel5k_train_Gist.fvec');
GIST_train = double(vec_read(feature_file));
feature_file = fullfile(IMAGE_ANNOTATION_DIR, 'corel5k_test_Gist.fvec');
GIST_test = double(vec_read(feature_file));

%% scale for each feature
% Hue
FeaMat = [denseHUE_train; denseHUE_test];
minFea = min(FeaMat);
maxFea = max(FeaMat);
maxDist = maxFea - minFea;
alpha = sum(maxDist,2);

train_samples.denseHUE = FeaMat(1:N_train,:) / alpha;
train_samples.alpha_denseHUE = alpha;
test_samples.denseHUE = FeaMat(N_train+1:end,:) / alpha;
test_samples.alpha_denseHUE = alpha;
% SIFT
FeaMat = [denseSIFT_train; denseSIFT_test];
minFea = min(FeaMat);
maxFea = max(FeaMat);
maxDist = maxFea - minFea;
alpha = sum(maxDist,2);

train_samples.denseSIFT = FeaMat(1:N_train,:) / alpha;
train_samples.alpha_denseSIFT = alpha;
test_samples.denseSIFT = FeaMat(N_train+1:end,:) / alpha;
test_samples.alpha_denseSIFT = alpha;
% GIST
FeaMat = [GIST_train; GIST_test];
minFea = min(FeaMat);
maxFea = max(FeaMat);
maxDist = maxFea - minFea;
alpha = sum(maxDist,2);

train_samples.GIST = FeaMat(1:N_train,:) / alpha;
train_samples.alpha_GIST = alpha;
test_samples.GIST = FeaMat(N_train+1:end,:) / alpha;
test_samples.alpha_GIST = alpha;



%% save mat file for train / test
train_features_full = [train_samples.denseHUE, train_samples.denseSIFT, train_samples.GIST];
%save trainning data
save(fullfile(RUN_DIR, Global.Train_Feature_Dir, 'train_multifeature_corel5k.mat'), 'train_samples');
save(fullfile(RUN_DIR, Global.Train_Feature_Dir, 'train_features_full.mat'), 'train_features_full');
display('save train_multifeature_corel5k.mat.');

test_features_full = [test_samples.denseHUE, test_samples.denseSIFT, test_samples.GIST];
%save trainning data
if ~exist(fullfile(RUN_DIR, Global.Test_Dir), 'dir')
    mkdir(fullfile(RUN_DIR, Global.Test_Dir));
end
save(fullfile(RUN_DIR, Global.Test_Dir, 'test_multifeature_corel5k.mat'), 'test_samples');
save(fullfile(RUN_DIR, Global.Test_Dir, 'test_features_full.mat'), 'test_features_full');
display('save test_multifeature_corel5k.mat.');