%%%%% Global configuration file %%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% DIRECTORIES - please change if copying the code to a new location
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Path for laptop
IMAGE_SOURCE_DIR = 'C:\workspace\program\image-annotation\others\Annotation_Demo\ESP-ImageSet\images';
IMAGE_ANNOTATION_DIR = 'D:\workspace-limu\image annotation\Annotation_Demo\corel5k.20091111';
RUN_DIR = 'D:\workspace-limu\image annotation\iciap2013\labelknn\run_corel5k';
LABEL_PAIRS_DIR = 'D:\workspace-limu\image annotation\iciap2013\labelknn\run_corel5k\label_pairs';
MODEL_DIR = 'D:\workspace-limu\image annotation\iciap2013\labelknn\run_corel5k\model';

% Path for desktop
% IMAGE_SOURCE_DIR = 'D:\workspace-limu\image annotation\Annotation_Demo\ESP-ImageSet\images';
% IMAGE_ANNOTATION_DIR = 'D:\workspace-limu\image annotation\Annotation_Demo\espgame.20091111';
% RUN_DIR = 'D:\workspace-limu\image annotation\iciap2013\labelknn\run_espgame';
% LABEL_PAIRS_DIR = 'D:\workspace-limu\image annotation\iciap2013\labelknn\run_espgame\label_pairs';
% MODEL_DIR = 'D:\workspace-limu\image annotation\iciap2013\labelknn\run_espgame\model';
% TEMP_DATA_DIR = 'D:\workspace-limu\image annotation\iciap2013\labelknn\run_espgame\tempdata';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% GLOBAL PARAMETERS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% how many zeros to prefix image, interest and model files by....
Global.Num_Zeros = 4;
Global.Semantic_Neighbor = 4;
%% subdirectory, file prefix and file extension of images 
Global.Image_Dir_Name = 'images';
Global.Image_File_Name = 'image_';
%%% changing the extension changes to image format used...
Global.Image_Extension = '.jpg';
Global.Train_Feature_Dir = 'train_feature';
Global.Test_Dir = 'test';
Global.Test_Neighbors = 30;
Global.Metric = 'base';
Global.Random_Rate = 1.0;

%% multiple feature extraction (train, train_subset, test)
Global.Multiple_Feature = 'test';
%% multiple feature scale type (normalize, scale)
Global.Feature_Scale = 'scale'; 
%% learn method
Global.Learn_method = 4;


Global.Max_Iteration = 500;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% INITIAL ADDITIONAL PATH
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ROOT_DIR = 'C:\workspace\program\image-annotation\iciap2013\labelknn';

TOOLBOX_DIR = fullfile(ROOT_DIR, 'toolbox');
addpath(genpath(TOOLBOX_DIR));

TOOLBOX_DIR = fullfile(ROOT_DIR, 'toolbox','pwmetric');
addpath(genpath(TOOLBOX_DIR));

