function do_random_train_indices(config_file)
%%Function that randomly select partial samples from the entire original
%%training set.
%%  1, statistic the entire label-based semantic group
%%  2, randomly subset, static the subset semantic group



eval(config_file);

%%ensure subset dir exists
if ~exist(fullfile(RUN_DIR, Global.Train_Feature_Dir), 'dir')
    [s, m1, m2] = mkdir(RUN_DIR, Global.Train_Feature_Dir);
end
%% first check if random_indices exist
if exist(fullfile(RUN_DIR, Global.Train_Feature_Dir, 'random_indices.mat'),'file');
  load(fullfile(RUN_DIR, Global.Train_Feature_Dir, 'random_indices.mat'));
else %% if it doesn't exist create it....
  display('random_indices.mat does not exist - run do_random_indices.m to create it');
  
  %% 1 step
  if ~exist(fullfile(RUN_DIR, Global.Train_Feature_Dir, 'seman_group_corel5k.mat'),'file')
      display('1, statistic the entire label-based semantic group.');
      %%first get original train annotation file
      train_annotation_file = 'corel5k_train_annot.hvecs';
      mat_img_anno = vec_read(fullfile(IMAGE_ANNOTATION_DIR, train_annotation_file));

      [D W] = size(mat_img_anno);

        for w = 1 : W
            %record img index for each label semantic group
            seman_group.label_img_index{w} = find(mat_img_anno(:,w) ~= 0);
        end

        save(fullfile(RUN_DIR, Global.Train_Feature_Dir, 'seman_group_corel5k.mat'),'seman_group');
        display('save seman_group_corel5k.');
  end
  
  %% 2 step
   display('2, randomly subset, static the subset semantic group');
   load(fullfile(RUN_DIR, Global.Train_Feature_Dir, 'seman_group_corel5k.mat'));
   
   W = length(seman_group.label_img_index);
   subset_unique_index = [];
   for w = 1 : W
       N_total = length(seman_group.label_img_index{w});
       N_random = floor(N_total * Global.Random_Rate);         
       index_random = randperm(N_total);

       seman_group_subset.label_img_index{w} = seman_group.label_img_index{w}(index_random(1:N_random)); % row vector N_random*1

       if isempty(subset_unique_index)
           subset_unique_index = seman_group_subset.label_img_index{w};
       else
           subset_unique_index = [subset_unique_index; seman_group_subset.label_img_index{w}];
       end
   end

   subset_unique_index = unique(subset_unique_index);

    save(fullfile(RUN_DIR, Global.Train_Feature_Dir, 'seman_group_subset_corel5k.mat'),'seman_group_subset','subset_unique_index');
    display('save seman_group_subset_corel5k finished.');   
  end
  
end