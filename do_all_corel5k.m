function do_all_espgame(config_file)
%% A simple script to run all the steps 
%% All sub steps below can be run on by on.
%% 1, random training sample indice for subset
%% 2, generate multiple feature for random indice
%% 3, generate label pairs for random sample subset
%% 4, learn model from subset training set
%% 5, generate label pairs for test samples
%% 6, predict labels for test samples 
%% 7, PR evaluation

do_random_train_indices(config_file);

do_generate_multiple_feature(config_file);

do_generate_label_based_pairs_train(config_file);

do_learn_label_based_distance_metric(config_file);

do_generate_label_based_pairs_test(config_file);

do_predict_labels_test(config_file);

do_pr_evaluation(config_file);