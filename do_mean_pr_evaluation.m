function do_mean_pr_evaluation(config_file)
%% This file is used to evaluate mean P and mean R
%   parameters

eval(config_file);

%load ground truth test annotation
word_matrix_gt = double(vec_read(fullfile(IMAGE_ANNOTATION_DIR, 'corel5k_test_annot.hvecs')));

%load predict test annotation;
if Global.Learn_method == 4
     model_dir = 'label_basedLRLINEAR';
end
load(fullfile(MODEL_DIR, model_dir, 'predict_test_tagprop.mat'));

[document_number, N] = size(word_matrix_gt);

word_matrix_predict = anno_score;
% word_matrix_predict = anno_score_decvalue;
%get top-5 word as final text word
[prob_value, prob_index] = sort(word_matrix_predict,2,'descend');
prob_index_tp5 = prob_index(:,1:5);

word_matrix_pred = zeros(size(word_matrix_gt));
for d = 1 : document_number
    word_matrix_pred(d, prob_index_tp5(d)) = 1;
end

Mean_P = zeros(N, 1);
Mean_R = zeros(N, 1);


%% calculate overall statistics for each label
for n = 1 : N
    gt_n = word_matrix_gt(:, n);
    pred_n = word_matrix_pred(:, n);
    
    Mean_P(n) = precision(gt_n, pred_n);
    Mean_R(n) = recall(gt_n, pred_n);
end


%% cacluate mean Recall for specific labels
% image 0-5
word_image_total = sum(word_matrix_gt, 1); %1xD
R_5 = mean(Mean_R(word_image_total <= 5));

R_10 = mean(Mean_R(word_image_total > 5 & word_image_total <= 10));

R_50 = mean(Mean_R(word_image_total > 10 & word_image_total <= 50));

R_100 = mean(Mean_R(word_image_total > 5 & word_image_total <= 100));

R_100_more = mean(Mean_R(word_image_total > 100));

end

function ret = precision(gt, label)
    tp = sum(label == 1 & gt == 1);
    tp_fp = sum(gt == 0);
    if tp_fp == 0;
      disp(sprintf('warning: No positive predict label.'));
      ret = 0;
    else
      ret = tp / tp_fp;
    end
    disp(sprintf('Precision = %g%% (%d/%d)', 100.0 * ret, tp, tp_fp));
end

function ret = recall(gt, label)
    tp = sum(label == 1 & gt == 1);
    tp_fn = sum(label == 1);
    if tp_fn == 0;
      disp(sprintf('warning: No postive true label.'));
      ret = 0;
    else
      ret = tp / tp_fn;
    end
    disp(sprintf('Recall = %g%% (%d/%d)', 100.0 * ret, tp, tp_fn));
end

