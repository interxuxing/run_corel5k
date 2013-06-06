function do_pr_evaluation(config_file)
%% This file is used to evaluate annotation word according to model
%   parameters

%According to the formuler
%   p(w | r) = sum_m{sum_zm}{q(zm | phi)p(w | zm,beta)}
%
eval(config_file);

%load ground truth test annotation
word_matrix_gt = double(vec_read(fullfile(IMAGE_ANNOTATION_DIR, 'corel5k_test_annot.hvecs')));

%load predict test annotation;
if Global.Learn_method == 4
    model_dir = 'label_basedLRLINEAR';
    fprintf('\n Evaluate with label_basedLRLINEAR \n');
elseif Global.Learn_method == 6
    model_dir = 'label_basedLMNN';
    fprintf('\n Evaluate with label_basedLMNN \n');
elseif Global.Learn_method == 7
    model_dir = 'label_basedLMNN_pegasos';
    fprintf('\n Evaluate with label_basedLMNN_pegasos \n');
end
% load(fullfile(MODEL_DIR, model_dir, 'predict_test_tagprop.mat'));
load(fullfile(MODEL_DIR, model_dir, 'predict_test.mat'));
[document_number, N] = size(word_matrix_gt);

%% use vote scheme: anno_socre
fprintf('(1)......Evaluation using vote scheme (anno_score): \n ');
word_matrix_predict = anno_score;
%get top-5 word as final text word
[prob_value, prob_index] = sort(word_matrix_predict,2,'descend');
prob_index_tp5 = prob_index(:,1:5);

Precision = []; Recall = []; MAP = 0; Mean_MAP = 0; total_words = 0;

for i = 1:document_number
    Index = find(word_matrix_gt(i,:) == 1);
    term = length(Index);
    total_words = total_words + term;
     
    count = 0;
    %find each gt anno in predict matrix
    for j = 1 : term
        try
            res = find(prob_index_tp5(i,:) == Index(j));
        catch
            lasterr;
        end
        if ~isempty(res)
            MAP = MAP + 1;
            count = count+1;
        end
    end
    
    Precision(i) = double(count / 5);
    if (term ~= 0)
        Recall(i) = double(count / term);
    else
        Recall(i) = 0;
    end
    
    if(mod(i,100)==0)
        display(['process ' num2str(i)  ' images finished!']);
    end
    
end

%caculate total annotated word for each image: number * 10
total_anno = (document_number  - length(find(sum(word_matrix_gt,2) == 0))) * 5;

Mean_MAP = MAP / total_anno;
Mean_MAR = MAP / total_words;
N_words = length(unique(prob_index_tp5(:)));
fprintf(sprintf('Evaluation using vote scheme: P %f, R %f, MAP %d, N+ %d\n',Mean_MAP, Mean_MAR, MAP, N_words));

%% use distance scheme anno_score_decvalue
fprintf('(2)......Evaluation using distance scheme (anno_score_decvalue): \n ');

word_matrix_predict = anno_score_decvalue;
%get top-5 word as final text word
[prob_value, prob_index] = sort(word_matrix_predict,2,'descend');
prob_index_tp5 = prob_index(:,1:5);

Precision = []; Recall = []; MAP = 0; Mean_MAP = 0; total_words = 0;

for i = 1:document_number
    Index = find(word_matrix_gt(i,:) == 1);
    term = length(Index);
    total_words = total_words + term;
     
    count = 0;
    %find each gt anno in predict matrix
    for j = 1 : term
        try
            res = find(prob_index_tp5(i,:) == Index(j));
        catch
            lasterr;
        end
        if ~isempty(res)
            MAP = MAP + 1;
            count = count+1;
        end
    end
    
    Precision(i) = double(count / 5);
    if (term ~= 0)
        Recall(i) = double(count / term);
    else
        Recall(i) = 0;
    end
    
    if(mod(i,100)==0)
        display(['process ' num2str(i)  ' images finished!']);
    end
    
end

%caculate total annotated word for each image: number * 10
total_anno = (document_number  - length(find(sum(word_matrix_gt,2) == 0))) * 5;

Mean_MAP = MAP / total_anno;
Mean_MAR = MAP / total_words;
N_words = length(unique(prob_index_tp5(:)));
fprintf(sprintf('Evaluation using distance scheme: P %f, R %f, MAP %d, N+ %d\n',Mean_MAP, Mean_MAR, MAP, N_words));


%% another evalution method, statistics for each dict word
fprintf('(3)......Evaluation using global scheme : \n ');
wPred = zeros(size(word_matrix_gt));

for d = 1 : document_number
    wPred(d, prob_index_tp5(d,:)) = 1;
end

[P,R,mAP, mAR] = evalRetrieval(wPred, word_matrix_gt);

fprintf(sprintf('Evaluation using global scheme: mAP %f, mAR %f \n',mAP, mAR));