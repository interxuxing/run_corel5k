%base_distance_vector(X1, X2)
%FUNCTION base_distance_vector is to calculate the basic weighted multiple
%distance on multiple feature type
%
% input parameters
%       X1, X2   ---     2 input row vector of samples
%       X2 could be N row vector , not only single row vector
% return a vector with same dimention of X1, X2
function X_OUT = base_distance_vector(sample_vector, set_vector)

%get count of feature types 

%use different metric to caculate distance score for different feature type
% L1 for color features HSV, LAB, RGB
% dist_score_HSV = slmetric_pw(sample_vector.HSV', set_vector.HSV', 'cityblk');
X1_HSV = sample_vector.HSV;
X2_HSV = set_vector.HSV;

[sample_num, feature_dim] = size(X2_HSV);
%COPY sample_number X1_HSV
X1_HSV_rep = repmat(X1_HSV,sample_num,1); %sample_num x X1_HSV

X_HSV = abs(X1_HSV_rep - X2_HSV);
% X_HSV = normalize(X_HSV, 2);

% dist_score_LAB = slmetric_pw(sample_vector.LAB', set_vector.LAB', 'cityblk');
X1_LAB = sample_vector.LAB;
X2_LAB = set_vector.LAB;

[sample_num, feature_dim] = size(X2_LAB);
%COPY sample_number X1_HSV
X1_LAB_rep = repmat(X1_LAB,sample_num,1); %sample_num x X1_HSV

X_LAB = abs(X1_LAB_rep - X2_LAB);
% X_LAB = normalize(X_LAB, 2);
% dist_score_RGB = slmetric_pw(sample_vector.RGB', set_vector.RGB', 'cityblk');
X1_RGB = sample_vector.RGB;
X2_RGB = set_vector.RGB;

[sample_num, feature_dim] = size(X2_RGB);
%COPY sample_number X1_HSV
X1_RGB_rep = repmat(X1_RGB,sample_num,1); %sample_num x X1_HSV

X_RGB = abs(X1_RGB_rep - X2_RGB);
% X_RGB = normalize(X_RGB, 2);
%L2 for global feature GIST
% dist_score_GIST = slmetric_pw(sample_vector.GIST', set_vector.GIST', 'eucdist');
X1_GIST = sample_vector.GIST;
X2_GIST = set_vector.GIST;

[sample_num, feature_dim] = size(X2_GIST);
%COPY sample_number X1_HSV
X1_GIST_rep = repmat(X1_GIST,sample_num,1); %sample_num x X1_HSV

X_GIST = X1_GIST_rep .* X1_GIST_rep + X2_GIST .* X2_GIST - 2 * X1_GIST_rep .* X2_GIST;
X_GIST = sqrt(X_GIST);
% X_GIST = normalize(X_GIST, 2);
%Chi-Square Distance for textual feature SIFT, HUE
% dist_score_SIFT = slmetric_pw(sample_vector.denseSIFT', set_vector.denseSIFT', 'chisq');
X1_SIFT = sample_vector.denseSIFT;
X2_SIFT = set_vector.denseSIFT;

[sample_num, feature_dim] = size(X2_SIFT);
%COPY sample_number X1_HSV
X1_SIFT_rep = repmat(X1_SIFT,sample_num,1); %sample_num x X1_HSV

X_SIFT = (X1_SIFT_rep - X2_SIFT).^2 ./ (2 * (X1_SIFT_rep + X2_SIFT));
X_SIFT((X1_SIFT_rep + X2_SIFT) == 0) = 0; %set NaN to be zero
% X_SIFT = normalize(X_SIFT, 2);
% dist_score_HUE = slmetric_pw(sample_vector.denseHUE', set_vector.denseHUE', 'chisq');
X1_HUE = sample_vector.denseHUE;
X2_HUE = set_vector.denseHUE;

[sample_num, feature_dim] = size(X2_HUE);
%COPY sample_number X1_HSV
X1_HUE_rep = repmat(X1_HUE,sample_num,1); %sample_num x X1_HSV

X_HUE = (X1_HUE_rep - X2_HUE).^2 ./ (2 * (X1_HUE_rep + X2_HUE));
X_HUE((X1_HUE_rep + X2_HUE) == 0) = 0;
% X_HUE = normalize(X_HUE, 2);

%merge the final base distance score
% dist_score = (dist_score_HSV + dist_score_LAB + dist_score_RGB + dist_score_GIST + ...
%     dist_score_SIFT + dist_score_HUE) .* weight;
X_OUT = [X_HSV, X_LAB, X_RGB, X_GIST, X_SIFT, X_HUE];


