function Dist = distance_multiple_features(sample_vector,set_vector, w)

% function distance_multiple_features()
% This function is to calculate distance between sample_vector and it's
% neighbors
%
% Input parameters:
%   sample_vector: 1xd
%   set_vector: nxd
%   w: 1xd weights
% Output parameters"
%   Dist: nxd

%% predefine the range of each descriptor
MULTIPLE_FEATURE_DIM = 13900;
RANGE_HUE = [1:100];
RANGE_SIFT = [101:1100];
RANGE_GIST = [1101:1612];
RANGE_HSV = [1613:5708];
RANGE_LAB = [5709:9804];
RANGE_RGB = [9805:13900];

[N, D] = size(set_vector);

%now calculate the distanc of each feature type
%% HUE  chi-squared
X1_HUE = sample_vector(RANGE_HUE);
X2_HUE = set_vector(:, RANGE_HUE);
X1_HUE_rep = repmat(X1_HUE,N,1); %sample_num x X1_HSV
X_HUE = (X1_HUE_rep - X2_HUE).^2 ./ (2 * (X1_HUE_rep + X2_HUE));
X_HUE((X1_HUE_rep + X2_HUE) == 0) = 0;

%% SIFT chi-squared
X1_SIFT = sample_vector(RANGE_SIFT);
X2_SIFT = set_vector(:, RANGE_SIFT);
X1_SIFT_rep = repmat(X1_SIFT,N,1); %sample_num x X1_HSV

X_SIFT = (X1_SIFT_rep - X2_SIFT).^2 ./ (2 * (X1_SIFT_rep + X2_SIFT));
X_SIFT((X1_SIFT_rep + X2_SIFT) == 0) = 0; %set NaN to be zero


%% GIST L2
X1_GIST = sample_vector(RANGE_GIST);
X2_GIST = set_vector(:, RANGE_GIST);
X1_GIST_rep = repmat(X1_GIST,N,1); %sample_num x X1_HSV
X_GIST = (X1_GIST_rep - X2_GIST).^2;

%% HSV L1
X1_HSV = sample_vector(RANGE_HSV);
X2_HSV = set_vector(:, RANGE_HSV);
X1_HSV_rep = repmat(X1_HSV,N,1); %sample_num x X1_HSV
X_HSV = abs(X1_HSV_rep - X2_HSV);

%% LAB L1
X1_LAB = sample_vector(RANGE_LAB);
X2_LAB = set_vector(:, RANGE_LAB);
X1_LAB_rep = repmat(X1_LAB,N,1); %sample_num x X1_HSV
X_LAB = abs(X1_LAB_rep - X2_LAB);

%% RGB L1
X1_RGB = sample_vector(RANGE_RGB);
X2_RGB = set_vector(:, RANGE_RGB);
X1_RGB_rep = repmat(X1_RGB,N,1); %sample_num x X1_HSV
X_RGB = abs(X1_RGB_rep - X2_RGB);

%% Now merge all these distances
Dist = [X_HUE, X_SIFT, X_GIST, X_HSV, X_LAB, X_RGB] .* repmat(w, N,1);