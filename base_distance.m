%base_distance(sample_vector, set_vector)
%FUNCTION base_distance is to calculate the basic weighted multiple
%distance on multiple feature type
% each feature takes equal weight
%
% input parameters
%       sample_vector   ---     a given sample, a cell of FT types of
%                                   features, total M dimension
%       set_vector      ---     N samples to be compared with sample_vector a cell of
%                                   FT types of features, total M
%                                   dimension
%       inter_weights   ---     a 1xK matrix
%       dist_socre      ---     a matrix / vector store the distance 1xN
function dist_score = base_distance(sample_vector, set_vector, inter_weights)
%get count of feature types 
if(nargin ~= 3)
    %use default inter_weight
    inter_weights = ones(1,3);
end

%use different metric to caculate distance score for different feature type
% L1 for color features HSV, LAB, RGB
% dist_score_HSV = slmetric_pw(sample_vector.HSV', set_vector.HSV', 'cityblk');
% 
% dist_score_LAB = slmetric_pw(sample_vector.LAB', set_vector.LAB', 'cityblk');
% 
% dist_score_RGB = slmetric_pw(sample_vector.RGB', set_vector.RGB', 'cityblk');
% 
% %L2 for global feature GIST
% dist_score_GIST = slmetric_pw(sample_vector.GIST', set_vector.GIST', 'eucdist');
% 
% %Chi-Square Distance for textual feature SIFT, HUE
% dist_score_SIFT = slmetric_pw(sample_vector.denseSIFT', set_vector.denseSIFT', 'chisq');
% 
% dist_score_HUE = slmetric_pw(sample_vector.denseHUE', set_vector.denseHUE', 'chisq');
% 
% %merge the final base distance score
% dist_score = (dist_score_HSV + dist_score_LAB + dist_score_RGB + dist_score_GIST + ...
%     dist_score_SIFT + dist_score_HUE) .* weight;

% note!!! slmetric_pw returns a row vector 1xN, GetXXDist returns a colunm
% vector Nx1

%L2 for global feature GIST
dist_score_GIST = GetL2Dist(set_vector.GIST, sample_vector.GIST, set_vector.alpha_GIST);

%L1 for textual feature SIFT, HUE
dist_score_SIFT = GetL1Dist(set_vector.denseSIFT ,sample_vector.denseSIFT, set_vector.alpha_denseSIFT);

dist_score_HUE = GetL1Dist(set_vector.denseHUE, sample_vector.denseHUE, set_vector.alpha_denseHUE);

%merge the final base distance score use inter_weights
dist_score = [dist_score_GIST, dist_score_SIFT, dist_score_HUE] * inter_weights';

dist_score = dist_score'; % return a row vector 1xN

