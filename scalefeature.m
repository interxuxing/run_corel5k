function [newFea, alpha] = scalefeature(oldFea, new_max, new_min)
%FUNCTION [newFea] = scalefeature(oldFea) is to scale the old feature matrix to
%   new feature matrix.
%
%   oldFea: a NxP matrix, N is sample number, P is feature dimension
%   newFea: a scaled NxP matrix
%scale method: 
% newFea = (oldFea - minF) * (new_max - new_min) / (maxF - minF) + new_min
%
%   return:
%   newFea: scaled NxP matrix
%   alpha: 1xP matrix, scaled parameter for each dimension

[N P] = size(oldFea);

minF = min(oldFea);
maxF = max(oldFea);
maxDist = maxF - minF;

%alpha = sum(maxDist(:,  2:size(oldFea, 2)));
alpha = maxDist;

newFea = (new_max - new_min) * ...
    (oldFea - repmat(minF, N, 1)) ./ repmat(maxDist, N, 1) + new_min;

%if newFea = NaN, set it to be 0
newFea(isnan(newFea)) = 0;