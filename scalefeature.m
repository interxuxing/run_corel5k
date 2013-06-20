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

if(nargin == 1)     % do not scale, just return original feature and alpha
    % first rescale oldFea
    minF = min(oldFea);
    maxF = max(oldFea);
    maxDist = maxF - minF;
    maxDist(maxDist == 0) = Inf;    
    newFea = bsxfun(@rdivide,oldFea, maxDist);
    
    minF = min(newFea);
    maxF = max(newFea);
    maxDist = maxF - minF;
    
    alpha.alpha_v = maxDist; 
    alpha.alpha_v(alpha.alpha_v == 0) = Inf;
    
    alpha.alpha_sum = sum(maxDist);
    
elseif(nargin == 3)
    [N P] = size(oldFea);
    minF = min(oldFea);
    maxF = max(oldFea);
    maxDist = maxF - minF;

    alpha.alpha_v = maxDist; 
    alpha.alpha_v(alpha.alpha_v == 0) = Inf;
    alpha.alpha_sum = sum(maxDist);

    newFea = (new_max - new_min) * ...
        (oldFea - repmat(minF, N, 1)) ./ repmat(maxDist, N, 1) + new_min;

    %if newFea = NaN, set it to be 0
    newFea(isnan(newFea)) = 0; 
else
    display('myToolbox:myFunction:fileNotFound');
    return;
end