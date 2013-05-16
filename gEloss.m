function [eloss, geloss] = gEloss(old_G, D_target, D_imposter, V_target, V_imposter, mu)

% function [eloss, geloss] = gEloss(old_G, D_target, D_imposter, V_target, V_imposter, mu)
%
% Input:
%       old_G: old gradient  1xd vector
%       D_target: distance value of (sample, target), Px1 matrix
%       D_imposter: distance matrix of (sample, imposter), Nx1 matrix
%       V_target: distance matrix of (sample, target), Pxd matrix
%       V_imposter: distance matrix of (sample, imposter), Nxd matrix
%       mu: scalar value
%
% Output:
%       eloss: energy loss value
%       geloss: new gradient 1xd vector
%
% Author: Xing Xu (xing.xu@ieee.org)
% Date: 2013.5.15
