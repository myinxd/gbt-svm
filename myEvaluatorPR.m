function [P,R] = myEvaluatorPR(PreLabel,RealLabel,TrueLabel,PreValue)
% [P,R,F1] = myEvaluatorNew(PreLabel,RealLabel,PreValue)
% This code aims to evaluate the performance of our classifier based on the
% PR thinking, i.e. the Precision and Recall curve and F1
%
% Input
% PreLabel: predict label
% RealLabel: real label
% TrueLabel: True label
% PreValue: the sample values generate from the SVM prediciton
% Output
% P, R:
% Precision = TP / (TP+FP)
% Recall = TP / (TP + FN);
% 
% Version: 1.0
% Date: 2016/06/10
% Zhixian MA

% Init
Ns =length(PreLabel);
P = zeros(Ns,1);
R = P; F1 = P;

% Sort
[PreSort,PreIdx] = sort(PreValue,'descend');
if TrueLabel == 1
    MatReal = RealLabel(PreIdx);
else
    MatReal = RealLabel(PreIdx);
    MatReal = abs(MatReal - 1);
end
% Main Body
for i = 1:Ns
    MatPre = [ones(i,1);zeros(Ns-i,1)];
    R(i) = sum(MatReal.*MatPre)/sum(MatReal);
    P(i) = sum(MatReal.*MatPre)/sum(MatPre);
end


    




