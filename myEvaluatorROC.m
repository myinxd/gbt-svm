function [TPR,FPR,AUC] = myEvaluatorROC(PreLabel,RealLabel,TrueLabel,PreValue)
% [P,R,F1] = myEvaluatorNew(PreLabel,RealLabel,TrueLabel,PreValue)
% This code aims to evaluate the performance of our classifier based on the
% PR thinking, i.e. the RoC curve
%
% Input
% PreLabel: predict label
% RealLabel: real label
% TrueLabel: the label of true 
% PreValue: the sample values generate from the SVM prediciton
% Output
% TPR: True positive rate
% FPR: False positive rate
% AUC: Area under ROC curve
%
% Version: 1.0
% Date: 2016/06/10
% Zhixian MA

% Init
Ns =length(PreLabel);
TPR = zeros(Ns,1);
FPR = TPR;
NumTrue = length(find(RealLabel == TrueLabel));
NumFalse = length(find(RealLabel ~= TrueLabel));
% Sort
[PreValueSort,PreIdx] = sort(PreValue,'descend');
if TrueLabel == 1
    MatLabel = [RealLabel(PreIdx),ones(Ns,1)];
else
    MatLabel = [RealLabel(PreIdx),zeros(Ns,1)];
end

% Main
for i = 2 : Ns
    if MatLabel(i,1) == MatLabel(i,2)
        TPR(i) = TPR(i-1) + 1/NumTrue;
        FPR(i) = FPR(i-1);
    else
        TPR(i) = TPR(i-1);
        FPR(i) = FPR(i-1) + 1/NumFalse;
    end
end

AUC = 1/2*sum((FPR(2:end)-FPR(1:end-1)).*(TPR(2:end)+TPR(1:end-1)));



    




