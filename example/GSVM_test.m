% Copyright (C) 2016-2018 Zhixian MA <zx@mazhixian.me>
% MIT License
% An example to train the granular SVM model
% Test code

clc; clear; close all

load ../data/SampleSet
load ../data/ModelGSVM

% Randomly select test samples
numSample = length(Y_test);
numTest = 600;
idx = randperm(numSample);

% Predict and output the result
[FinalPredict,Acc,dv] = myGSVMpredict(X_test(idx(1:numTest),:), Y_test(idx(1:numTest)), ModelGSVM);
fprintf('GSVM accuracy: %f\n', Acc)
