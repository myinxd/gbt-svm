% Copyright (C) 2016-2018 Zhixian MA <zx@mazhixian.me>
% MIT License
% An example to train the granular SVM model

% Init
clc; clear; close all

% load data
load ../data/SampleSet

% Get granules
DataGranules = getGranule(X_train,Y_train);

GraNum = DataGranules.GraNum;
% Init the models
ModelGSVM = cell(1,GraNum);
SubSetMin = DataGranules.MinData.SampleSet;
SubLabelMin = DataGranules.MinData.SampleLabel;

% Training with multi-thread
parpool(2) % Two threads
parfor i = 1 : GraNum
    SubSetMaj = DataGranules.MajData.SampleSet(DataGranules.GraIdx{i},:);
    SubLabelMaj = DataGranules.MajIdx * ones(length(DataGranules.GraIdx{i}),1);
    SubTrainSet = [SubSetMaj;SubSetMin];
    SubTrainLabel = [SubLabelMaj;SubLabelMin];
    [ModelGSVM{i}.model,ModelGSVM{i}.PS] = myCrossSVM(SubTrainSet,SubTrainLabel,'RBF');
end
%matlabpool close

save ModelGSVM ModelGSVM
