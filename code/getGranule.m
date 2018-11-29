function [DataGranules] = getGranule(SampleSet,SampleLabel)
% [DataGranules] = getGranule(SampleSet,SampleLabel)
% This code process the unbiased dataset using the idea of GSVM and output
% the granule sets
% 
% Input
% SampleSet: training data(normalized), which size is (m,n),i.e. m samples with n features
%            in each.
% SampleLabel: label matrix of training data, (m,1)
%
% Output
% DataGranules, a struct with .MajIdx, .GraNum, .GraIdx, MinIdx
%
% Version:3.0
% Date: 2018/11/29
% Author: Zhixian MA <zx@mazhixian.me>
% https://github.com/myinxd/gbt-svm

% Init
DataGranules.MajIdx = 0;
DataGranules.GraNum = 0;
DataGranules.MinIdx = 0;

% Obtain the number of samples in each class
y_col = length(SampleLabel(1,:));
if (y_col > 1)
    disp('Label matrix should be a vector, not onehot style.');
else
    ClassIdx = unique(SampleLabel);  % Index
    ClassNum = length(ClassIdx); % Number of classes
    SampleNum = zeros(1,ClassNum); % Number of samples for each class
    for i = 1 : ClassNum
        SampleNum(i) = length(find(SampleLabel == ClassIdx(i)));
    end
    % Judge whether the data is unbalanced
    [ClassMax,MaxIdx] = max(SampleNum);
    DataGranules.MajIdx = ClassIdx(MaxIdx); % Label w.r.t the major number class
    [ClassMin,MinIdx] = min(SampleNum);
    DataGranules.MinIdx = ClassIdx(MinIdx); % Label w.r.t the minor number class
    if (ClassMax/ClassMin > 1)
        % the minor
        DataGranules.MinData.index = find(SampleLabel ~= ClassIdx(MaxIdx));
        DataGranules.MinData.SampleSet =  SampleSet(DataGranules.MinData.index,:);
        DataGranules.MinData.SampleLabel =  SampleLabel(DataGranules.MinData.index);
        % the major
        DataGranules.MajData.index = find(SampleLabel == ClassIdx(MaxIdx));
        DataGranules.MajData.SampleSet = SampleSet(DataGranules.MajData.index,:);
        DataGranules.MajData.SampleLabel = SampleLabel(DataGranules.MajData.index);
        % Gen granules, randomly divide
        DataGranules.GraNum = floor(ClassMax/ClassMin);
        DataGranules.GraIdx = cell(1,DataGranules.GraNum);
        for i = 1 : DataGranules.GraNum
            DataGranules.GraIdx{i} = i:DataGranules.GraNum:ClassMin*DataGranules.GraNum+i-1;
        end
    else
        DataGranules.GraIdx{1} = DataTemp.index;
        DataGranules.GraNum = 1;
        % Major
        DataGranules.MajData.index = find(SampleLabel == ClassIdx(MaxIdx));
        DataGranules.MajData.SampleSet = SampleSet(DataGranules.MajData.index,:);
        DataGranules.MajData.SampleLabel = SampleLabel(DataGranules.MajData.index);
        % Minor
        DataGranules.MinData.index = find(SampleLabel ~= ClassIdx(MaxIdx));
        DataGranules.MinData.SampleSet =  SampleSet(DataGranules.MinData.index,:);
        DataGranules.MinData.SampleLabel =  SampleLabel(DataGranules.MinData.index);
    end
end
