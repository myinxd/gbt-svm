function [DataGranules] = myGranuleSub(SampleSet,SampleLabel)
% [DataGranules] = myGranuleSub(SampleSet,SampleLabel)
% This code process the unbiased dataset using the idea of GSVM and output
% the granule sets
% 
% Input
% SampleSet: training data(normalized), which size is (m,n),i.e. m samples with n features
%            in each.
% SampleLabel: label matrix of training data, (m,1)
%
% Output
% DataGranules, a structure with .MajIdx, .GraNum, .GraIdx, MinIdx
%
% Version:2.0
% Date: 2016/06/01
% Zhixian MA

% Init
DataGranules.MajIdx = 0;
DataGranules.GraNum = 0;
DataGranules.MinIdx = 0;

% Estimate the amount of samples of each class
y_col = length(SampleLabel(1,:));
if (y_col > 1)
    disp('Label matrix should be a vector.');
else
    ClassIdx = unique(SampleLabel);  % Index
    ClassNum = length(ClassIdx); % Number of classes
    SampleNum = zeros(1,ClassNum); % Number of samples for each class
    for i = 1 : ClassNum
        SampleNum(i) = length(find(SampleLabel == ClassIdx(i)));
    end
    % Judge whether the data is unbalanced
    [ClassMax,MaxIdx] = max(SampleNum);
    DataGranules.MajIdx = ClassIdx(MaxIdx); % Class index of the maximum amount classification
    [ClassMin,MinIdx] = min(SampleNum);
    DataGranules.MinIdx = ClassIdx(MinIdx);
    if (ClassMax/ClassMin > 1)
        % the rest
        DataGranules.OtherData.index = find(SampleLabel ~= ClassIdx(MaxIdx));
        DataGranules.OtherData.SampleSet =  SampleSet(DataGranules.OtherData.index,:);
        DataGranules.OtherData.SampleLabel =  SampleLabel(DataGranules.OtherData.index);
        % Gen granules, knn
        DataGranules.GraNum = floor(ClassMax/ClassMin);
        DataGranules.GraIdx = cell(1,DataGranules.GraNum);
        DataGranules.MajData.index = find(SampleLabel == ClassIdx(MaxIdx));
        DataGranules.MajData.SampleSet = SampleSet(DataGranules.MajData.index,:);
        DataGranules.MajData.SampleLabel = SampleLabel(DataGranules.MajData.index);
        for i = 1 : DataGranules.GraNum
            DataGranules.GraIdx{i} = i:DataGranules.GraNum:ClassMin*DataGranules.GraNum+i-1;
        end
    else
        DataTemp.index = find(SampleLabel == ClassIdx(MaxIdx));
        DataTemp.SampleSet = SampleSet(DataTemp.index,:);
        DataTemp.SampleLabel = SampleLabel(DataTemp.index);
        DataGranules.MajData = DataTemp;
        DataGranules.GraIdx{1} = DataTemp.index;
        DataGranules.GraNum = 1;
        DataGranules.OtherData.index = find(SampleLabel ~= ClassIdx(MaxIdx));
        DataGranules.OtherData.SampleSet =  SampleSet(DataGranules.OtherData.index,:);
        DataGranules.OtherData.SampleLabel =  SampleLabel(DataGranules.OtherData.index);
    end
end
