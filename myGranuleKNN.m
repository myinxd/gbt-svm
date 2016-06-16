function [DataGranules] = myGranuleKNN(SampleSet,SampleLabel)
% [DataGranules] = myGranuleKNN(SampleSet,SampleLabel)
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
% Version:1.0
% Date: 2016/05/27
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
    if (ClassMax/ClassMin >= 2)
        % Gen granules, knn
        DataGranules.GraNum = ceil(ClassMax/ClassMin);
        DataGranules.GraIdx = cell(1,DataGranules.GraNum);
        DataTemp.index = find(SampleLabel == ClassIdx(MaxIdx));
        DataTemp.SampleSet = SampleSet(DataTemp.index,:);
        DataTemp.SampleLabel = SampleLabel(DataTemp.index);
        DataGranules.MajData = DataTemp;
        k = ClassMin;
        % the rest
        DataGranules.OtherData.index = find(SampleLabel ~= ClassIdx(MaxIdx));
        DataGranules.OtherData.SampleSet =  SampleSet(DataGranules.OtherData.index,:);
        DataGranules.OtherData.SampleLabel =  SampleLabel(DataGranules.OtherData.index);
        % knn
        i = 1;
        while ~isempty(DataTemp.SampleSet)
            SampleTemp = DataTemp.SampleSet(1,:);
            % repeat matrix 
            SampleCmp = repmat(SampleTemp,length(DataTemp.SampleSet(:,1)),1);
            % Get distance, Euclidean
            SampleDist = (DataTemp.SampleSet - SampleCmp); 
            SampleDist = SampleDist.*SampleDist;
            SampleDist = sum(SampleDist');
            % Save granule
            [SampleDistSort,Idx] = sort(SampleDist);
            if length(Idx) < k
                DataGranules.GraIdx{i} = Idx;
                DataTemp.SampleSet(Idx,:) = [];
            else
                DataGranules.GraIdx{i} = Idx(1:k);
                DataTemp.SampleSet(Idx(1:k),:) = [];
            end
            % Discard the extracted
            i = i + 1;
        end
    else
        DataTemp.index = find(SampleLabel == ClassIdx(MaxIdx));
        DataTemp.SampleSet = SampleSet(DataTemp.index,:);
        DataTemp.SampleLabel = SampleLabel(DataTemp.index);
        DataGranules.MajData = DataTemp;
        DataGranules.GraIdx = DataTemp.index;
        DataGranules.OtherData.index = find(SampleLabel ~= ClassIdx(MaxIdx));
        DataGranules.OtherData.SampleSet =  SampleSet(DataGranules.OtherData.index,:);
        DataGranules.OtherData.SampleLabel =  SampleLabel(DataGranules.OtherData.index);
    end
end
