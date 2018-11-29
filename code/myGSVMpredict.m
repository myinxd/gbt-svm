function [predict_label,accuracy,decisionValue] = myGSVMpredict(x_test,y_test,model)
% [predict_label,accuracy,decisionValue] = myGSVMpredict(x_test,y_test,model)
% This function predict the class of test samples using the models
% genreated from GSVM
% 
% Input
% y_test: the predict label of test data
% x_test: the test data
% model: models from GSVM
%
% Output
% predict_label: prodecited labels
% accuarcy: predict accuracy
% decisionValue: decision values for classification
% Version 4.0
% Date: 2018/11/29
% Author: Zhixian MA <zx@mazhixian.me>
% https://github.com/myinxd/gbt-svm

% Init
predict_label = zeros(size(y_test));
ModelNum = length(model); % number of granule models for voting

% predict
p = zeros(length(y_test),ModelNum);
acc = cell(1,ModelNum);
dv = zeros(length(y_test),ModelNum);


for i = 1 : ModelNum
    TestSet = mapminmax(x_test',model{i}.PS); % Normalization
    TestSet = TestSet';
    [p(:,i),acc{i},dv(:,i)] = svmpredict(y_test,TestSet,model{i}.model);
end
% Final judge: voting
for j = 1 : length(y_test)
   ClassType = unique(p(j,:));
   ClassNum = zeros(1,length(ClassType));
   ClassProb = zeros(1,length(ClassType));
   for i = 1 : length(ClassType)
       ClassNum(i) = length(find(p(j,:) == ClassType(i)));
       ClassProb(i) = abs(sum(dv(j, (find(p(j,:) == ClassType(i))))));
   end
   [~,Idx] = max(ClassNum);
   if length(Idx) > 1
       [~, idxj] = max(ClassProb(Idx));
   else
       idxj = Idx(1);
   end
   predict_label(j) = ClassType(idxj);
end

accuracy = (1 - sum(double(xor(predict_label,y_test)))/length(y_test))*100;
decisionValue = sum(dv,2)/ModelNum;



