function [predict_label,accuracy,decisionValue] = myGSVMpredict(x_test,y_test,model)
% [predict_label,accuracy,decisionValue] = myGSVMpredict(x_test,y_test,model)
% This function predict the class of test samples using the models
% genreated from GSVM
% Input
% y_test: the predict label of test data
% x_test: the test data
% model: models from GSVM
%
% Output
% predict_label: prodecited labels
% accuarcy: predict accuracy
% decisionValue: decisiont values for classification
% Author: Zhixian Ma (Email: zxma_sjtu@qq.com)
% Version 3.0
% Date: 2016/06/11

% Init
predict_label = zeros(size(y_test));
ModelNum = length(model);

% predict
p = zeros(length(y_test),ModelNum);
acc = cell(1,ModelNum);
dv = p;

for i = 1 : ModelNum
    TestSet = mapminmax(x_test',model{i}.PS);
    TestSet = TestSet';
    [p(:,i),acc{i},dv(:,i)] = svmpredict(y_test,TestSet,model{i}.model);
end
% Final judge
for j = 1 : length(y_test)
   ClassType = unique(p(j,:));
   ClassNum = zeros(1,length(ClassType));
   for i = 1 : length(ClassType)
       ClassNum(i) = length(find(p(j,:) == ClassType(i)));
   end
   [ClassMax,Idx] = max(ClassNum);
   predict_label(j) = ClassType(Idx(1));
end

accuracy = (1 - sum(double(xor(predict_label,y_test)))/length(y_test))*100;
decisionValue = sum(dv,2)/ModelNum;

% for i = 1 : ModelNum
%     if(accuracy < acc{i}(1))
%         predict_label = p(:,i);
%         accuracy = acc{i}(1);
%     end
% end



