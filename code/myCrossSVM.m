function [Model,NormMat] = myCrossSVM(TrainSet,TrainLabel,KernelType)
% [Model, NormMat] = myCrossSVM(TrainSet,TrainLabel,KernelType)
% This code generate the SVM Model of our GBT-SVM algorithm
% We use the crossvalidation and gridding to determine the parameters based 
% on the kernel type 
%
% Input
% TrainSet: the training feature set
% TrainLabel: the label vector
% KernerType: Kerner function,could be
%             |        Ker func       | Abbr.|
%             | Radial basis function |  RBF |
%             |        Linuear        |  LIN |
%             |       Polynomial      |  POL |
%                           ...
% Output
% Model: the classification model
% NormMat: the mapminmax parameters of the TrainSet
%
% Version 1.0
% Date: 2016/05/27
% Author: Zhixian MA <zx@mazhixian.me>
% https://github.com/myinxd/gbt-svm

if nargin < 3
    KernelType = 'RBF';
end

% Normalization
[TrainSet,NormMat] = mapminmax(TrainSet');
TrainSet = TrainSet';

% Judge whether data structure of TrainLabel is right
y_col = length(TrainLabel(1,:));
if (y_col > 1)
    disp('Label matrix should be a vector.');
else
    switch KernelType
        case 'LIN'
            c = -10:0.5:10;m = size(c);
            cg = zeros(m); eps = 10^-4;
            v = 5; t = 0;
            bestc = 1; bestg = 0.1; bestacc = 0;
            for i = 1 : m
                cmd = ['-v ',num2str(v), ' -c ', num2str(2^c(i)), ' -t ', num2str(t)];
                cg(i) = svmtrain(TrainLabel,TrainSet,cmd);
                if (cg(i) > bestacc)
                    bestacc = cg(i);
                    bestc = 2^c(i);
                end
                if abs(cg(i) - bestacc) <= eps && bestc > 2^c(i)
                        bestacc = cg(i);
                        bestc = 2^c(i);
                end
            end
            cmd = ['-c ',num2str(bestc),' -t ', num2str(t)];
            Model = svmtrain(TrainLabel,TrainSet,cmd);
        case 'RBF'
            [c,g] = meshgrid(-5:0.5:5,-5:0.2:5);
            [m,n] = size(c);
            cg = zeros(m,n); eps = 10^-4; v = 5;
            bestc = 1; bestg = 0.1; bestacc = 0;
            for i = 1 : m
                for j = 1 : n
                    cmd = ['-v ',num2str(v), ' -c ', num2str(2^c(i,j)), ' -g ', num2str(2^g(i,j))];
                    cg(i,j) = svmtrain(TrainLabel,TrainSet,cmd);
                    if (cg(i,j) > bestacc)
                        bestacc = cg(i,j);
                        bestc = 2^c(i,j);
                        bestg = 2^g(i,j);
                    end
                    if abs(cg(i,j) - bestacc) <= eps %&& bestc > 2^c(i,j)
                        bestacc = cg(i,j);
                        bestc = 2^c(i,j);
                        bestg = 2^g(i,j);
                    end
                end
            end
            cmd = ['-c ',num2str(bestc), ' -g ', num2str(bestg)];
            Model = svmtrain(TrainLabel,TrainSet,cmd);
        case 'SIG'
            [c,g] = meshgrid(-10:0.5:10,-10:0.2:10);
            [m,n] = size(c);
            cg = zeros(m,n); eps = 10^-4; v = 5; t = 3;
            bestc = 1; bestg = 0.1; bestacc = 0;
            for i = 1 : m
                for j = 1 : n
                    cmd = ['-v ',num2str(v), ' -c ', num2str(2^c(i,j)),' -t ', num2str(t), ' -g ', num2str(2^g(i,j))];
                    cg(i,j) = svmtrain(TrainLabel,TrainSet,cmd);
                    if (cg(i,j) > bestacc)
                        bestacc = cg(i,j);
                        bestc = 2^c(i,j);
                        bestg = 2^g(i,j);
                    end
                    if abs(cg(i,j) - bestacc) <= eps && bestc > 2^c(i,j)
                        bestacc = cg(i,j);
                        bestc = 2^c(i,j);
                        bestg = 2^g(i,j);
                    end
                end
            end
            cmd = ['-c ',num2str(bestc),' -t ', num2str(t), ' -g ', num2str(bestg)];
            Model = svmtrain(TrainLabel,TrainSet,cmd);
        case 'POL'
            [c,g] = meshgrid(-10:0.5:10,-10:0.2:10); d = 1 : 10;
            [m,n] = size(c);
            cg = zeros(m,n); eps = 10^-4; v = 5; t = 3;
            bestc = 1; bestg = 0.1; bestd = 1; bestacc = 0;
            for i = 1 : m
                for j = 1 : n
                    for k = 1 : length(d)
                        cmd = ['-v ',num2str(v), ' -c ', num2str(2^c(i,j)),' -t ', num2str(t), ' -g ', num2str(2^g(i,j)),' -d ',num2str(d(k))];
                        cg(i,j) = svmtrain(TrainLabel,TrainSet,cmd);
                        if (cg(i,j) > bestacc)
                            bestacc = cg(i,j);
                            bestc = 2^c(i,j);
                            bestg = 2^g(i,j);
                            bestd = d(k);
                        end
                        if abs(cg(i,j) - bestacc) <= eps && bestc > 2^c(i,j)
                            bestacc = cg(i,j);
                            bestc = 2^c(i,j);
                            bestg = 2^g(i,j);
                            bestd = d(k);
                        end
                    end
                end
            end
            cmd = ['-c ',num2str(bestc),' -t ', num2str(t), ' -g ', num2str(bestg),' -d ',num2str(bestd)];
            Model = svmtrain(TrainLabel,TrainSet,cmd);
    end
end
                      




