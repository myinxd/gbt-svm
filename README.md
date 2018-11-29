# GBT-SVM
In this repository, a MATLAB toolbox is provided to train and test the support vector machine (SVM) based models for imbalanced and multi-type classification problem. Two classification tricks namely `granularization` and `binary-tree` have been appended to form the GBT-SVM model. Please refer to my [paper](https://github.com/myinxd/document/paper-gbtsvm.pdf) for details of the model. 

### Construction of the toolbox
Codes and scripts in the [code](https://github.com/myinxd/gbt-svm/code) folder can be used to build your granular SVM (GSVM) models, which are able to form the tree-structure classifiers. I list the name and the corresponding comment here.

| Method | Comment |
|:-------|:--------|
|getGranule| Generate balanced datasets by spitting the major class into subsets or granules.|
|myCrossSVM| Train SVM models with cross-validation and gridding, kernels are optional.|
|myGSVMpredict| Classify on new samples with the trained model.|

#### Get granules
To get balanced dataset, you may use `getGranule` as
```matlab
>>> [DataGranules] = getGranule(data,label)
```
The output `DataGranules` is a structure, which is formed as,

DataGranular
- `MajIdx`: label of the major class
- `MinIdx`: label of the minor class
- `GraNum`: number of granules
- `MinData`: data and labels of the minor-class samples
- `MajData`: data and labels of the major-class samples
- `GraIdx`: indices of the major samples that are split into the granules 

#### Train a granule with SVM
To train a SVM model, please use the `myCrossSVM` method, 
```matlab
>>> [model,normMat] = myCrossSVM(data,label,'RBF');
```
where 'RBF' is the radical basis function kernel, other kernel like linear can be selected.The output `normMat` keeps the column-wise minimum and maximum values to normalize the new samples as the same as the samples for training. **The normalization is a very important issue in machine-learning tasks.**

### Example
An [example](https://github.com/myinxd/gbt-svm/example) is demonstrated to use the GBT-SVM toolbox, please refer to [GSVM_train.m](https://github.com/myinxd/gbt-svm/example/GSVM_train.m) for details. The [sample set](https://github.com/myinxd/gbt-svm/data/SampleSet.mat) and trained [models](https://github.com/myinxd/gbt-svm/data/ModelGSVM.mat) are also included in this repo for a quick start. To evaluate the models and classify a new sample by the GSVM under a voting strategy, please refer to [GSVM_test.m](https://github.com/myinxd/gbt-svm/example/GSVM_test.m).

If you want to train the GSVM models with multi-threads, please use the codes as follows,
```matlab
parpool(2) % Two threads
parfor i = 1 : GraNum
    SubSetMaj = DataGranules.MajData.SampleSet(DataGranules.GraIdx{i},:);
    SubLabelMaj = DataGranules.MajIdx * ones(length(DataGranules.GraIdx{i}),1);
    SubTrainSet = [SubSetMaj;SubSetMin];
    SubTrainLabel = [SubLabelMaj;SubLabelMin];
    [ModelGSVM{i}.model,ModelGSVM{i}.PS] = myCrossSVM(SubTrainSet,SubTrainLabel,'RBF');
end
```

### Requirements
To use the toolbox, the [libsvm](https://www.csie.ntu.edu.tw/~cjlin/libsvm/) is required. Before using the methods in the libsvm, you may reconfigure the library with system based compiler, e.g., the `VC++` for windows, and the `g++` for linux. 

Please follow the steps to install the libsvm (my libsvm version is 3.18)
1. Unzip and put 'libsvm-3.18' into the path `xxx/matlab/toolbox`
2. Add the path of `xxx/matlab/toolbox/libsvm-3.18/matlab/` to the matlab environment, you would like to use the `Set Path` button in the `Home` panel. Remember to save the change.
3. Reconfigure the library with matlab
	a. In the `Command Window`, input `mex -setup` to choose a compiler;
    b. Check the compiler, and configure. Input `make` to begin the compilation.

## Citation
This work has been published in ICSP 2016, which can be cited as following bibtex,
```tex
@InProceedings{Ma2016,
    author = {Ma, Z. and Wang, L. and Li, W. and Xu, H. and Zhu, J.},
    title  = {X-ray astronomical point sources recognition using granular binary-tree svm},
    booktitle = {Signal Processing (ICSP), 2016 13th IEEE International Conference on},
    year = {2016},
    pages = {1021--1026},
    address = {Chengdu, China},
    month   = nov,
    organization = {IEEE},
    doi = {10.1109/ICSP.2016.7877984},
    url = {http://ieeexplore.ieee.org/document/7877984/}
    }
```
