# gbt-svm
In this repo, a MATLAB toolbox for support vector machine (SVM) is provided. In order to handle imbalanced sample set and classification for multiple types, two strategies namely granularization and tree-structure are applied.

## Requirements
To use the toolbox, the famous [libsvm](https://www.csie.ntu.edu.tw/~cjlin/libsvm/) is required. Note that, to config this package, either the `VC++` for windows or the `g++` for linux should be installed.

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