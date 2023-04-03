# DGCNN_MindSpore

本仓库提供了一份 MindSpore 版本的 Dynamic Graph CNN for Learning on Point Clouds (DGCNN)（ https://arxiv.org/pdf/1801.07829 ）代码实现，代码框架来源于 [antao97 / dgcnn.pytorch](https://github.com/antao97/dgcnn.pytorch)

## 运行需求

- MindSpore [1.10.1 / 2.0.0-alpha](https://www.mindspore.cn/install)

- Numpy

- h5py

```shell
# 运行单机训练
bash scripts/run_standalone_train.sh
```