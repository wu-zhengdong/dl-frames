# 深度学习框架

## 环境配置:python 3.6.2, pytorch 1.1, numpy, pandas, seaborn, matplotlib, scikit-learn

### 2019/07/31: add DeeplearningRegression ANN model. "./frames/DeeplearningRegression.py"    

### 2019/08/03: add DeeplearningRegression CNN model. './frames/DeeplearningRegression.py'
### 增加了梯度衰减策略，ANN，CNN 模型上限为 5 层。

### 2019/08/04: add DeeplearningRegression LSTM model. './frams/DeeplearningRegression.py'

### 2019/08/09: add DeeplearningClassification ANN model. './frams/DeeplearningClassification.py'
三大基本模型的回归框架已经完成，分类完成ANNmodel，cnn，lstm仍在继续。

### 2019/08/10: 更新 tools 里面的 create_dataset 函数，适用于 lstm model 数据集构建，example 里写了注释。add ELMRegression model (极限学习机回归模型) './frams/elm.py'

### 2019/08/09: add ElmClassification.py （极限学习机分类模型）
*修改了 is_PCA 的保存参数，改为自由填充降维方法（str）*

### 2019/08/29: 添加了三个回归预测的指标，mape，r2 rejected, rmsle。

## 使用例子
### Regression
1.'ANN Example (Regression).ipynb' ANN回归模型使用例子
2.'CNN Example (Regression).ipynb' CNN回归模型使用例子
3.'LSTM Example (Regression).ipynb' LSTM回归模型使用例子
4.'ELM Example (Regression).ipynb' ELM回归模型使用例子
### Classification
1.'ANN Example (Classification).ipynb' ANN分类模型使用例子
2.'ELM Example (Classification).ipynb' Elm分类模型使用例子


