# face_race
tensorflow+opencv 人脸识别 
## 1.Opencv 人脸追踪
  - face_race.py 写了人脸追踪代码
  简单讲是用opencv 的分类器追踪到人脸矩阵区域

## 2. opencv  人脸追踪采集数据， 尽量要采集全面
  - get_data.py 采集人脸数据代码
  采集数据为训练模型做准备

## 3. 训练模型
  - train_model.py 训练模型 简单的cnn
  利用tensorflow 搭建简单的卷积神经网络 载入数据集， sgd训练数据
  保存参数数据 model/.h5
  
## 4. 载入网络并用视频流捕捉圈出自己

  - face_predict_use_kera.py opencv视频流读取并用网络预测id
  
