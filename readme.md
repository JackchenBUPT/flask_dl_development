# 关于侧信道模型深度学习模型在web上部署的方法
## 采用Flask框架进行搭建
### 原理
 load_resent_onnx.py 是创建一个本地服务器local_host,将深度学习模型Resnet_6.23_-20.onnx部署在网页上，部署过程包括加载预训练模型，读取输
 入数据，模型输出结果，结果再进行处理，最后输出给请求方。

sentpost.py 是向服务器发送一个POST，POST的内容包括一个包含30603个点的txt文件。并且接收来自服务器的返回值。

在之后的改进中，可以写一个服务器的前端，输入并且识别。
