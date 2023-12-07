import os
import io
import json
import time
import torch
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import onnxruntime
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS


app = Flask(__name__)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_onnx_model():
    global ort_session
    ort_session = onnxruntime.InferenceSession(r'Resnet_6.23_-20.onnx')

load_onnx_model()


def transform_text(text):

    tensor = torch.tensor(text)
    return tensor

def get_prediction(img_bytes):
    # 记录该帧开始处理的时间
    start_time = time.time()
    img_bytes = img_bytes.decode('utf-8')
    img_bytes = img_bytes[0:-2]
    img_bytes = img_bytes.split(" ")
    img_bytes = np.array(img_bytes, dtype='float32')
    img_bytes = img_bytes.reshape(1, 1, 30603)
    tensor = transform_text(text=img_bytes)
    ort_inputs = {'input': tensor.numpy()}
    pred_logits = ort_session.run(['output'], ort_inputs)[0]
    pred_logits = torch.from_numpy(pred_logits)

    pred_softmax = F.softmax(pred_logits, dim=1)  # 对 logit 分数做 softmax 运算
    top_n = torch.topk(pred_softmax, 16)

    pred_ids = top_n.indices[0].cpu().detach().numpy()  # 将索引转换为NumPy数组，并分离梯度
    confs = top_n.values[0].cpu().detach().numpy() * 100  # 将置信度转换为NumPy数组，并分离梯度

    # 记录该帧处理完毕的时间
    end_time = time.time()
    # 计算每秒处理图像帧数FPS
    FPS = 1 / (end_time - start_time)

    # 载入类别和对应 ID
    #idx_to_labels = np.load('idx_to_labels1.npy', allow_pickle=True).item()
    idx_to_labels = np.load('dict_transform.npy', allow_pickle=True).item()
    results = []  # 用于存储结果的列表
    for i in range(16):
        class_name = idx_to_labels[pred_ids[i]]  # 获取类别名称
        confidence = confs[i]  # 获取置信度
        text = 'key为：{:<6}的概率为 {:>.3f}'.format(class_name,confidence)
        results.append(text)  # 将结果添加到列表中


    return results, FPS  # 返回包含类别和置信度的列表

@app.route("/predict", methods=["POST"])
@torch.no_grad()
def predict():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()

        class_info, FPS = get_prediction(img_bytes=img_bytes)
        response_data = {'class_info': class_info, 'FPS': FPS}
        return jsonify(response_data)

if __name__ == '__main__':
    #server = pywsgi.WSGIServer(('127.0.0.1', 5000), app)
    #server.serve_forever()
    app.run()