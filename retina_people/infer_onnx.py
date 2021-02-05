import os
import torch
import onnx
import onnxruntime

onnx_path = "retinanet/checkpoints/exported.onnx"

sess = onnxruntime.InferenceSession(onnx_path)

dummy_ip = torch.randn(1,3, 540, 960).numpy()
input_names = ['input_1']
output_names = ['score_1', 'score_2', 'score_3', 'score_4', 'score_5',
                'box_1', 'box_2', 'box_3', 'box_4', 'box_5']

for op in sess.run(output_names, input_feed={sess.get_inputs()[0].name: dummy_ip}):
    print(op.shape)