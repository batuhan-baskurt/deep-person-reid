import torchreid

torchreid.models.show_avai_models()

# The num_classes variable doesn't matter as the last linear classifier is not used during testing.
model = torchreid.models.build_model(name='osnet_x0_25', num_classes=1000)
torchreid.utils.load_pretrained_weights(model, "model.pth.tar-100") 

model.eval()

from torch.autograd import Variable
import torch
import onnx

# An example input you would normally provide to your model's forward() method.
input = torch.ones(1, 3, 256, 256)
raw_output = model(input)
print(raw_output.size())
torch.onnx.export(model, input, 'osnet_x0_25_apm_cosine.onnx', verbose=False, export_params=True)

print("-------------------------check model---------------------------------------\n")

try:
    onnx_model = onnx.load("osnet_x0_25_apm_cosine.onnx")
    onnx.checker.check_model(onnx_model)
    graph_output = onnx.helper.printable_graph(onnx_model.graph)
    with open("graph_output.txt", mode="w") as fout:
        fout.write(graph_output)	
except:
    print("Something went wrong")
	
	
import onnxruntime
import numpy as np

ort_session = onnxruntime.InferenceSession("osnet_x0_25_apm_cosine.onnx")

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(input)}
ort_outs = ort_session.run(None, ort_inputs)	
print(np.shape(ort_outs))

# compare ONNX Runtime and PyTorch results
np.testing.assert_allclose(to_numpy(raw_output), ort_outs[0], rtol=1e-03, atol=1e-05)

print("Exported model has been tested with ONNXRuntime, and the result looks good!")
