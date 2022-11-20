import argparse

import torch.onnx
from network import EmbeddedYolo

if __name__ == "__main__":
    # gem. https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html
    # zum umwandeln:  --weights checkpoint\training_synth\epoch-17.pt
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)  # 500200 batches at bs 16, 117263 COCO images = 273 epochs
    parser.add_argument('--batch-size', type=int, default=64)  # effective bs = batch_size * accumulate = 16 * 4 = 64
    parser.add_argument('--data', type=str, default='data/radar.data', help='*.data path')
    parser.add_argument('--resume', action='store_true', help='resume training from last.pt')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--l2', type=float, default=0.0001)
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    # parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--test', action='store_true', help='evaluate test data')
    parser.add_argument('--load_weights', action='store_true', help='load weights')
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--weights', type=str, default='checkpoint/yolov4-tiny.weights', help='initial weights path')
    parser.add_argument('--name', default='', help='renames results.txt to results_name.txt if supplied')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--adam', action='store_true', help='use adam optimizer')
    opt = parser.parse_args()
    opt.n_class = 5
    opt.conf_threshold = 0.05  # object confidence threshold
    opt.top_n = 1000
    opt.nms_threshold = 0.6  # iou threshold
    opt.post_top_n = 100
    opt.min_size = 0
    device = 'cuda' if torch.cuda.is_available() == 1 else 'cpu'

    model = EmbeddedYolo(opt)
    model.load_state_dict(torch.load(opt.weights)['model'])
    # model = model.to(device)
    model.eval()

    # Input to the model
    batch_size = 1
    # x = torch.randn(batch_size, 3, 416, 416, requires_grad=True)
    # torch_out = model(x)

    # Export the model
    # funktioniert nur wenn Netzwerk wie folgt abgeändert wird:
    # model muss location, cls_pred, box_pred, center_pred returnen anstatt BoxTarget
    # Hardswish muss gegen ReLU ausgetauscht werden
    # torch.onnx.export(model,  # model being run
    #                   x,
    #                   # model input (or a tuple for multiple inputs)
    #                   "embeddedYolo.onnx",  # where to save the model (can be a file or file-like object)
    #                   export_params=True,  # store the trained parameter weights inside the model file
    #                   opset_version=11,  # the ONNX version to export the model to
    #                   do_constant_folding=True,  # whether to execute constant folding for optimization
    #                   input_names=['input'],  # the model's input names
    #                   output_names=['output'],  # the model's output names
    #                   dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
    #                                 'output': {0: 'batch_size'}})

    import onnx
    import numpy as np

    # Then, onnx.checker.check_model(onnx_model) will verify the model’s structure and confirm that the model has a valid
    # schema. The validity of the ONNX graph is verified by checking the model’s version, the graph’s structure, as well as
    # the nodes and their inputs and outputs.
    onnx_model = onnx.load("embeddedYoloN.onnx")
    onnx.checker.check_model(onnx_model)

    print("keine Exception heißt test bestanden.")

    # import onnxruntime
    #
    # ort_session = onnxruntime.InferenceSession("embeddedYolo.onnx")
    #
    #
    # def to_numpy(tensor):
    #     return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    #
    #
    # def to_numpy2(tup):
    #     outlist = []
    #     for l in tup:
    #         for ten in l:
    #             outlist.append(ten.detach().cpu().numpy() if ten.requires_grad else ten.cpu().numpy())
    #     return outlist
    #
    #
    # # compute ONNX Runtime output prediction
    # ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    # ort_outs = ort_session.run(None, ort_inputs)
    #
    # # compare ONNX Runtime and PyTorch results
    # torch_outs = to_numpy2(torch_out)
    # for i in range(len(torch_outs)):
    #     np.testing.assert_allclose(torch_outs[i], ort_outs[i], rtol=1e-03, atol=1e-05)
    #
    # print("Exported model has been tested with ONNXRuntime, and the result looks good!")
