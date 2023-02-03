#import argparse
import os
import os.path as osp
import time
from typing import Dict, Any, Optional
import cv2
import torch
from tracker.byte_tracker import BYTETracker

from yolox import get_detector
from yolox.utils.preprocessing import preproc
from yolox.utils.model_utils import fuse_model
from yolox.utils.boxes import postprocess
from yolox.configs.base_config import YoloXTrackerConfig

try:
    from torch2trt import TRTModule
except ImportError:
    print("[Warning] torch2trt not found. Skipping acceleration")
    pass

# Predictor(model, exp, trt_file, decoder, args["device"], args["fp16"])
class ByteTrackWrapper(object):
    def __init__(
        self,
        model,
        exp: YoloXTrackerConfig,
        trt_file=None,
        decoder=None
    ):
        self.model = model
        self.decoder = decoder
        self.exp = exp
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.input_size
        self.device = exp.get_torch_device()
        self.fp16 = exp.fp16
        if trt_file is not None:
            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))
            x = torch.ones((1, 3, exp.test_size[0], exp.test_size[1]), device=self.device)
            self.model(x)
            self.model = model_trt

        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

        ### tracker

        self.tracker = BYTETracker(exp, frame_rate=30)

    def inference(self, img, batchsize:int=1):
        # detector
        outputs, img_info = self._inference_model(img)
        if len(outputs) < 0:
            return None
        if batchsize != 1:
            raise NotImplementedError("todo: adapt code for batchsize>1")

        if outputs[0] is None:
            return None
        outputs = outputs[0]  # batch size

        # tracker
        # online_targets = self.tracker.update(outputs[0], [img_info['height'], img_info['width']], self.test_size)
        online_targets = self.tracker.update(outputs, [img_info['height'], img_info['width']], self.test_size)
        
        online_tlwhs = []
        online_ids = []
        online_scores = []

        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id

            vertical = tlwh[2] / tlwh[3] > self.exp.aspect_ratio_thresh # 0 is x, 1 is y, 2 is width, 3 is length. tlwh is the bounding box
            # if tlwh[2] * tlwh[3] > self.exp.min_box_area and not vertical:
            if tlwh[2] * tlwh[3] > self.exp.min_box_area:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
                #print(tid, "\n")
                online_scores.append(t.score)
                 
        out = {
            "tlws": online_tlwhs,
            "ids": online_ids,
            "scores": online_scores
        }      

        return online_tlwhs, online_ids, online_scores
        


    def _inference_model(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = osp.basename(img) # stampa ultima parte indirizzo (os.path.basename)
            img = cv2.imread(img) # legge l'immagine -> dimensione 3 (altezza, larghezza, channels)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2] 
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        img, ratio = preproc(img, self.test_size, self.rgb_means, self.std) # applica padding
        img_info["ratio"] = ratio
        # from_numpy crea un tensore dall'immagine, unsqueeze aumenta la dimensione
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)

        if self.fp16:
            img = img.half()  # to FP16

        with torch.no_grad(): # loop in cui i tensori hanno il calcolo del gradiente disabilitato
            # timer.tic()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())

            outputs = postprocess(
                outputs, self.num_classes, self.confthre, self.nmsthre
            )

        return outputs, img_info


# def get_bytetrack_tracker(demo, video, exp_file, trained, track_thresh):

#     args = {
#         "demo": demo,
#         "expn": None,
#         "name": None,
#         "path": video,
#         "camid": 0,
#         "save_result": True,
#         "exp_file": exp_file,
#         "ckpt": trained,
#         "device": "gpu",
#         "conf": None,
#         "nms": None,
#         "tsize": None,
#         "fps": 30,
#         "fp16": True,
#         "fuse": True,
#         "trt": False,
#         "track_thresh": track_thresh,
#         "track_buffer": 30,
#         "match_thresh": 0.8,
#         "aspect_ratio_thresh": 1.6,
#         "min_box_area": 10,
#         "mot20": False
#     }

#     exp = get_exp(args["exp_file"], args["name"]) # restituisce oggetto Exp di yolox_x_mix_det.py

#     if "experiment_name" not in args:
#         args["experiment_name"] = exp.exp_name

#     output_dir = osp.join(exp.output_dir, args["experiment_name"]) # os.path as osp e join concatena ./YOLOX_outputs con /yolox_x_mix_det 
#     os.makedirs(output_dir, exist_ok=True)

#     if args["save_result"]:
#         vis_folder = osp.join(output_dir, "track_vis") # crea il percorso ./YOLOX_outputs/yolox_x_mix_det/track_vis
#         os.makedirs(vis_folder, exist_ok=True)

#     if args["trt"]: # argomento per TensorRT model for testing
#         args["device"] = "gpu"
#     args["device"] = torch.device("cuda" if args["device"] == "gpu" else "cpu") # argomento device che usa

#     # logger.info("Args: {}".format(args))

#     if args["conf"] is not None:
#         exp.test_conf = args["conf"]
#     if args["nms"] is not None:
#         exp.nmsthre = args["nms"]
#     if args["tsize"] is not None:
#         exp.test_size = (args["tsize"], args["tsize"])

#     model = exp.get_model().to(args["device"]) # prende il modello
#     # logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
#     model.eval()
    
#     if args["trt"] == False:
#         if args["ckpt"] is None:
#             ckpt_file = osp.join(output_dir, "best_ckpt.pth.tar")
#         else:
#             ckpt_file = args["ckpt"]
#         # logger.info("loading checkpoint")
#         ckpt = torch.load(ckpt_file, map_location="cpu")
#         # load the model state dict
#         model.load_state_dict(ckpt["model"])
#         # logger.info("loaded checkpoint done.")

#     if args["fuse"]:
#         # logger.info("\tFusing model...")
#         model = fuse_model(model)

#     if args["fp16"]:
#         model = model.half()  # to FP16

#     if args["trt"]:
#         assert not args["fuse"], "TensorRT model is not support model fusing!"
#         trt_file = osp.join(output_dir, "model_trt.pth")
#         assert osp.exists(
#             trt_file
#         ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
#         model.head.decode_in_inference = False
#         decoder = model.head.decode_outputs
#         # logger.info("Using TensorRT to inference")
#     else:
#         trt_file = None
#         decoder = None

#     predictor = ByteTrackWrapper(model, exp, trt_file, decoder, args["device"], args["fp16"])
#     current_time = time.localtime()

#     return predictor, vis_folder, current_time, args, exp



def get_bytetrack_tracker(parameters: YoloXTrackerConfig, checkpoint: str, fps_expected: int = 20):


    if parameters.trt: # argomento per TensorRT model for testing
        parameters.device = "cuda"

    device = parameters.get_torch_device()

    model = get_detector(
        depth=parameters.depth,
        width=parameters.width,
        num_classes=parameters.num_classes,
    )
    
    if parameters.trt == False:
        ckpt = torch.load(checkpoint, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        # logger.info("loaded checkpoint done.")
        print("loaded checkpoint done.")

    model = model.to(device)
    model.eval()

    if parameters.fuse:
        # logger.info("\tFusing model...")
        model = fuse_model(model)

    if parameters.fp16:
        model = model.half()  # to FP16

    if parameters.trt:
        assert not parameters.fuse, "TensorRT model is not support model fusing!"
        trt_file = osp.join(os.getcwd(), "model_trt.pth")
        assert osp.exists(
            trt_file
        ), "TensorRT model is not found!\n Run python3 yolox/utils/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        print("Using tensorRT acceleration")
    else:
        trt_file = None
        decoder = None

    parameters.fps = fps_expected
    predictor = ByteTrackWrapper(model, parameters, trt_file, decoder)

    return predictor

