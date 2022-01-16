from glob import glob
import os
import json

import numpy as np
import torch
from detectron2.utils.visualizer import ColorMode
from detectron2.engine.defaults import DefaultPredictor
from detectron2.data.detection_utils import read_image

from adet.config import get_cfg

CTLABELS = [' ','!','"','#','$','%','&','\'','(',')','*','+',',','-','.','/','0','1','2','3','4','5','6','7','8','9',':',';','<','=','>','?','@','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','[','\\',']','^','_','`','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','{','|','}','~']


def bezier_to_poly(bezier):
    # bezier to polygon
    u = np.linspace(0, 1, 20)
    bezier = bezier.reshape(2, 4, 2).transpose(0, 2, 1).reshape(4, 4)
    points = np.outer((1 - u) ** 3, bezier[:, 0]) \
        + np.outer(3 * u * ((1 - u) ** 2), bezier[:, 1]) \
        + np.outer(3 * (u ** 2) * (1 - u), bezier[:, 2]) \
        + np.outer(u ** 3, bezier[:, 3])
    points = np.concatenate((points[:, :2], points[:, 2:]), axis=0)
    return points


def decode_recognition(rec, voc_size):
    s = ''
    for c in rec:
        c = int(c)
        if c < voc_size - 1:
            if voc_size == 96:
                s += CTLABELS[c]
            else:
                s += str(chr(CTLABELS[c]))
        elif c == voc_size -1:
            s += u'口'
    return s


if __name__ == '__main__':
    config_file = '/content/AdelaiDet/configs/BAText/TotalText/v2_attn_R_50.yaml'
    model_path = '/content/model_v2_totaltext.pth'
    test_frames_dir = '/content/yvt/test/frames'
    output_file = '/gdrive/MyDrive/video-text-spotting/ABCNetV2/evaluation/yvt-text_results.json'

    # setup configuration
    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.DEVICE = 'cuda'
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.3
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3
    cfg.MODEL.FCOS.INFERENCE_TH_TEST = 0.3
    cfg.MODEL.MEInst.INFERENCE_TH_TEST = 0.3
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = 0.3
    cfg.freeze()

    voc_size = cfg.MODEL.BATEXT.VOC_SIZE
    use_customer_dictionary = cfg.MODEL.BATEXT.CUSTOM_DICT
    instance_mode = ColorMode.IMAGE
    vis_text = cfg.MODEL.ROI_HEADS.NAME == "TextHead"

    # Save prediction results in json file
    predictor = DefaultPredictor(cfg)
    image_files = glob(os.path.join(test_frames_dir, '*.jpg'))
    prediction_data = []
    for image_path in image_files:
        image_id = image_path.rsplit('/', 1)[1].rsplit('.', 1)[0]
        img = read_image(image_path, format="BGR")
        predictions = predictor(img)
        instances = predictions["instances"].to(torch.device('cpu'))
        if instances:
            beziers = instances.beziers.numpy()
            scores = instances.scores.tolist()
            recs = instances.recs
            for bezier, rec, score in zip(beziers, recs, scores):
                polygon = bezier_to_poly(bezier)
                text = decode_recognition(rec, voc_size)
                prediction_data.append({
                    "image_id": image_id,
                    "category_id": 1,
                    "polys": polygon.tolist(),
                    "rec": str(text),
                    "score": float(score)
                })
    with open(output_file, 'w') as fp:
        json.dump(prediction_data, fp)
