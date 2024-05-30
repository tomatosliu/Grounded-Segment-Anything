import cv2
import numpy as np
import os
import argparse
from tqdm import tqdm

import torch
import torchvision
import supervision as sv

from groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator

import cv2
import numpy as np
import json
from pycocotools import mask as maskUtils
from typing import Union

CLASSES = ['grass', 'road', 'tree', 'person',
           'building', 'shrub', 'bicycle', 'car', 'cat', 'dog',
           'fence', 'wall', 'floor', 'pavement', 'rock', 'table', 'chair',
           'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
           'stop sign', 'bench', 'bird', 'horse', 'sheep', 'cow', 'backpack', 'umbrella', 'handbag',
           'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
           'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
           'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
           'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'couch', 'potted plant', 'bed',
           'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
           'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
           'hair drier', 'toothbrush', 'blanket', 'bridge', 'cardboard', 'counter', 'curtain',
           'door-stuff', 'flower', 'fruit', 'gravel', 'house', 'light', 'mirror-stuff',
           'net', 'pillow', 'platform', 'playingfield', 'railroad', 'river', 'roof', 'sand', 'sea',
           'shelf', 'snow', 'stairs', 'tent', 'towel',
           'water', 'window', 'ceiling', 'sky', 'cabinet',
           'mountain', 'dirt', 'paper', 'food', 'rug', 'trampoline', 'pipes']

COLOR_LOOKUP = np.full((len(CLASSES), 3), (255, 0, 0), dtype=np.uint8)
COLOR_LOOKUP[:4, :] = [[0, 255, 0], [0, 0, 255], [0, 125, 0], [125, 0, 0]]

OASA_CLASSES = ['others', 'grass', 'road', 'tree', 'person', 'base', 'treeroot', 'leaf', 'sem_08',
                'sem_09', 'sem_10', 'sem_11', 'sem_12', 'sem_13', 'sem_14', 'sem_15', 'sem_16', 'sem_17', 'sem_18']


def ClassToOasaLabelID(cls):
    GRASS_CLASSES = {'grass'}
    ROAD_CLASSES = {'road', 'floor', 'pavement',
                    'gravel', 'sand', 'dirt', 'rug'}
    TREE_CLASSES = {'tree', 'shrub', 'potted plant', 'flower'}
    PERSON_CLASSES = {'person', 'cat', 'dog'}
    BASE_CLASSES = {'base'}
    TREEROOT_CLASSES = {'treeroot'}
    LEAF_CLASSES = {'leaf'}
    # FENCE_CLASSES = {'fence'}

    if cls in GRASS_CLASSES:
        return 1
    if cls in ROAD_CLASSES:
        return 2
    if cls in TREE_CLASSES:
        return 3
    if cls in PERSON_CLASSES:
        return 4
    if cls in BASE_CLASSES:
        return 5
    if cls in TREEROOT_CLASSES:
        return 6
    if cls in LEAF_CLASSES:
        return 7
    # if cls in FENCE_CLASSES:
    #     return 8
    return 0  # others


class CocoWriter:
    def __init__(self, categories) -> None:
        annotations = {}
        annotations['info'] = {'description': {}}
        annotations['licenses'] = ['']
        annotations['images'] = []
        annotations['annotations'] = []
        annotations['categories'] = []
        self.map_categories = {}
        for i, name in enumerate(categories):
            annotations['categories'].append({})
            annotations['categories'][i]['id'] = i
            annotations['categories'][i]['name'] = name
            annotations['categories'][i]['supercategory'] = ''
            self.map_categories[name] = i
        self.annotations = annotations
        self.map_imgid = {}

    def add_image(self, imgfile, imw, imh) -> int:
        image_id = len(self.annotations['images'])
        imginfo = {}
        imginfo['id'] = image_id
        imginfo['width'] = imw
        imginfo['height'] = imh
        imginfo['file_name'] = imgfile
        self.annotations['images'].append(imginfo)
        self.map_imgid[imgfile] = image_id
        return image_id

    def add_pano_result(self, imgid: Union[str, int], panno_seg: np.ndarray, instanceid2category: dict, auto_resize=False) -> None:
        for id, category in instanceid2category.items():
            self.add_anno_mask(imgid, panno_seg == id,
                               category, auto_resize=auto_resize)

    def add_seg_result(self, imgid: Union[str, int], seg_label: np.ndarray, labelid2category: dict = None, ignore_label=255, auto_resize=False) -> None:
        for labelid in np.unique(seg_label):
            if labelid == ignore_label:
                continue
            catogery_id = labelid
            if not labelid2category is None:
                catogery_id = self.map_categories[labelid2category[labelid]]
            mask = seg_label == labelid
            self.add_anno_mask(imgid, mask, catogery_id, auto_resize)

    def add_anno_mask(self, img_id: Union[str, int], binary_mask: np.ndarray, category_id: Union[str, int], auto_resize=False) -> None:
        if type(category_id) is str:
            category_id = self.map_categories[category_id]
        assert category_id < len(self.annotations['categories'])
        if type(img_id) is str:
            img_id = self.map_imgid[img_id]
        assert img_id < len(self.annotations['images'])
        imw, imh = self.annotations['images'][img_id]['width'], self.annotations['images'][img_id]['height']
        if auto_resize:
            binary_mask = cv2.resize(binary_mask.astype(
                np.uint8), (imw, imh), interpolation=cv2.INTER_NEAREST)
        assert imh == binary_mask.shape[
            0], f'different mask size {binary_mask.shape} with image {imw},{imh}, check if bug or use auto_resize=True'
        assert imw == binary_mask.shape[
            1], f'different mask size {binary_mask.shape} with image {imw},{imh}, check if bug or use auto_resize=True'
        rle = maskUtils.encode(np.asfortranarray(binary_mask))
        rle['counts'] = rle['counts'].decode()
        y, x = np.nonzero(binary_mask)
        if len(x) == 0 or len(y) == 0:
            return
        anno = {}
        anno['id'] = len(self.annotations['annotations'])
        anno['area'] = len(x)
        anno['bbox'] = [int(i) for i in [x.min(), y.min(),
                                         x.max()-x.min(), y.max()-y.min()]]
        anno['image_id'] = int(img_id)
        anno['segmentation'] = rle
        anno['category_id'] = int(category_id)
        anno['iscrowd'] = 0
        self.annotations['annotations'].append(anno)

    def save(self, savefilename) -> None:
        with open(savefilename, "w", encoding='utf-8') as f:
            f.write(json.dumps(self.annotations))


def annotate_one_directory(image_dir):
    assert os.path.exists(image_dir)
    vis_dir = image_dir + '_vis'
    label_dir = os.path.join(image_dir, 'label')
    imglist_txt = os.path.join(label_dir, "all.txt")
    anno_json = os.path.join(label_dir, "annotations.json")
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # GroundingDINO config and checkpoint
    GROUNDING_DINO_CONFIG_PATH = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    GROUNDING_DINO_CHECKPOINT_PATH = "./groundingdino_swint_ogc.pth"

    # Segment-Anything checkpoint
    SAM_ENCODER_VERSION = "vit_h"
    SAM_CHECKPOINT_PATH = "./sam_vit_h_4b8939.pth"

    # Building GroundingDINO inference model
    grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH,
                                 model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)

    # Building SAM Model and SAM Predictor
    sam = sam_model_registry[SAM_ENCODER_VERSION](
        checkpoint=SAM_CHECKPOINT_PATH)
    sam.to(device=DEVICE)
    sam_predictor = SamPredictor(sam)

    # Predict classes and hyper-param for GroundingDINO
    BOX_THRESHOLD = 0.2
    TEXT_THRESHOLD = 0.2
    NMS_THRESHOLD = 0.7

    coco = CocoWriter(OASA_CLASSES)
    imglist_f = open(imglist_txt, 'w')
    for img_path in tqdm(os.listdir(image_dir)):
        if not img_path.endswith('.jpg'):
            continue
        imglist_f.write(img_path + '\n')

        # load image
        image = cv2.imread(os.path.join(image_dir, img_path))
        # image = cv2.resize(
        #     image, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)

        # detect objects
        detections = grounding_dino_model.predict_with_classes(
            image=image,
            classes=CLASSES,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD
        )

        # NMS post process
        nms_idx = torchvision.ops.nms(
            torch.from_numpy(detections.xyxy),
            torch.from_numpy(detections.confidence),
            NMS_THRESHOLD
        ).numpy().tolist()

        detections.xyxy = detections.xyxy[nms_idx]
        detections.confidence = detections.confidence[nms_idx]
        detections.class_id = detections.class_id[nms_idx]


        # Prompting SAM with detected boxes

        def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
            sam_predictor.set_image(image)
            result_masks = []
            for box in xyxy:
                masks, scores, logits = sam_predictor.predict(
                    box=box,
                    multimask_output=False
                )
                index = np.argmax(scores)
                result_masks.append(masks[index])
            return np.array(result_masks)

        # convert detections to masks
        detections.mask = segment(
            sam_predictor=sam_predictor,
            image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            xyxy=detections.xyxy
        )

        # Mask fisheye edge
        camera_mask = np.zeros(image.shape[:2], np.uint8)
        cv2.circle(
            camera_mask, (image.shape[1]//2, image.shape[0]//2), image.shape[1]//2-10, 255, -1)
        for i, m in enumerate(detections.mask):
            detections.mask[i][camera_mask == 0] = 0

        occupied_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        # Other as the lowest priority
        for oasa_ind in list(range(1, len(OASA_CLASSES))) + [0]:
            for mask_ind, (xyxy, mask, confidence, class_id, tracker_id, _) in enumerate(detections):
                if (ClassToOasaLabelID(CLASSES[class_id])) == oasa_ind:
                    mask[occupied_mask != 0] = 0
                    occupied_mask[mask != 0] = 1
                    detections.mask[mask_ind] = mask
        # for mask_ind, (xyxy, mask, confidence, class_id, tracker_id, _) in enumerate(detections):
        #     if class_id != 0:
        #         detections.mask[mask_ind] = 0

        # Visualization
        colors = np.zeros((len(detections.xyxy), 3), dtype=np.uint8)
        for ind, class_id in enumerate(detections.class_id):
            colors[ind] = COLOR_LOOKUP[class_id]

        box_annotator = sv.BoundingBoxAnnotator(
            thickness=1)
        mask_annotator = sv.MaskAnnotator()
        label_annotator = sv.LabelAnnotator(
            text_scale=0.3, text_padding=3, text_position=sv.Position.CENTER)
        labels = [
            f"{CLASSES[class_id]} {confidence:0.2f}"
            for _, _, confidence, class_id, _, _
            in detections]
        annotated_image = mask_annotator.annotate(
            scene=image.copy(), detections=detections)
        annotated_image = box_annotator.annotate(
            scene=annotated_image, detections=detections)
        annotated_image = label_annotator.annotate(
            scene=annotated_image, detections=detections, labels=labels)

        # save the annotated grounded-sam image
        cv2.imwrite(os.path.join(vis_dir, img_path), annotated_image)

        # Write coco-format
        coco.add_image(
            img_path, annotated_image.shape[1], annotated_image.shape[0])
        for xyxy, mask, confidence, class_id, tracker_id, _ in detections:
            coco.add_anno_mask(
                img_path, mask, ClassToOasaLabelID(CLASSES[class_id]))
        coco.save(anno_json)
    imglist_f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('img_dir', nargs='+')
    args = parser.parse_args()
    print('process:\n', args.img_dir)
    for d in args.img_dir:
        print("labeling", d)
        annotate_one_directory(d)
