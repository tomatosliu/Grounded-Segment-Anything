import cv2
import numpy as np
import os
import datetime
import argparse
from tqdm import tqdm
import json
from PIL import Image

import torch
import torchvision
import supervision as sv
import torchvision.transforms as TS

from groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor
from ram.models import ram
from ram import inference_ram

import cv2
import numpy as np
import json
from pycocotools import mask as maskUtils
from typing import Union

CAMERA_MASK = [cv2.imread('oasa_assets/new_camera_mask.png', cv2.IMREAD_GRAYSCALE),
               cv2.imread('oasa_assets/old_camera_mask.png', cv2.IMREAD_GRAYSCALE)]
with open("oasa_assets/label/ram_tag_list.txt") as f:
    CLASSES = [name.strip('\n') for name in f.readlines()]
print("Class num:", len(CLASSES))

with open("oasa_assets/label/ram_reserved_tags.txt") as f:
    RESERVED_TAGS = [name.strip('\n') for name in f.readlines()]

with open("oasa_assets/label/ram_discard_tags.txt") as f:
    DISCARD_TAGS = {name.strip('\n') for name in f.readlines()}


def postprocess_tag(tags):
    res_tags = set(tags)
    res_tags -= set(RESERVED_TAGS)
    res_tags -= DISCARD_TAGS
    return RESERVED_TAGS + list(res_tags)


def get_ram_class_id(cls_name):
    for i, n in enumerate(CLASSES):
        if cls_name == n:
            return i


def get_camera_mask(img_fn):
    mod_time = os.path.getmtime(img_fn)
    readable_time = datetime.datetime.fromtimestamp(mod_time)
    if readable_time.year >= 2024 and readable_time.month >= 5:
        return CAMERA_MASK[0]
    else:
        return CAMERA_MASK[1]


OASA_CLASSES = ['others', 'grass', 'road', 'tree', 'person', 'base', 'treeroot', 'leaf', 'sem_08',
                'sem_09', 'sem_10', 'sem_11', 'sem_12', 'sem_13', 'sem_14', 'sem_15', 'sem_16', 'sem_17', 'sem_18']
CLASS_RAM_TO_OASA = json.load(open("oasa_assets/label/ram_to_oasa.json"))

# GroundingDINO config and checkpoint
GROUNDING_DINO_CONFIG_PATH = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT_PATH = "./groundingdino_swint_ogc.pth"

# Segment-Anything checkpoint
SAM_ENCODER_VERSION = "vit_h"
SAM_CHECKPOINT_PATH = "./sam_vit_h_4b8939.pth"

RAM_CHECKPOINT_PATH = "./ram_swin_large_14m.pth"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Building GroundingDINO inference model
grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH,
                             model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)

# Building SAM Model and SAM Predictor
sam = sam_model_registry[SAM_ENCODER_VERSION](
    checkpoint=SAM_CHECKPOINT_PATH)
sam.to(device=DEVICE)
sam_predictor = SamPredictor(sam)

ram_model = ram(pretrained=RAM_CHECKPOINT_PATH,
                image_size=384,
                vit='swin_l')
ram_model.eval()
ram_model = ram_model.to(DEVICE)

# Predict classes and hyper-param for GroundingDINO
BOX_THRESHOLD = 0.3
TEXT_THRESHOLD = 0.3
NMS_THRESHOLD = 0.7

normalize = TS.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
transform = TS.Compose(
    [
        TS.Resize((384, 384)),
        TS.ToTensor(),
        normalize
    ]
)


def ClassToOasaLabelID(cls):
    if cls in CLASS_RAM_TO_OASA['grass']:
        return 1
    if cls in CLASS_RAM_TO_OASA['road']:
        return 2
    if cls in CLASS_RAM_TO_OASA['tree']:
        return 3
    if cls in CLASS_RAM_TO_OASA['person']:
        return 4
    if cls in CLASS_RAM_TO_OASA['leaf']:
        return 7
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
    vis_dir = os.path.join('/tmp', os.path.basename(image_dir) + '_vis')
    imglist_txt = os.path.join(image_dir, "all.txt")
    anno_json = os.path.join(image_dir, "tmp_annotations.json")
    sam_anno_json = os.path.join(image_dir, "ram_dino_sam_annotations.json")
    os.makedirs(vis_dir, exist_ok=True)

    coco = CocoWriter(OASA_CLASSES)
    sam_coco = CocoWriter(CLASSES)
    imglist_f = open(imglist_txt, 'w')
    for img_path in tqdm(os.listdir(image_dir)):
        if not img_path.endswith('.jpg'):
            continue
        imglist_f.write(img_path + '\n')

        # load image
        image_path = os.path.join(image_dir, img_path)
        image = cv2.imread(image_path)

        # detect tags
        image_pillow = Image.fromarray(
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        image_pillow = image_pillow.resize((384, 384))
        image_pillow = transform(image_pillow).unsqueeze(0).to(DEVICE)
        ram_tag_res = inference_ram(image_pillow, ram_model)
        image_pillow = image_pillow.cpu()
        del image_pillow
        auto_tags = ram_tag_res[0].split(" | ")
        auto_tags = postprocess_tag(auto_tags)

        # detect objects
        detections = grounding_dino_model.predict_with_classes(
            image=image,
            classes=auto_tags,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD
        )
        nms_idx = torchvision.ops.nms(
            torch.from_numpy(detections.xyxy),
            torch.from_numpy(detections.confidence),
            NMS_THRESHOLD
        ).numpy().tolist()
        detections.xyxy = detections.xyxy[nms_idx]
        detections.confidence = detections.confidence[nms_idx]
        detections.class_id = detections.class_id[nms_idx]

        # segment with detected boxes
        def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
            sam_predictor.set_image(image)
            result_masks = []
            for box in xyxy:
                masks, scores, logits = sam_predictor.predict(
                    box=box,
                    multimask_output=True
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

        # mask camera frame
        camera_mask = get_camera_mask(image_path)
        camera_mask = cv2.resize(
            camera_mask, (image.shape[1], image.shape[0]), cv2.INTER_NEAREST)
        for i, m in enumerate(detections.mask):
            detections.mask[i][camera_mask == 0] = 0

        # occupied by priority
        occupied_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        coco.add_image(img_path, image.shape[1], image.shape[0])
        for oasa_ind in list(range(1, len(OASA_CLASSES))) + [0]:
            class_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            for mask_ind, (xyxy, mask, confidence, class_id, tracker_id, _) in enumerate(detections):
                if (ClassToOasaLabelID(auto_tags[class_id])) == oasa_ind:
                    mask[occupied_mask != 0] = 0
                    occupied_mask[mask != 0] = 1
                    detections.mask[mask_ind] = mask
                    class_mask[mask > 0] = 1
            coco.add_anno_mask(img_path, class_mask, oasa_ind)
        coco.save(anno_json)

        # Visualization
        box_annotator = sv.BoundingBoxAnnotator(
            thickness=1)
        mask_annotator = sv.MaskAnnotator()
        label_annotator = sv.LabelAnnotator(
            text_scale=0.3, text_padding=3, text_position=sv.Position.CENTER)
        labels = [
            f"{auto_tags[class_id]} {confidence:0.2f}"
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
        sam_coco.add_image(
            img_path, annotated_image.shape[1], annotated_image.shape[0])
        for xyxy, mask, confidence, class_id, tracker_id, _ in detections:
            sam_coco.add_anno_mask(
                img_path, mask, auto_tags[class_id])
        sam_coco.save(sam_anno_json)
    imglist_f.close()


def find_directories_with_jpg(root_dir):
    directories_with_jpg = set()
    # Walk through all directories and files in root_dir
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Check if any '.jpg' files are in the current directory
        if any(fname.endswith('.jpg') for fname in filenames) \
            and not any(fname.endswith('.mcap.jpg') for fname in filenames) \
                and not any(fname.endswith('_rosbag.jpg') for fname in filenames):
            directories_with_jpg.add(dirpath)
    return directories_with_jpg


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('img_dir', nargs='+')
    args = parser.parse_args()
    print('process:\n', args.img_dir)

    # Specify the root directory to search
    for big_d in args.img_dir:
        jpg_directories = find_directories_with_jpg(big_d)
        for d in tqdm(jpg_directories):
            print("labeling", d)
            annotate_one_directory(d)
