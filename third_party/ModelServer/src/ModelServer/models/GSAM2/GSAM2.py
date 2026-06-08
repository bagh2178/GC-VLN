import os
import sys
# Relative path from this file to Grounded-SAM-2
current_dir = os.path.dirname(os.path.abspath(__file__))
GSAM2_root_dir = os.path.abspath(os.path.join(current_dir, '../../../../../Grounded-SAM-2/'))
sys.path.append(GSAM2_root_dir)
from typing import Tuple
import numpy as np
from PIL import Image
import torch

from torchvision.ops import box_convert
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from grounding_dino.groundingdino.util.inference import load_model, predict
import grounding_dino.groundingdino.datasets.transforms as T


def load_image(image_source) -> Tuple[np.array, torch.Tensor]:
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_source = Image.fromarray(image_source)
    image = np.asarray(image_source)
    image_transformed, _ = transform(image_source, None)
    return image, image_transformed


class GSAM2():
    def __init__(self):
        self.sam2_model = build_sam2(
            'configs/sam2.1/sam2.1_hiera_l.yaml', 
            os.path.join(GSAM2_root_dir, 'checkpoints/sam2.1_hiera_large.pt'), 
            device='cuda'
        )
        self.sam2_predictor = SAM2ImagePredictor(self.sam2_model)
        self.grounding_model = load_model(
            model_config_path = os.path.join(GSAM2_root_dir, 'grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py'),
            model_checkpoint_path = os.path.join(GSAM2_root_dir, 'gdino_checkpoints/groundingdino_swint_ogc.pth'),
            device='cuda'
        )

    def __call__(self, image_rgb_byte_io, text_prompt):
        image_rgb_byte_io.seek(0)
        image_rgb = Image.open(image_rgb_byte_io)
        image_rgb = np.array(image_rgb)

        with torch.no_grad():
            masks, xyxy, captions, masks_conf, captions_conf = \
                self.get_sam_segmentation_dense(image_rgb, text_prompt)
        return masks, xyxy, captions, masks_conf, captions_conf

    def get_sam_segmentation_dense(
        self, image: np.ndarray, text_prompt: str
    ) -> tuple:
        '''get the segmentation and caption
        '''
        image_source, image_trans = load_image(image)

        self.sam2_predictor.set_image(image_source.copy())

        boxes, confidences, labels = predict(
            model=self.grounding_model, 
            image=image_trans, 
            caption=text_prompt, 
            box_threshold=0.3, 
            text_threshold=0.25
        )
        if len(boxes) == 0:
            return [],[],[],[],[]
        # process the box prompt for SAM 2
        h, w, _ = image_source.shape
        boxes = boxes * torch.Tensor([w, h, w, h])
        input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

        masks, scores, _ = self.sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )

        if masks.ndim == 4:
            masks = masks.squeeze(1).astype(np.int32)
        confidences = confidences.numpy().tolist()

        return masks, input_boxes, labels, scores, confidences 