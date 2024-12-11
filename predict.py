from dataclasses import dataclass
import json
from typing import Any, List, Dict, Optional, Union, Tuple
from cog import BasePredictor
import torch
import requests
import numpy as np
from PIL import Image
from transformers import AutoModelForMaskGeneration, AutoProcessor, pipeline

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class BoundingBox:
    def __init__(self, xmin: int, ymin: int, xmax: int, ymax: int):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax

    @property
    def xyxy(self) -> List[float]:
        return [self.xmin, self.ymin, self.xmax, self.ymax]

@dataclass
class DetectionResult:
    def __init__(self, score: float, label: str, box: BoundingBox, mask: Optional[np.array] = None):
        self.score = score
        self.label = label
        self.box = box
        self.mask = mask

    def to_dict(self) -> Dict:
        return {
            'score': self.score,
            'label': self.label,
            'box': {
                'xmin': self.box.xmin,
                'ymin': self.box.ymin,
                'xmax': self.box.xmax,
                'ymax': self.box.ymax
            },
            'mask': self.mask.tolist() if self.mask is not None else None
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

def load_image(image_str: str) -> Image.Image:
    if image_str.startswith("http"):
        image = Image.open(requests.get(image_str, stream=True).raw).convert("RGB")
    else:
        image = Image.open(image_str).convert("RGB")
    return image

def get_boxes(results: List[DetectionResult]) -> List[List[List[float]]]:
    return [[result.box.xyxy for result in results]]

def list_to_json(detection_results: List[DetectionResult]) -> str:
    return json.dumps([result.to_dict() for result in detection_results], indent=2)

class Predictor(BasePredictor):

    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.threshold = 0.3
        self.polygon_refinement = True

        # Load Grounding Dino model
        self.gd_model = pipeline(
            model="IDEA-Research/grounding-dino-tiny", 
            task="zero-shot-object-detection", 
            device=DEVICE
        )

        # Load SAM
        self.sam_segmentator = AutoModelForMaskGeneration.from_pretrained("facebook/sam-vit-base").to(DEVICE)
        self.sam_processor = AutoProcessor.from_pretrained("facebook/sam-vit-base")

    def detect(
        self,
        image: Image.Image,
        labels: List[str],
        threshold: float = 0.3,
    ) -> List[DetectionResult]:
        """
        Use Grounding DINO to detect a set of labels in an image in a zero-shot fashion.
        """
        labels = [label if label.endswith(".") else label + "." for label in labels]
        results = self.gd_model(image, candidate_labels=labels, threshold=threshold)
        detections = [
            DetectionResult(
                score=result['score'],
                label=result['label'],
                box=BoundingBox(
                    xmin=result['box']['xmin'],
                    ymin=result['box']['ymin'],
                    xmax=result['box']['xmax'],
                    ymax=result['box']['ymax']
                )
            ) for result in results
        ]
        print("Grounding DINO done!")
        return detections

    def segment(
        self,
        image: Image.Image,
        detection_results: List[DetectionResult],
    ) -> List[DetectionResult]:
        """
        Use Segment Anything (SAM) to generate masks given an image + a set of bounding boxes.
        """
        boxes = get_boxes(detection_results)
        inputs = self.sam_processor(images=image, input_boxes=boxes, return_tensors="pt").to(DEVICE)

        outputs = self.sam_segmentator(**inputs)
        masks = self.sam_processor.post_process_masks(
            masks=outputs.pred_masks,
            original_sizes=inputs.original_sizes,
            reshaped_input_sizes=inputs.reshaped_input_sizes
        )[0]

        for detection_result, mask in zip(detection_results, masks):
            detection_result.mask = mask

        print("SAM done!")
        return detection_results

    def grounded_segmentation(
        self,
        image: Union[Image.Image, str],
        labels: List[str],
        threshold: float = 0.3,
    ) -> Tuple[np.ndarray, List[DetectionResult]]:
        if isinstance(image, str):
            image = load_image(image)

        detections = self.detect(image, labels, threshold)
        detections = self.segment(image, detections)

        return np.array(image), detections

    def predict(
        self,
        image_url: str,
        text_prompts: str,
    ) -> str:
        """Run a single prediction on the model"""
        text_prompts_list = [prompt.strip() for prompt in text_prompts.split(',') if prompt.strip()]
        if not text_prompts_list:
            raise ValueError("No valid text prompts provided.")

        _, detections = self.grounded_segmentation(image_url, text_prompts_list)
        return list_to_json(detections)
