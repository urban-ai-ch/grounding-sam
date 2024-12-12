# Prediction interface for Cog ⚙️
# https://cog.run/python

import tempfile
from typing import Any, Dict, List
from cog import BasePredictor, Path
import numpy as np
import requests
from torch import cuda, Tensor, from_numpy, zeros_like
from transformers import AutoModelForMaskGeneration, AutoProcessor, pipeline
from PIL import Image


class Predictor(BasePredictor):

    device = "cuda" if cuda.is_available() else "cpu"
    detector_id = "IDEA-Research/grounding-dino-tiny"
    segmenter_id = "facebook/sam-vit-base"

    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.gd_model = pipeline(model=self.detector_id, task="zero-shot-object-detection", device=self.device)
        self.segmentator = AutoModelForMaskGeneration.from_pretrained(self.segmenter_id).to(self.device)
        self.processor = AutoProcessor.from_pretrained(self.segmenter_id)

    def get_boxes(self, results: List[Dict[str, Any]]):
        boxes = []
        for result in results:
            box = result['box']
            boxes.append([box['xmin'], box['ymin'], box['xmax'], box['ymax']])
        return [boxes]
    
    def detect(
            self,
            image: Image.Image,
            labels: List[str],
            threshold: float = 0.3,
    ) -> List[Dict[str, Any]]:
        
        labels = [label if label.endswith(".") else label+"." for label in labels]
        results = self.gd_model(image,  candidate_labels=labels, threshold=threshold)

        return results

    def segment(
        self,
        image: Image.Image,
        detections: List[Dict[str, Any]], # output of grounding dino  
    ) -> Tensor:
        
        print(detections)

        boxes = self.get_boxes(detections)

        inputs = self.processor(images=image, input_boxes=boxes, return_tensors="pt").to(self.device)
        outputs = self.segmentator(**inputs)

        masks = self.processor.post_process_masks(
            masks=outputs.pred_masks,
            original_sizes=inputs.original_sizes,
            reshaped_input_sizes=inputs.reshaped_input_sizes
        )[0]

        return masks

    def grounded_segmentation(
        self,
        image_url: str,
        labels: List[str],
        threshold: float = 0.3,  
    ) -> np.ndarray:
        image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")

        detections = self.detect(image, labels, threshold)
        masks = self.segment(image, detections)

        image_tensor = from_numpy(np.array(image)).permute(2, 0, 1)

        num_images = masks.shape[0]
        result = zeros_like(image_tensor) # OR of all the masks
        for i in range(num_images):
            result = result | masks[i]
        
        ## Apply the mask
        masked_image = image_tensor * result.float()
        masked_image_np = masked_image.numpy().transpose(1, 2, 0)
        masked_image_np = np.clip(masked_image_np, 0, 255).astype(np.uint8)

        return masked_image_np

    def predict(
        self,
        image_url: str,
        labels: List[str]
    ) -> Path:
        """Run a single prediction on the model"""
        masked_image = self.grounded_segmentation(image_url, labels)

        masked_image = Image.fromarray(masked_image.astype(np.uint8))

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            masked_image.save(temp_file.name)
        
        return Path(temp_file.name)
