import os
import logging
import json
from PIL import Image
from typing import Dict, Any, List, Union, Optional

import numpy as np
import torch

import fiftyone as fo
from fiftyone import Model, SamplesMixin

from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from transformers.utils.import_utils import is_flash_attn_2_available

from .modular_isaac import IsaacProcessor

# Perceptron SDK imports for parsing
from perceptron import extract_points, strip_tags, extract_reasoning

# Local converters for Perceptron -> FiftyOne types
from .converters import (
    perceptron_to_fiftyone_detections,
    perceptron_to_fiftyone_keypoints,
    perceptron_to_fiftyone_polylines,
    box_to_center_point
)

logger = logging.getLogger(__name__)

OPERATIONS = {
    "detect": {
        "prompt": """You are a helpful assistant specializing visual grounding for object detection and counting.

Return each detection using this format:

<point_box mention="object label"> (x1,y1) (x2,y2) </point_box>

Or for multiple instances of the same type:

<collection mention="object label">
  <point_box> (x1,y1) (x2,y2) </point_box>
  <point_box> (x1,y1) (x2,y2) </point_box>
</collection>

Detect all relevant objects and provide their labels based on the user's request.""",
        "hint": "BOX"
    },
    "point": {
        "prompt": """You are a helpful assistant specializing visual grounding for pointing at and counting objects.

Return each keypoint using this format:

<point mention="point label"> (x,y) </point>

Or for multiple instances of the same type:

<collection mention="point label">
  <point> (x1,y1) </point>
  <point> (x2,y2) </point>
  <point> (x3,y3) </point>
</collection>

Point to all relevant objects and provide their labels based on the user's request.""",
        "hint": "POINT"
    },
    "ocr_detection": {
        "prompt": """You are an OCR (Optical Character Recognition) system. Accurately detect, extract, and transcribe all readable text from the image.

Return each text detection using this format, where "text content" is the actual text you detect:

<point_box mention="text content"> (x1,y1) (x2,y2) </point_box>

Detect and read the text in the image.""",
        "hint": "BOX"
    },
    "ocr_polygon": {
        "prompt": """You are an OCR (Optical Character Recognition) system. Accurately detect, extract, and transcribe all readable text from the image.

Return each text detection using this format, where "text content" is the actual text you detect:

<polygon mention="text content"> (x1,y1) (x2,y2) (x3,y3) (x4,y4) ... </polygon>

Detect and read the text in the image.""",
        "hint": "POLYGON"
    },
    "classify": {
        "prompt": """You are a helpful assistant. You specializes in comprehensive classification across any visual domain, capable of analyzing:

Unless specifically requested for single-class output, multiple relevant classifications can be provided.

Always return your response as valid JSON wrapped in ```json blocks.

```json
{
    "classifications": [
        {
            "label": "descriptive class label",
        }
    ]
}
```

The JSON should contain a list of classifications where:
- Each classification must have a 'label' field
- Labels should be descriptive strings describing what you've identified in the image, but limited to one or two word responses
- The response should be a list of classifications"""
    },
    "segment": {
        "prompt": """You are a helpful assistant specializing visual grounding for drawing polygons around objects.

Return each polygon using this format:

<polygon mention="polygon label"> (x1,y1) (x2,y2) (x3,y3) (x4,y4) ... </polygon>

Or for multiple instances of the same type:

<collection mention="polygon label">
  <polygon> (x1,y1) (x2,y2) (x3,y3) ... </polygon>
  <polygon> (x1,y1) (x2,y2) (x3,y3) ... </polygon>
</collection>

Draw polygons around all relevant objects and provide their labels based on the user's request.""",
        "hint": "POLYGON"
    },
    "ocr": {
        "prompt": """You are an OCR (Optical Character Recognition) system. Accurately detect, extract, and transcribe all readable text from the image. Preserve the original formatting as closely as possible, including:

- Line breaks and paragraphs  
- Headings and subheadings  
- Any tables, lists, bullet points, or numbered items  
- Special characters, spacing, and alignment  

Respond with 'No Text' if there is no text in the provided image."""
    },
    "vqa": {
        "prompt": "You are a visual question answering assistant. Provide a direct, concise answer."
    }
}

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

class IsaacModel(SamplesMixin, Model):
    """A FiftyOne model for running Isaac 0.1 vision tasks"""

    def __init__(
        self,
        model_path: str,
        operation: str = None,
        prompt: str = None,
        system_prompt: str = None,
        **kwargs
    ):
        self._fields = {}
        
        self.model_path = model_path
        self._custom_system_prompt = system_prompt
        self._operation = operation
        self.prompt = prompt
        
        self.device = get_device()
        logger.info(f"Using device: {self.device}")

        model_kwargs = {
            "device_map": self.device,
        }

        # Set optimizations based on CUDA device capabilities
        if self.device == "cuda" and torch.cuda.is_available():
            capability = torch.cuda.get_device_capability(self.device)
            
            # Enable flash attention if available, otherwise use sdpa
            model_kwargs["attn_implementation"] = "flash_attention_2" if is_flash_attn_2_available() else "sdpa"
            
            # Enable bfloat16 on Ampere+ GPUs (compute capability 8.0+), otherwise use float16
            model_kwargs["torch_dtype"] = torch.bfloat16 if capability[0] >= 8 else torch.float16

        # Load model and processor
        logger.info(f"Loading model from {model_path}")

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            **model_kwargs
        )
        
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            trust_remote_code=True, 
            use_fast=False
            )
        
        self.config = AutoConfig.from_pretrained(
            model_path,
            trust_remote_code=True
            )

        logger.info("Loading processor")

        self.processor = IsaacProcessor(
            tokenizer=self.tokenizer,
            config=self.config
        )

    @property
    def needs_fields(self):
        """A dict mapping model-specific keys to sample field names."""
        return self._fields

    @needs_fields.setter
    def needs_fields(self, fields):
        self._fields = fields
    
    def _get_field(self):
        if "prompt_field" in self.needs_fields:
            prompt_field = self.needs_fields["prompt_field"]
        else:
            prompt_field = next(iter(self.needs_fields.values()), None)

        return prompt_field

    @property
    def media_type(self):
        return "image"
    
    @property
    def ragged_batches(self):
        """Enable handling of varying image sizes in batches."""
        return True
    
    @property
    def operation(self):
        return self._operation

    @operation.setter
    def operation(self, value):
        if value not in OPERATIONS:
            raise ValueError(f"Invalid operation: {value}. Must be one of {list(OPERATIONS.keys())}")
        self._operation = value

    @property
    def system_prompt(self):
        # Return custom system prompt if set, otherwise return default for current operation
        if self._custom_system_prompt is not None:
            return self._custom_system_prompt
        return OPERATIONS[self.operation].get("prompt", "")

    @system_prompt.setter
    def system_prompt(self, value):
        self._custom_system_prompt = value

    def _to_classifications(self, classes: List[Dict]) -> fo.Classifications:
        """Convert a list of classification dictionaries to FiftyOne Classifications.
        
        Args:
            classes: List of dictionaries containing classification information.
                Each dictionary should have:
                - 'label': String class label
                
        Returns:
            fo.Classifications object containing the classification annotations
        """
        if not classes:
            return fo.Classifications(classifications=[])
            
        classifications = []
        
        for cls in classes:
            try:
                classification = fo.Classification(
                    label=str(cls["label"]),
                )
                classifications.append(classification)
            except Exception as e:
                logger.debug(f"Error processing classification {cls}: {e}")
                continue
                
        return fo.Classifications(classifications=classifications)

    def _parse_classification_output(self, output_text: str) -> List[Dict]:
        """Parse classification output from JSON format.

        Classifications use JSON output rather than XML tags.

        Args:
            output_text: Raw text output from the model

        Returns:
            List of classification dictionaries with 'label' keys
        """
        # Remove think blocks first, then strip geometry tags
        text_without_think = extract_reasoning(output_text).text
        cleaned_text = strip_tags(text_without_think).strip()

        # Extract JSON from markdown code block if present
        if "```json" in cleaned_text:
            try:
                cleaned_text = cleaned_text.split("```json")[1].split("```")[0].strip()
            except IndexError:
                logger.warning("Failed to extract JSON from code block")

        try:
            parsed = json.loads(cleaned_text)
            if isinstance(parsed, dict):
                classifications = parsed.get('classifications', [])
                # Validate and normalize classification entries
                validated = []
                for cls in classifications:
                    if isinstance(cls, dict) and 'label' in cls:
                        cls['label'] = str(cls['label'])
                        validated.append(cls)
                return validated
            elif isinstance(parsed, list):
                # Handle case where model returns a list directly
                validated = []
                for cls in parsed:
                    if isinstance(cls, dict) and 'label' in cls:
                        cls['label'] = str(cls['label'])
                        validated.append(cls)
                return validated
        except json.JSONDecodeError as e:
            logger.warning(f"Classification JSON parse error: {e}")
        except Exception as e:
            logger.warning(f"Classification parse error: {e}")

        return []

    def _process_output(self, output_text: str, image: Image.Image):
        """Process model output text based on the current operation type.

        Uses Perceptron SDK for parsing XML-style tags and converts to FiftyOne types.

        Args:
            output_text: Raw text output from the model
            image: PIL Image that was processed

        Returns:
            Processed output in the appropriate format for the operation
        """
        if self.operation == "vqa":
            # Remove think blocks, then strip geometry tags
            text_without_think = extract_reasoning(output_text).text
            return strip_tags(text_without_think).strip()

        elif self.operation == "ocr":
            # Remove think blocks, then strip geometry tags
            text_without_think = extract_reasoning(output_text).text
            return strip_tags(text_without_think).strip()

        elif self.operation == "detect":
            # Extract bounding boxes using Perceptron SDK
            boxes = extract_points(output_text, expected="box")
            return perceptron_to_fiftyone_detections(boxes, is_ocr=False)

        elif self.operation == "point":
            # Extract points using Perceptron SDK
            points = extract_points(output_text, expected="point")

            # Fallback: if model returns boxes instead of points, use box centers
            if not points:
                boxes = extract_points(output_text, expected="box")
                points = [box_to_center_point(b) for b in boxes]

            return perceptron_to_fiftyone_keypoints(points)

        elif self.operation == "classify":
            # Classification uses JSON output, keep existing parsing
            parsed = self._parse_classification_output(output_text)
            return self._to_classifications(parsed)

        elif self.operation == "ocr_detection":
            # Extract bounding boxes with OCR text labels
            boxes = extract_points(output_text, expected="box")
            return perceptron_to_fiftyone_detections(boxes, is_ocr=True)

        elif self.operation == "ocr_polygon":
            # Extract polygons with OCR text labels
            polygons = extract_points(output_text, expected="polygon")
            return perceptron_to_fiftyone_polylines(polygons, is_ocr=True)

        elif self.operation == "polygon" or self.operation == "segment":
            # Extract polygons using Perceptron SDK
            polygons = extract_points(output_text, expected="polygon")
            return perceptron_to_fiftyone_polylines(polygons, is_ocr=False)

        else:
            return None

    def _predict(self, image: Image.Image, sample=None) -> Union[fo.Detections, fo.Keypoints, fo.Classifications, str]:
        """Process a single image through the model and return predictions.
        
        Args:
            image: PIL Image to process
            sample: Optional FiftyOne sample containing the image filepath
            
        Returns:
            Model predictions in the appropriate format for the current operation
        """
        # Use local prompt variable instead of modifying self.prompt
        prompt = self.prompt
        
        if sample is not None and self._get_field() is not None:
            field_value = sample.get_field(self._get_field())
            if field_value is not None:
                prompt = str(field_value)

        # Prepare input with optional hint
        messages = [
            {"type": "text", "content": self.system_prompt, "role": "system"},
        ]
        
        # Add hint if available for this operation
        hint = OPERATIONS[self.operation].get("hint")
        if hint:
            messages.append({"type": "text", "content": f"<hint>{hint}</hint>", "role": "user"})
        
        # Add image and user prompt
        messages.extend([
            {"type": "image", "content": "<image>", "role": "user"},
            {"type": "text", "content": prompt, "role": "user"}
        ])
        
        images = [image]

        # Process input
        text = self.processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
            )

        inputs = self.processor(
            text=text, 
            images=images, 
            return_tensors="pt"
            )
        
        tensor_stream = inputs["tensor_stream"].to(self.device)

        # Generate response
        with torch.no_grad():
            generated_ids = self.model.generate(
                tensor_stream=tensor_stream,
                max_new_tokens=16384,
                do_sample=False,
                pad_token_id=self.processor.tokenizer.eos_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
            )

        # Decode and process output
        output_text = self.processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return self._process_output(output_text, image)

    def _predict_batch(self, images: List[Image.Image], samples: Optional[List] = None) -> List:
        """Process multiple images in a single model call for efficiency.
        
        Args:
            images: List of PIL Images to process
            samples: Optional list of FiftyOne samples corresponding to each image
            
        Returns:
            List of predictions, one for each input image
        """
        batch_size = len(images)
        results = []
        
        # Process each image with its corresponding sample (if provided)
        for i in range(batch_size):
            image = images[i]
            sample = samples[i] if samples and i < len(samples) else None
            
            # Get prompt for this specific image/sample
            prompt = self.prompt
            
            if sample is not None and self._get_field() is not None:
                field_value = sample.get_field(self._get_field())
                if field_value is not None:
                    prompt = str(field_value)
            
            # Prepare messages for this image with optional hint
            messages = [
                {"role": "system", "content": self.system_prompt},
            ]
            
            # Add hint if available for this operation
            hint = OPERATIONS[self.operation].get("hint")
            if hint:
                messages.append({"role": "user", "content": f"<hint>{hint}</hint>"})
            
            # Add image and user prompt
            messages.extend([
                {"role": "user", "content": "<image>"},
                {"role": "user", "content": prompt}
            ])
            
            # Process input
            text = self.processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            inputs = self.processor(
                text=text, 
                images=[image],
                return_tensors="pt"
            )
            
            tensor_stream = inputs["tensor_stream"].to(self.device)
            
            # Generate response
            with torch.no_grad():
                generated_ids = self.model.generate(
                    tensor_stream=tensor_stream,
                    max_new_tokens=8192,
                    do_sample=False,
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                )
            
            # Decode and process output
            output_text = self.processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            result = self._process_output(output_text, image)
            results.append(result)
        
        return results

    def predict_all(self, args):
        """Efficient batch prediction for multiple images.
        
        Args:
            args: List of tuples where each tuple contains (image, sample) or just images
            
        Returns:
            List of predictions, one for each input
        """
        if not args:
            return []
        
        # Separate images and samples
        images = []
        samples = []
        
        for arg in args:
            if isinstance(arg, tuple):
                image, sample = arg
                samples.append(sample)
            else:
                image = arg
                samples.append(None)
            
            # Convert numpy arrays to PIL Images
            if isinstance(image, np.ndarray):
                images.append(Image.fromarray(image))
            else:
                images.append(image)
        
        # Process batch
        return self._predict_batch(images, samples if any(s is not None for s in samples) else None)

    def predict(self, image, sample=None):
        """Process an image with the model.
        
        Args:
            image: PIL Image or numpy array to process
            sample: Optional FiftyOne sample containing the image filepath
            
        Returns:
            Model predictions in the appropriate format for the current operation
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        return self._predict(image, sample)