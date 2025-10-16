import os
import logging
import json
import re
from PIL import Image
from typing import Dict, Any, List, Union, Optional

import numpy as np
import torch

import fiftyone as fo
from fiftyone import Model, SamplesMixin

from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from transformers.utils.import_utils import is_flash_attn_2_available

from .modular_isaac import IsaacProcessor

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

    def _strip_think_blocks(self, text: str) -> str:
        """Remove <think>...</think> blocks from text."""
        return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

    def _parse_coordinates(self, coord_str: str) -> Optional[List[float]]:
        """Extract coordinates from parentheses-enclosed tuples.
        
        Matches patterns like (x,y) or (x, y) with optional whitespace.
        Ensures we only extract numbers that are actually coordinate pairs,
        not stray numbers in the text.
        
        Args:
            coord_str: String containing coordinate pairs in format (x,y)
            
        Returns:
            List of floats representing coordinates, or None if no valid coords found
            
        Examples:
            "(100, 200)" -> [100.0, 200.0]
            "(0,0) (100,100)" -> [0.0, 0.0, 100.0, 100.0]
            "Found 2 objects at (50, 75)" -> [50.0, 75.0]  # Ignores "2"
        """
        # Match coordinate pairs in parentheses with optional whitespace
        coord_pattern = r'\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)'
        matches = re.findall(coord_pattern, coord_str)
        
        if not matches:
            return None
        
        # Flatten list of tuples into single list of floats
        coords = []
        for pair in matches:
            coords.extend([float(pair[0]), float(pair[1])])
        
        return coords

    def _extract_point_boxes(self, text: str) -> List[Dict]:
        """Extract all <point_box> elements from text."""
        boxes = []
        
        # Pattern for point_box with optional mention attribute
        pattern = r'<point_box(?:\s+mention="([^"]*)")?\s*>\s*(.*?)\s*</point_box>'
        
        unlabeled_count = 0
        for match in re.finditer(pattern, text, flags=re.DOTALL):
            mention = match.group(1)
            coords_text = match.group(2)
            
            coords = self._parse_coordinates(coords_text)
            if coords and len(coords) >= 4:
                # Generate unique label for unlabeled detections
                if mention:
                    label = mention
                else:
                    unlabeled_count += 1
                    label = f'object_{unlabeled_count}'
                
                boxes.append({
                    'bbox_2d': coords[:4],
                    'label': label
                })
        
        return boxes

    def _extract_points(self, text: str) -> List[Dict]:
        """Extract all <point> elements from text."""
        points = []
        
        # Pattern for point with optional mention attribute
        pattern = r'<point(?:\s+mention="([^"]*)")?\s*>\s*(.*?)\s*</point>'
        
        unlabeled_count = 0
        for match in re.finditer(pattern, text, flags=re.DOTALL):
            mention = match.group(1)
            coords_text = match.group(2)
            
            coords = self._parse_coordinates(coords_text)
            if coords and len(coords) >= 2:
                # Generate unique label for unlabeled points
                if mention:
                    label = mention
                else:
                    unlabeled_count += 1
                    label = f'point_{unlabeled_count}'
                
                points.append({
                    'point_2d': coords[:2],
                    'label': label
                })
        
        return points

    def _extract_polygons(self, text: str) -> List[Dict]:
        """Extract all <polygon> elements from text."""
        polygons = []
        
        # Pattern for polygon with optional mention attribute
        pattern = r'<polygon(?:\s+mention="([^"]*)")?\s*>\s*(.*?)\s*</polygon>'
        
        unlabeled_count = 0
        for match in re.finditer(pattern, text, flags=re.DOTALL):
            mention = match.group(1)
            coords_text = match.group(2)
            
            coords = self._parse_coordinates(coords_text)
            if not coords:
                logger.debug(f"No coordinates found in polygon: {coords_text[:100]}")
                continue
                
            # Polygons need at least 3 points (6 numbers)
            if len(coords) < 6:
                logger.debug(f"Insufficient coordinates for polygon (need >=6, got {len(coords)}): {coords}")
                continue
            
            # Check for odd number of coordinates and fix
            if len(coords) % 2 != 0:
                logger.warning(f"Odd number of coordinates in polygon ({len(coords)}), dropping last value")
                coords = coords[:-1]
            
            # Group coordinates into vertex pairs
            vertices = [[coords[i], coords[i + 1]] for i in range(0, len(coords), 2)]
            
            # Final validation: ensure we have at least 3 vertices
            if len(vertices) < 3:
                logger.debug(f"Insufficient vertices for polygon (need >=3, got {len(vertices)})")
                continue
            
            # Generate unique label for unlabeled polygons
            if mention:
                label = mention
            else:
                unlabeled_count += 1
                label = f'polygon_{unlabeled_count}'
            
            polygons.append({
                'vertices': vertices,
                'label': label
            })
        
        return polygons

    def _extract_all_elements(self, text: str, element_type: str = None) -> List[Dict]:
        """Extract all elements from text, both from collections and standalone, with proper inheritance."""
        all_elements = []
        
        # Pattern for collection with optional mention attribute
        collection_pattern = r'<collection(?:\s+mention="([^"]*)")?\s*>(.*?)</collection>'
        
        # First, process collections
        for match in re.finditer(collection_pattern, text, flags=re.DOTALL):
            collection_mention = match.group(1)
            collection_content = match.group(2)
            
            # Extract elements from within the collection
            if element_type == 'point_box' or element_type is None:
                boxes = self._extract_point_boxes(collection_content)
                for box in boxes:
                    # Inherit collection mention if box has default label
                    if box['label'].startswith('object_') and collection_mention:
                        box['label'] = collection_mention
                    all_elements.append({'bbox_2d': box['bbox_2d'], 'label': box['label']})
            
            if element_type == 'point' or element_type is None:
                points = self._extract_points(collection_content)
                for point in points:
                    # Inherit collection mention if point has default label
                    if point['label'].startswith('point_') and collection_mention:
                        point['label'] = collection_mention
                    all_elements.append({'point_2d': point['point_2d'], 'label': point['label']})
            
            if element_type == 'polygon' or element_type is None:
                polygons = self._extract_polygons(collection_content)
                for polygon in polygons:
                    # Inherit collection mention if polygon has default label
                    if polygon['label'].startswith('polygon_') and collection_mention:
                        polygon['label'] = collection_mention
                    all_elements.append({'vertices': polygon['vertices'], 'label': polygon['label']})
        
        # Remove collections from text to avoid double-processing
        text_without_collections = re.sub(collection_pattern, '', text, flags=re.DOTALL)
        
        # Process standalone elements (not in collections)
        if element_type == 'point_box' or element_type is None:
            all_elements.extend(self._extract_point_boxes(text_without_collections))
        if element_type == 'point' or element_type is None:
            all_elements.extend(self._extract_points(text_without_collections))
        if element_type == 'polygon' or element_type is None:
            all_elements.extend(self._extract_polygons(text_without_collections))
        
        return all_elements

    def _validate_and_fix_json_structure(self, parsed_json: Dict) -> Dict:
        """Validate and fix JSON structure from model output.
        
        Ensures the parsed JSON has all required keys with properly formatted values.
        Also validates the structure of nested elements.
        
        Args:
            parsed_json: Dictionary parsed from model JSON output
            
        Returns:
            Validated and fixed dictionary with all required keys and valid structure
        """
        # Ensure all expected keys exist as lists
        required_keys = ['detections', 'keypoints', 'polygons', 'classifications', 
                         'text_detections', 'text_polygons']
        
        for key in required_keys:
            if key not in parsed_json:
                parsed_json[key] = []
            elif not isinstance(parsed_json[key], list):
                logger.warning(f"Expected list for '{key}', got {type(parsed_json[key])}. Converting to empty list.")
                parsed_json[key] = []
        
        # Validate detections structure
        validated_detections = []
        for det in parsed_json.get('detections', []):
            if not isinstance(det, dict):
                logger.warning(f"Invalid detection format (not a dict): {det}")
                continue
            
            # Check for required bbox field (accept either bbox or bbox_2d)
            if 'bbox' in det or 'bbox_2d' in det:
                bbox = det.get('bbox', det.get('bbox_2d'))
                # Validate bbox is a list of 4 numbers
                if isinstance(bbox, list) and len(bbox) == 4:
                    try:
                        # Ensure all bbox values are numeric
                        _ = [float(x) for x in bbox]
                        validated_detections.append(det)
                    except (ValueError, TypeError):
                        logger.warning(f"Invalid bbox coordinates (not numeric): {bbox}")
                else:
                    logger.warning(f"Invalid bbox format (need 4 numbers): {bbox}")
            else:
                logger.warning(f"Detection missing bbox field: {det}")
        
        parsed_json['detections'] = validated_detections
        parsed_json['text_detections'] = validated_detections.copy()
        
        # Validate keypoints structure
        validated_keypoints = []
        for kp in parsed_json.get('keypoints', []):
            if not isinstance(kp, dict):
                logger.warning(f"Invalid keypoint format (not a dict): {kp}")
                continue
            
            # Check for required point field
            if 'point_2d' in kp or 'point' in kp:
                point = kp.get('point_2d', kp.get('point'))
                # Validate point is a list of 2 numbers
                if isinstance(point, list) and len(point) == 2:
                    try:
                        _ = [float(x) for x in point]
                        validated_keypoints.append(kp)
                    except (ValueError, TypeError):
                        logger.warning(f"Invalid point coordinates (not numeric): {point}")
                else:
                    logger.warning(f"Invalid point format (need 2 numbers): {point}")
            else:
                logger.warning(f"Keypoint missing point field: {kp}")
        
        parsed_json['keypoints'] = validated_keypoints
        
        # Validate polygons structure
        validated_polygons = []
        for poly in parsed_json.get('polygons', []):
            if not isinstance(poly, dict):
                logger.warning(f"Invalid polygon format (not a dict): {poly}")
                continue
            
            # Check for required vertices field
            if 'vertices' in poly:
                vertices = poly['vertices']
                # Validate vertices is a list of coordinate pairs
                if isinstance(vertices, list) and len(vertices) >= 3:
                    try:
                        # Ensure all vertices are valid coordinate pairs
                        valid_vertices = True
                        for vertex in vertices:
                            if not isinstance(vertex, list) or len(vertex) != 2:
                                valid_vertices = False
                                break
                            _ = [float(x) for x in vertex]
                        
                        if valid_vertices:
                            validated_polygons.append(poly)
                        else:
                            logger.warning(f"Invalid vertex format in polygon: {vertices}")
                    except (ValueError, TypeError):
                        logger.warning(f"Invalid polygon coordinates (not numeric): {vertices}")
                else:
                    logger.warning(f"Invalid polygon format (need >=3 vertices): {vertices}")
            else:
                logger.warning(f"Polygon missing vertices field: {poly}")
        
        parsed_json['polygons'] = validated_polygons
        parsed_json['text_polygons'] = validated_polygons.copy()
        
        # Validate classifications structure
        validated_classifications = []
        for cls in parsed_json.get('classifications', []):
            if not isinstance(cls, dict):
                logger.warning(f"Invalid classification format (not a dict): {cls}")
                continue
            
            # Check for required label field
            if 'label' in cls:
                # Ensure label is a string
                if isinstance(cls['label'], str) or isinstance(cls['label'], (int, float)):
                    cls['label'] = str(cls['label'])
                    validated_classifications.append(cls)
                else:
                    logger.warning(f"Invalid label type: {type(cls['label'])}")
            else:
                logger.warning(f"Classification missing label field: {cls}")
        
        parsed_json['classifications'] = validated_classifications
        
        return parsed_json

    def _parse_model_output(self, output_text: str) -> Dict:
        """
        Unified parser that handles both XML-like and JSON formats.
        Returns a standardized dictionary structure regardless of input format.
        """
        # Step 1: Clean the output
        cleaned_text = self._strip_think_blocks(output_text)
        
        # Step 2: Detect format and parse accordingly
        # Check if it's XML-like format
        has_xml_tags = any(tag in cleaned_text for tag in ['<point>', '<point_box>', '<polygon>', '<collection>'])
        
        if has_xml_tags:
            # Parse XML-like format
            result = {
                'detections': [],
                'keypoints': [],
                'polygons': [],
                'text_detections': [],
                'text_polygons': [],
                'classifications': []
            }
            
            # Extract all elements (both from collections and standalone)
            all_elements = self._extract_all_elements(cleaned_text)
            
            # Sort elements into appropriate categories
            for elem in all_elements:
                if 'bbox_2d' in elem:
                    result['detections'].append(elem)
                    result['text_detections'].append(elem)
                elif 'point_2d' in elem:
                    result['keypoints'].append(elem)
                elif 'vertices' in elem:
                    result['polygons'].append(elem)
                    result['text_polygons'].append(elem)
            
            return result
        else:
            # Try JSON format parsing
            json_text = cleaned_text
            
            # Handle JSON wrapped in markdown code blocks
            if "```json" in json_text:
                try:
                    json_text = json_text.split("```json")[1].split("```")[0].strip()
                except IndexError:
                    logger.debug("Failed to extract JSON from code block markers")
            
            # Attempt to parse the JSON string
            try:
                parsed_json = json.loads(json_text)
                if isinstance(parsed_json, dict):
                    # Validate and fix the structure
                    return self._validate_and_fix_json_structure(parsed_json)
                else:
                    logger.warning(f"Parsed JSON is not a dict: {type(parsed_json)}")
            except json.JSONDecodeError as e:
                logger.debug(f"Failed to parse as JSON: {e}. Text: {json_text[:200]}")
            except Exception as e:
                logger.warning(f"Unexpected error parsing JSON: {e}")
            
            # If both fail, return empty structure
            logger.debug(f"Could not parse output in any known format: {cleaned_text[:200]}")
            return {
                'detections': [],
                'keypoints': [],
                'polygons': [],
                'classifications': [],
                'text_detections': [],
                'text_polygons': []
            }

    def _to_detections(self, boxes: List[Dict], is_ocr: bool = False) -> fo.Detections:
        """Convert bounding boxes to FiftyOne Detections.
        
        This method handles both regular object detections and OCR text detections.
        
        Args:
            boxes: List of dictionaries containing bounding box info.
                Each box should have:
                - 'bbox_2d' or 'bbox': List of [x1,y1,x2,y2] coordinates in 0-1000 range
                - 'label': String label. For OCR detections, this contains the detected text.
            is_ocr: If True, treats label as OCR text and adds it as a custom attribute
        
        Returns:
            fo.Detections object with normalized coordinates [0,1] x [0,1]
        """
        if not boxes:
            return fo.Detections(detections=[])
            
        detections = []
        
        for box in boxes:
            try:
                # Try to get bbox from either bbox_2d or bbox field
                bbox = box.get('bbox_2d', box.get('bbox', None))
                if not bbox:
                    continue
                    
                # Model outputs coordinates in 0-1000 range, normalize to 0-1
                x1, y1, x2, y2 = map(float, bbox)
                x = x1 / 1000.0
                y = y1 / 1000.0
                w = (x2 - x1) / 1000.0
                h = (y2 - y1) / 1000.0
                
                label = str(box.get("label", "object"))
                
                # Create Detection object with normalized coordinates
                detection = fo.Detection(
                    label=label,
                    bounding_box=[x, y, w, h],
                )
                
                # For OCR mode, also store the text in a dedicated attribute
                if is_ocr:
                    detection["text"] = label
                
                detections.append(detection)
                
            except Exception as e:
                logger.debug(f"Error processing box {box}: {e}")
                continue
                
        return fo.Detections(detections=detections)

    def _to_keypoints(self, points: List[Dict]) -> fo.Keypoints:
        """Convert a list of point dictionaries to FiftyOne Keypoints.
        
        Args:
            points: List of dictionaries containing point information.
                Each point should have:
                - 'point_2d': List of [x,y] coordinates in 0-1000 range
                - 'label': String label describing the point
                
        Returns:
            fo.Keypoints object with normalized coordinates [0,1] x [0,1]
        """
        if not points:
            return fo.Keypoints(keypoints=[])
            
        keypoints = []
        
        for point in points:
            try:
                # Get coordinates from point_2d field and convert to float
                x, y = point["point_2d"]
                x = float(x)
                y = float(y)
                
                # Model outputs coordinates in 0-1000 range, normalize to 0-1
                normalized_point = [
                    x / 1000.0,
                    y / 1000.0
                ]
                
                keypoint = fo.Keypoint(
                    label=str(point.get("label", "point")),
                    points=[normalized_point],
                )
                keypoints.append(keypoint)
            except Exception as e:
                logger.debug(f"Error processing point {point}: {e}")
                continue
                
        return fo.Keypoints(keypoints=keypoints)

    def _to_polygons(self, polygons: List[Dict], is_ocr: bool = False) -> fo.Polylines:
        """Convert polygon data to FiftyOne Polylines.
        
        This method handles both regular polygon segmentations and OCR text polygons.
        
        Args:
            polygons: List of dictionaries containing polygon information.
                Each dictionary should have:
                - 'vertices': List of [x,y] coordinate pairs in 0-1000 range
                - 'label': String label. For OCR polygons, this contains the detected text.
            is_ocr: If True, treats label as OCR text and adds it as a custom attribute
                
        Returns:
            fo.Polylines object containing the polygon annotations
        """
        if not polygons:
            return fo.Polylines(polylines=[])
            
        polylines = []
        
        for polygon in polygons:
            try:
                vertices = polygon.get('vertices', [])
                if not vertices or len(vertices) < 3:
                    continue
                    
                # Convert vertices from 0-1000 range to 0-1 normalized
                normalized_points = []
                for x, y in vertices:
                    norm_x = float(x) / 1000.0
                    norm_y = float(y) / 1000.0
                    normalized_points.append([norm_x, norm_y])
                
                label = str(polygon.get('label', 'polygon'))
                
                # Create Polyline
                polyline = fo.Polyline(
                    label=label,
                    points=[normalized_points],
                    closed=True,
                    filled=True
                )
                
                # For OCR mode, also store the text in a dedicated attribute
                if is_ocr:
                    polyline["text"] = label
                
                polylines.append(polyline)
                
            except Exception as e:
                logger.debug(f"Error processing polygon {polygon}: {e}")
                continue
                
        return fo.Polylines(polylines=polylines)

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

    def _process_output(self, output_text: str, image: Image.Image):
        """Process model output text based on the current operation type.
        
        Args:
            output_text: Raw text output from the model
            image: PIL Image that was processed
            
        Returns:
            Processed output in the appropriate format for the operation
        """
        if self.operation == "vqa":
            cleaned = self._strip_think_blocks(output_text)
            return cleaned.strip()
        
        elif self.operation == "ocr":
            cleaned = self._strip_think_blocks(output_text)
            return cleaned.strip()
        
        elif self.operation == "detect":
            parsed = self._parse_model_output(output_text)
            data = parsed.get('detections', [])
            return self._to_detections(data, is_ocr=False)
        
        elif self.operation == "point":
            parsed = self._parse_model_output(output_text)
            data = parsed.get('keypoints', [])
            
            # Special case: if model outputs point_boxes when asked for points,
            # convert them to keypoints (use center of box)
            if not data and parsed.get('detections'):
                for detection in parsed['detections']:
                    if 'bbox_2d' in detection:
                        x1, y1, x2, y2 = detection['bbox_2d']
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        data.append({
                            'point_2d': [center_x, center_y],
                            'label': detection.get('label', 'point')
                        })
            
            return self._to_keypoints(data)
        
        elif self.operation == "classify":
            parsed = self._parse_model_output(output_text)
            data = parsed.get('classifications', [])
            return self._to_classifications(data)
        
        elif self.operation == "ocr_detection":
            parsed = self._parse_model_output(output_text)
            data = parsed.get('text_detections', [])
            return self._to_detections(data, is_ocr=True)
        
        elif self.operation == "ocr_polygon":
            parsed = self._parse_model_output(output_text)
            data = parsed.get('text_polygons', [])
            return self._to_polygons(data, is_ocr=True)
        
        elif self.operation == "polygon" or self.operation == "segment":
            parsed = self._parse_model_output(output_text)
            data = parsed.get('polygons', [])
            return self._to_polygons(data, is_ocr=False)
        
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