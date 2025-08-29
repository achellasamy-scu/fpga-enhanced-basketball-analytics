#!/usr/bin/env python3
"""
ONNX YOLO Inference with Vitis-AI Execution Provider
Runs quantized YOLO models on ZCU104 with DPUCZDX8G_B4096
"""

import argparse
import os
import sys
import time
import glob
from pathlib import Path
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    print("ERROR: onnxruntime not found. Install with: pip install onnxruntime")
    sys.exit(1)

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    print("ERROR: PIL not found. Install with: pip install Pillow")
    sys.exit(1)

# Basketball detection class names
BASKETBALL_CLASSES = [
    "3pt_area", "ball", "court", "hoop", "number", "paint", "player"
]

def letterbox_resize(img, target_size=(640, 640)):
    """Resize PIL image with letterboxing to maintain aspect ratio"""
    w, h = img.size
    target_w, target_h = target_size
    
    # Calculate scale factor
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize image
    resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    # Create letterboxed image with gray padding
    letterboxed = Image.new('RGB', (target_w, target_h), (114, 114, 114))
    
    # Calculate offsets for centering
    offset_x = (target_w - new_w) // 2
    offset_y = (target_h - new_h) // 2
    
    # Paste resized image in center
    letterboxed.paste(resized, (offset_x, offset_y))
    
    return letterboxed, scale, offset_x, offset_y

def preprocess_image(img, input_size=(640, 640)):
    """Preprocess PIL image for YOLO model"""
    # Letterbox resize
    processed, scale, offset_x, offset_y = letterbox_resize(img, input_size)
    
    # Convert to numpy array and normalize
    img_array = np.array(processed, dtype=np.float32) / 255.0
    
    # Convert HWC to CHW format
    img_array = np.transpose(img_array, (2, 0, 1))  # HWC to CHW
    img_array = np.expand_dims(img_array, axis=0)   # Add batch dimension
    
    return img_array, scale, offset_x, offset_y

def postprocess_detections(outputs, scale, offset_x, offset_y, conf_threshold=0.25, iou_threshold=0.45):
    """Post-process model outputs to get final detections"""
    detections = []
    
    try:
        # Handle different output formats
        if len(outputs) == 1:
            output = outputs[0]
        else:
            output = outputs[0]
            
        print(f"Output tensor shape: {output.shape}")
        
        # Check if this is raw YOLO head format [batch, classes+5, anchors] or [batch, anchors, classes+5]
        if len(output.shape) == 3:
            batch_size, dim1, dim2 = output.shape
            
            # Determine format: (1, 11, 8400) suggests (batch, bbox_attrs, num_anchors)
            if dim1 < dim2:  # (1, 11, 8400) format
                # Transpose to (batch, num_anchors, bbox_attrs)
                output = np.transpose(output, (0, 2, 1))  # (1, 8400, 11)
                print(f"Transposed to: {output.shape}")
            
            # Now we should have (batch, num_anchors, bbox_attrs)
            batch_detections = output[0]  # Remove batch dimension -> (8400, 11)
            
            # Determine number of classes from bbox_attrs
            num_attrs = batch_detections.shape[1]  # Should be 11
            num_classes = num_attrs - 5  # 11 - 5 = 6 classes (but you said 7?)
            print(f"Detected {num_classes} classes from {num_attrs} attributes")
            
            # Process each anchor/detection
            for i, det in enumerate(batch_detections):
                if len(det) >= 5:
                    # YOLO format: [center_x, center_y, width, height, objectness, class1, class2, ...]
                    cx, cy, w, h, obj_conf = det[:5]
                    class_probs = det[5:5+num_classes]
                    
                    # Calculate overall confidence for each class
                    class_confidences = obj_conf * class_probs
                    
                    # Find best class
                    best_class_id = np.argmax(class_confidences)
                    best_conf = class_confidences[best_class_id]
                    
                    if best_conf > conf_threshold:
                        # Convert center coords to corner coords (in model space 0-640)
                        x1 = cx - w/2
                        y1 = cy - h/2
                        x2 = cx + w/2
                        y2 = cy + h/2
                        
                        # Validate coordinates
                        if x1 >= x2 or y1 >= y2 or x1 < 0 or y1 < 0:
                            continue  # Skip invalid boxes
                        
                        # Adjust coordinates for letterboxing (convert back to original image space)
                        x1 = (x1 - offset_x) / scale
                        y1 = (y1 - offset_y) / scale  
                        x2 = (x2 - offset_x) / scale
                        y2 = (y2 - offset_y) / scale
                        
                        # Final validation after scaling
                        if x1 >= x2 or y1 >= y2:
                            continue  # Skip invalid boxes after scaling
                        
                        detections.append({
                            'bbox': [float(x1), float(y1), float(x2), float(y2)],
                            'confidence': float(best_conf),
                            'class_id': int(best_class_id),
                            'class_name': BASKETBALL_CLASSES[int(best_class_id)] if int(best_class_id) < len(BASKETBALL_CLASSES) else f"class_{int(best_class_id)}"
                        })
            
            # Apply NMS to remove overlapping detections
            if detections:
                detections = apply_nms(detections, iou_threshold)
                        
        else:
            print(f"Unexpected output format: {output.shape}")
            print("Expected 3D tensor for raw YOLO head")
            
    except Exception as e:
        print(f"Error processing detections: {e}")
        print(f"Output shapes: {[out.shape for out in outputs]}")
        import traceback
        traceback.print_exc()
        
    return detections

def apply_nms(detections, iou_threshold=0.45):
    """Apply Non-Maximum Suppression to remove overlapping detections"""
    if len(detections) == 0:
        return []
    
    # Sort by confidence (highest first)
    detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
    
    def calculate_iou(box1, box2):
        """Calculate Intersection over Union of two boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
            
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    # Apply NMS
    final_detections = []
    
    while detections:
        # Take the detection with highest confidence
        best_det = detections.pop(0)
        final_detections.append(best_det)
        
        # Remove detections with high IoU with the best detection (same class only)
        remaining_detections = []
        for det in detections:
            if (det['class_id'] != best_det['class_id'] or 
                calculate_iou(best_det['bbox'], det['bbox']) < iou_threshold):
                remaining_detections.append(det)
        
        detections = remaining_detections
    
    return final_detections

def draw_detections(img, detections):
    """Draw bounding boxes and labels on PIL image"""
    # Convert PIL to drawing object
    draw = ImageDraw.Draw(img)
    
    # Try to load a font, fall back to default if not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except:
        try:
            font = ImageFont.load_default()
        except:
            font = None
    
    colors = [
        (255, 0, 0),    # 3pt_area - red
        (0, 255, 0),    # ball - green  
        (0, 0, 255),    # court - blue
        (255, 255, 0),  # hoop - cyan
        (255, 0, 255),  # number - magenta
        (0, 255, 255),  # paint - yellow
        (128, 0, 128)   # player - purple
    ]
    
    for det in detections:
        x1, y1, x2, y2 = [int(coord) for coord in det['bbox']]
        conf = det['confidence']
        class_id = det['class_id']
        class_name = det['class_name']
        
        # Get color for this class
        color = colors[class_id % len(colors)]
        
        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        
        # Draw label with background
        label = f"{class_name}: {conf:.2f}"
        
        if font:
            bbox = draw.textbbox((0, 0), label, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        else:
            # Estimate text size if no font available
            text_width = len(label) * 8
            text_height = 12
            
        # Draw label background
        draw.rectangle([x1, y1 - text_height - 4, x1 + text_width + 4, y1], 
                      fill=color)
        
        # Draw text
        if font:
            draw.text((x1 + 2, y1 - text_height - 2), label, fill=(255, 255, 255), font=font)
        else:
            draw.text((x1 + 2, y1 - text_height - 2), label, fill=(255, 255, 255))
    
    return img

def load_onnx_session(model_path, cache_dir):
    """Load ONNX session with Vitis-AI EP"""
    print(f"Loading model: {model_path}")
    
    # Ensure cache directory exists
    os.makedirs(cache_dir, exist_ok=True)
    print(f"Cache directory: {cache_dir}")
    
    # Setup providers
    providers = [
        ("VitisAIExecutionProvider", {
            "log_level": "info",
            "cache_dir": cache_dir
        }),
        "CPUExecutionProvider"
    ]
    
    try:
        session = ort.InferenceSession(model_path, providers=providers)
        print(f"Session providers: {session.get_providers()}")
        
        # Get input info
        input_info = session.get_inputs()[0]
        print(f"Input tensor: {input_info.name}, shape: {input_info.shape}")
        
        return session, input_info
        
    except Exception as e:
        print(f"Error loading session with Vitis-AI EP: {e}")
        print("Falling back to CPU-only execution")
        
        try:
            session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
            input_info = session.get_inputs()[0]
            print(f"CPU fallback successful - Input: {input_info.name}, shape: {input_info.shape}")
            return session, input_info
        except Exception as e2:
            print(f"ERROR: Failed to load model even with CPU EP: {e2}")
            sys.exit(1)

def run_inference_on_images(args):
    """Run inference on directory of images"""
    
    # Load model
    model_stem = Path(args.model).stem
    cache_dir = args.cache.replace('<model_stem>', model_stem)
    session, input_info = load_onnx_session(args.model, cache_dir)
    
    # Get list of images
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(args.frames, ext)))
        image_files.extend(glob.glob(os.path.join(args.frames, ext.upper())))
    
    if not image_files:
        print(f"ERROR: No images found in {args.frames}")
        return
        
    image_files.sort()
    print(f"Found {len(image_files)} images")
    
    if args.max_frames:
        image_files = image_files[:args.max_frames]
        print(f"Limited to {len(image_files)} images")
    
    # Warmup runs
    print(f"Running {args.warmup} warmup iterations...")
    if image_files:
        warmup_img = Image.open(image_files[0])
        if warmup_img is not None:
            processed, _, _, _ = preprocess_image(warmup_img, tuple(args.size))
            for _ in range(args.warmup):
                _ = session.run(None, {input_info.name: processed})
    
    print("Starting timed inference...")
    inference_times = []
    total_detections = 0
    
    start_time = time.time()
    
    for i, img_path in enumerate(image_files):
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Warning: Could not load {img_path}: {e}")
            continue
            
        # Preprocess
        processed, scale, offset_x, offset_y = preprocess_image(img, tuple(args.size))
        
        # Inference
        inf_start = time.time()
        outputs = session.run(None, {input_info.name: processed})
        inf_time = time.time() - inf_start
        inference_times.append(inf_time)
        
        # Postprocess
        detections = postprocess_detections(outputs, scale, offset_x, offset_y, args.conf, args.iou)
        total_detections += len(detections)
        
        if args.show and detections:
            # Draw detections
            display_img = draw_detections(img.copy(), detections)
            
            # Show image (basic display, press any key to continue)
            display_img.show()
            input("Press Enter to continue to next image (or Ctrl+C to quit)...")
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(image_files)} images")
    
    total_time = time.time() - start_time
    
    # Print results
    print("\n" + "="*50)
    print("INFERENCE RESULTS")
    print("="*50)
    print(f"Total images processed: {len(inference_times)}")
    print(f"Total inference time: {sum(inference_times):.3f}s")
    print(f"Total runtime: {total_time:.3f}s")
    print(f"FPS: {len(inference_times) / total_time:.2f}")
    print(f"Mean inference time: {np.mean(inference_times)*1000:.2f}ms")
    print(f"Median inference time: {np.median(inference_times)*1000:.2f}ms")
    print(f"Total detections: {total_detections}")
    print(f"Avg detections per image: {total_detections/len(inference_times):.1f}")
    
    # Note: No cleanup needed for PIL (unlike cv2.destroyAllWindows())

def main():
    parser = argparse.ArgumentParser(description='ONNX YOLO Inference with Vitis-AI EP')
    
    # Required arguments
    parser.add_argument('--model', required=True, help='Path to ONNX model')
    
    # Input source (exactly one required)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--frames', help='Directory containing images')
    input_group.add_argument('--video', help='Path to video file (not implemented yet)')
    
    # Model parameters
    parser.add_argument('--size', nargs=2, type=int, default=[640, 640], 
                       help='Input size (height width)')
    parser.add_argument('--conf', type=float, default=0.25, 
                       help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45, 
                       help='IoU threshold for NMS')
    
    # Execution parameters
    parser.add_argument('--cache', default='/tmp/vaip_cache/<model_stem>', 
                       help='VAIP cache directory')
    parser.add_argument('--warmup', type=int, default=1, 
                       help='Number of warmup iterations')
    parser.add_argument('--max_frames', type=int, 
                       help='Maximum number of frames to process')
    
    # Display
    parser.add_argument('--show', action='store_true', 
                       help='Save detection results to ./detection_results/ directory')
    
    # Advanced options
    parser.add_argument('--raw_head', action='store_true',
                       help='Use raw head processing (not implemented)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.model):
        print(f"ERROR: Model file not found: {args.model}")
        sys.exit(1)
    
    if args.video:
        print("ERROR: Video processing not implemented yet. Use --frames instead.")
        sys.exit(1)
        
    if args.frames and not os.path.isdir(args.frames):
        print(f"ERROR: Frames directory not found: {args.frames}")
        sys.exit(1)
    
    if args.raw_head:
        print("WARNING: --raw_head not implemented yet")
    
    # Run inference
    if args.frames:
        run_inference_on_images(args)

if __name__ == '__main__':
    main()