import cv2
import numpy as np
from paddleocr import PaddleOCR
import json

# Initialize PaddleOCR engine
ocr_engine = PaddleOCR(
    det_model_dir="ocr_models/ch_PP-OCRv4_det_infer",
    rec_model_dir="ocr_models/ch_PP-OCRv4_rec_infer",
    cls_model_dir="ocr_models/ch_ppocr_mobile_v2.0_cls_infer",
    rec_char_dict_path="ocr_models/ppocr_keys_v1.txt",
    use_angle_cls=True,
    lang="en",
    det_db_box_thresh=0.3,
    drop_score=0.2
)

# Deskewing function
def deskew(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    coords = np.column_stack(np.where(gray > 0))
    angle = cv2.minAreaRect(coords)[-1]
    
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

# Function to apply SWT
def apply_swt(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
    # Placeholder SWT implementation (simplified for demo)
    # SWT implementation is complex and would require a custom library or extended logic.
    # For now, let's filter contours based on stroke width by assuming strokes have similar width.
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    filtered_contours = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h
        if 0.1 < aspect_ratio < 5.0 and w > 10 and h > 10:  # Filter based on aspect ratio and size
            filtered_contours.append(contour)
    
    return filtered_contours

# Processing image using MSER, SWT, and contour filtering
def process_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # MSER detection
    mser = cv2.MSER_create()
    regions, _ = mser.detectRegions(gray)
    
    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
    mask = np.zeros_like(gray)
    cv2.fillPoly(mask, hulls, 255)
    
    # SWT: apply stroke width filtering on the masked image
    swt_contours = apply_swt(image)

    # Find contours in the SWT mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    text_results = []
    output_image = image.copy()

    for contour in contours:
        x_min, y_min, w, h = cv2.boundingRect(contour)

        # Padding around the region
        padding = 5
        x_min = max(0, x_min - padding)
        y_min = max(0, x_min - padding)
        x_max = min(image.shape[1], x_min + w + padding)
        y_max = min(image.shape[0], y_min + h + padding)

        w = x_max - x_min
        h = y_max - y_min

        # Use both SWT and MSER regions for text detection
        if w > 10 and h > 10 and any([cv2.pointPolygonTest(contour, (x_min + w // 2, y_min + h // 2), False) >= 0 for contour in swt_contours]):
            cropped = image[y_min:y_max, x_min:x_max]
            
            # Deskew the detected region
            deskewed = deskew(cropped)
            
            # OCR processing
            deskewed_rgb = cv2.cvtColor(deskewed, cv2.COLOR_BGR2RGB)
            ocr_result = ocr_engine.ocr(deskewed_rgb, cls=True)

            if ocr_result and ocr_result[0]:
                for line in ocr_result[0]:
                    text = line[1][0]
                    text_results.append({
                        "bounding_box": {"x": int(x_min), "y": int(y_min), "width": int(w), "height": int(h)},
                        "text": text.strip()
                    })
                    
                    # Draw bounding box and text on the original image
                    cv2.rectangle(output_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    cv2.putText(output_image, text.strip(), (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save the processed image with bounding boxes
    output_image_path = r"C:\Users\sruth\Desktop\bounded_after_prep\recognized_text_image_swt_mser.png"
    cv2.imwrite(output_image_path, output_image)
    
    return text_results

image_path = r"C:\Users\sruth\Desktop\bounded_after_prep\05-07-2022_6-XRayTest.png"
ocr_results = process_image(image_path)

# Save the OCR results in JSON format
output_json_path = r"C:\Users\sruth\Desktop\bounded_after_prep\results_swt_mser.json"
with open(output_json_path, 'w') as json_file:
    json.dump(ocr_results, json_file, indent=4)













