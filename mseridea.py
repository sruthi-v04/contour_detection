import cv2
import numpy as np
from paddleocr import PaddleOCR
import json

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

def process_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    mser = cv2.MSER_create()
    regions, _ = mser.detectRegions(gray)

    text_results = []

    for region in regions:
        x_min, y_min, w, h = cv2.boundingRect(region)
        
        padding = 5
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(image.shape[1], x_min + w + padding)
        y_max = min(image.shape[0], y_min + h + padding)

        w = x_max - x_min
        h = y_max - y_min

        if w > 10 and h > 10:
            cropped = image[y_min:y_max, x_min:x_max]
            deskewed = deskew(cropped)
            deskewed_rgb = cv2.cvtColor(deskewed, cv2.COLOR_BGR2RGB)
            ocr_result = ocr_engine.ocr(deskewed_rgb, cls=True)

            if ocr_result and ocr_result[0]:
                for line in ocr_result[0]:
                    text = line[1][0]
                    text_results.append({
                        "bounding_box": {"x": int(x_min), "y": int(y_min), "width": int(w), "height": int(h)},
                        "text": text.strip()
                    })
                    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    cv2.putText(image, text.strip(), (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    output_image_path = r"C:\Users\sruth\Desktop\bounded_after_prep\new_image.png"
    cv2.imwrite(output_image_path, image)
    return text_results

image_path = r"C:\Users\sruth\Desktop\bounded_after_prep\05-07-2022_6-XRayTest.png"
ocr_results = process_image(image_path)

output_json_path = r"C:\Users\sruth\Desktop\bounded_after_prep\resultsmser_indivudual.json"
with open(output_json_path, 'w') as json_file:
    json.dump(ocr_results, json_file, indent=4)

print(output_json_path)

