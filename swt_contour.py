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
    
    # MSER detector initialization
    mser = cv2.MSER_create()
    regions, _ = mser.detectRegions(gray)
    
    # Contours processing
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    text_results = []
    output_image = image.copy()

    # Combine MSER and contours by iterating through both
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        if w > 10 and h > 10:
            cropped = image[y:y+h, x:x+w]
            deskewed = deskew(cropped)
            deskewed_rgb = cv2.cvtColor(deskewed, cv2.COLOR_BGR2RGB)
            ocr_result = ocr_engine.ocr(deskewed_rgb, cls=True)

            if ocr_result and ocr_result[0]:
                for line in ocr_result[0]:
                    text = line[1][0]
                    text_results.append({
                        "bounding_box": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)},
                        "text": text.strip()
                    })
                    cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    for region in regions:
        hull = cv2.convexHull(region.reshape(-1, 1, 2))
        x, y, w, h = cv2.boundingRect(hull)

        if w > 10 and h > 10:
            cropped = image[y:y+h, x:x+w]
            deskewed = deskew(cropped)
            deskewed_rgb = cv2.cvtColor(deskewed, cv2.COLOR_BGR2RGB)
            ocr_result = ocr_engine.ocr(deskewed_rgb, cls=True)

            if ocr_result and ocr_result[0]:
                for line in ocr_result[0]:
                    text = line[1][0]
                    text_results.append({
                        "bounding_box": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)},
                        "text": text.strip()
                    })
                    cv2.rectangle(output_image, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue for MSER boxes

    # Save the output image with boxes and OCR results
    output_image_path = r"C:\Users\sruth\Desktop\bounded_after_prep\output_with_boxes_contour_mser.png"
    cv2.imwrite(output_image_path, output_image)

    return text_results, output_image_path

image_path = r"C:\Users\sruth\Desktop\bounded_after_prep\05-07-2022_6-XRayTest.png"
ocr_results, output_image_path = process_image(image_path)

output_json_path = r"C:\Users\sruth\Desktop\bounded_after_prep\contours_mser_ocr_results.json"
with open(output_json_path, 'w') as json_file:
    json.dump(ocr_results, json_file, indent=4)

print("Output image with bounding boxes:", output_image_path)
print("OCR results saved to:", output_json_path)




