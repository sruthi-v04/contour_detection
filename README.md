# contour_detection



## Overview
This repository implements a pipeline that combines **MSER (Maximally Stable Extremal Regions)** and **SWT (Stroke Width Transform)** techniques for efficient text region detection in X-Ray images. By leveraging these contour detection methods, along with **OCR using PaddleOCR**, the goal is to extract and recognize text from complex medical images.

This method is designed to handle rotated, skewed, and noisy X-Ray images, providing accurate OCR results for further analysis.

## Key Features
- **MSER (Maximally Stable Extremal Regions)**: Detects stable and reliable regions that are most likely to contain text, handling rotated and skewed images.
- **SWT (Stroke Width Transform)**: Enhances the detection of text strokes by analyzing the width of edge pixels in the image, useful for differentiating text from background.
- **OCR Integration**: Uses PaddleOCR to extract text from detected regions.
- **Deskewing**: Corrects skewed text regions to improve OCR accuracy.
- **Bounding Boxes**: Draws bounding boxes around detected text regions for visual validation.

## Background
This project combines the strengths of **MSER** and **SWT** to identify and process text regions in complex medical images, such as X-rays, which are often noisy and contain irregularly placed text. The techniques were tested and fine-tuned to ensure that even in challenging conditions, accurate text extraction could be achieved for further analysis.

### MSER & SWT Workflow
1. **MSER**: Identifies stable regions that are likely to contain text in an image, regardless of rotation or scale.
2. **SWT**: Refines text detection by analyzing edge pixel widths and helps to filter out irrelevant contours.
3. **Deskewing**: Detects and corrects the orientation of detected text regions.
4. **OCR**: Extracts text from deskewed text regions using PaddleOCR.

## Workflow
1. **Preprocessing**: Convert image to grayscale and apply MSER to detect stable text regions.
2. **SWT Filtering**: Apply Stroke Width Transform to filter contours and highlight potential text strokes.
3. **Deskewing**: Rotate the detected regions to correct their orientation.
4. **Text Extraction**: Run PaddleOCR to extract and recognize the text from these regions.
5. **Output**: Save processed images with bounding boxes around detected text and export OCR results in JSON format.
