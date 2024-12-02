import cv2
import numpy as np

def compute_gradient(image):
    # Calculate gradients in x and y direction
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    return grad_x, grad_y

def compute_swt(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    
    height, width = edges.shape
    swt_image = np.zeros((height, width), dtype=np.float32)

    grad_x, grad_y = compute_gradient(gray)

    for y in range(height):
        for x in range(width):
            if edges[y, x] == 255:  # Edge pixel
                theta = np.arctan2(grad_y[y, x], grad_x[y, x])
                dx, dy = np.cos(theta), np.sin(theta)
                stroke_width = 0

                # Traverse along the gradient direction
                for k in range(1, max(height, width)):
                    nx, ny = int(x + k * dx), int(y + k * dy)
                    if 0 <= nx < width and 0 <= ny < height:
                        if edges[ny, nx] == 255:  # Found another edge
                            stroke_width = k
                            break
                
                if stroke_width > 0:
                    swt_image[y, x] = stroke_width

    return swt_image

def filter_strokes(swt_image):
    # Normalize the SWT image for better visualization
    swt_norm = cv2.normalize(swt_image, None, 0, 255, cv2.NORM_MINMAX)
    _, swt_thresh = cv2.threshold(swt_norm.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Apply morphological dilation to thicken the lines
    kernel = np.ones((3, 3), np.uint8)
    swt_thickened = cv2.dilate(swt_thresh, kernel, iterations=1)
    
    return swt_thickened



def process_image(image_path, output_path):
    image = cv2.imread(image_path)
    swt_image = compute_swt(image)
    filtered_strokes = filter_strokes(swt_image)

    # Display images
    cv2.imshow("Original Image", image)
    cv2.imshow("Stroke Width Transform", filtered_strokes)
    
    # Save the filtered strokes image
    cv2.imwrite(output_path, filtered_strokes)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

image_path = r"C:\Users\sruth\Desktop\bounded_after_prep\05-07-2022_6-XRayTest.png"
output_path = r"C:\Users\sruth\Desktop\bounded_after_prep\filtered_strokes.png"
process_image(image_path, output_path)
