import cv2
from skimage.metrics import structural_similarity as ssim
import numpy as np
import sys

# Allow passing image paths as arguments; fall back to default names
original_path  = sys.argv[1] if len(sys.argv) > 1 else 'sample.png'
processed_path = sys.argv[2] if len(sys.argv) > 2 else 'output.png'

original  = cv2.imread(original_path)
processed = cv2.imread(processed_path)

if original is None:
    raise FileNotFoundError(f"Could not load original image: {original_path}")
if processed is None:
    raise FileNotFoundError(f"Could not load processed image: {processed_path}")

gray1 = cv2.cvtColor(original,  cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)

(score, diff) = ssim(gray1, gray2, full=True)
print(f"SSIM Score: {score}")

diff = (diff * 255).astype("uint8")

cv2.imshow("Difference", diff)
cv2.waitKey(0)
cv2.destroyAllWindows()

