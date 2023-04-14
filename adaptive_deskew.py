import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract
from nltk import sent_tokenize
import numpy as np
import re
from matplotlib import pyplot as plt
from skimage import io
from skimage.transform import rotate
from skimage.color import rgb2gray
#from deskew import determine_skew


# Load the image
img = cv2.imread('data/signinsheets/SS1.png')

# Convert the image into a grayscale image
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.medianBlur(img, 3)

# denoising of image
# dst = cv2.fastNlMeansDenoisingColored(img, None, 10,10,7,15)

# Apply thresholding to the image
# thresh1 = cv2.threshold(img, 158, 255,  cv2.THRESH_BINARY_INV)[1]
# Apply adaptive thresholding to the image
# thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)


# Increase the size of the structuring element
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))


# # # Apply morphological operations
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
next = cv2.morphologyEx(opening, cv2.MORPH_RECT, kernel)
closing = cv2.morphologyEx(next, cv2.MORPH_CLOSE, kernel)

# # Find the skew angle of the image
coords = np.column_stack(np.where(closing > 0))
angle = cv2.minAreaRect(coords)[-1]

# # # Define the region of interest as a rectangular mask
(h, w) = img.shape[:2]
x, y= 500, 500
mask = np.zeros(closing.shape, dtype=np.uint8)
mask[y:y+h, x:x+w] = 200

# # # Compute the rotation matrix
center = (x + w//2, y + h//2)
M = cv2.getRotationMatrix2D(center, angle, 1.0)

# # # Apply the rotation only to the pixels within the ROI
roi = cv2.bitwise_and(closing, closing, mask=mask)
rotated_roi = cv2.warpAffine(roi, M, (closing.shape[1], closing.shape[0]))



# # Combine the rotated ROI with the original image outside the ROI
result = closing.copy()
#result[y:y+h, x:x+w] = rotated_roi[y:y+h, x:x+w]

# plt.imshow(result)
# plt.show()
# Apply pytesseract with custom configurations
custom_config = '-l eng --psm 6'
data = pytesseract.image_to_data(result, output_type=pytesseract.Output.DICT, config=custom_config)

# #Post-processing of OCR results
results = []
for i in range(len(data['text'])):
    text = data['text'][i]
    results.append(text)
final_result = ' '.join(results)


print(final_result)
