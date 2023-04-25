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
img = cv2.imread('data/signinsheets/SS3.jpeg')

# Convert the image into a grayscale image
# img = cv2.medianBlur(img, 3)
img = cv2.GaussianBlur(img, (5,5), 0)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)

coords = cv2.findNonZero(thresh)
angle_skewed = cv2.minAreaRect(coords)[-1]
if angle_skewed < -45:
     angle_skewed = -(90 + angle_skewed)

else:
     angle_skewed = -angle_skewed


(h, w) = thresh.shape[:2]
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, angle_skewed, 1.0)
gray = cv2.warpAffine(thresh, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

# Use pytesseract to extract the text from the image
lines = cv2.HoughLinesP(gray, 1, 1*np.pi/180, 100, minLineLength=100, maxLineGap=10)

for line in lines:
     x1, y1, x2, y2 = line[0]
     cv2.line(gray, (x1, y1), (x2, y2), (255, 255, 255), 3)

words = pytesseract.image_to_data(gray, lang='eng', config='--psm 12', output_type=pytesseract.Output.DICT)

for i in range(len(words['text'])):
     if int(words['conf'][i]) > 60:
          x, y, w, h = words['left'][i], words['top'][i], words['width'][i], words['height'][i]
          cv2.rectangle(gray, (x, y), (x + w, y + h), (255, 255, 255), -1)

results = []
for i in range(len(words['text'])):
    text = words['text'][i]
    results.append(text)
final_result = ' '.join(results)

print(final_result)
