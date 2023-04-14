import cv2
import pytesseract
from nltk import sent_tokenize
import numpy as np
import re


# Load the image
img = cv2.imread('data/signinsheets/SS3.jpeg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.medianBlur(img, 3)

# Apply thresholding to the image
thresh = cv2.threshold(img, 158, 255,  cv2.THRESH_BINARY_INV)[1]
# Apply adaptive thresholding to the image
thresh = cv2.adaptiveThreshold(thresh, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 99, 5)

# thresh = cv2.adaptiveThreshold(thresh, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 3)

# Increase the size of the structuring element
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))

# Apply morphological operations
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
next = cv2.morphologyEx(opening, cv2.MORPH_RECT, kernel)
closing = cv2.morphologyEx(next, cv2.MORPH_CLOSE, kernel)

# # Find the skew angle of the image
# coords = np.column_stack(np.where(closing > 0))
# # print(coords)
# angle = cv2.minAreaRect(coords)[-1]
# # print(angle)
# # Rotate the image to correct for skew
# # if angle < 90:
# #     angle = angle
# #     print("in here")
# # else:
# #     angle = -angle
# (h, w) = img.shape[:2]
# print(h)
# print(w)
# center = (w // 2, h // 2)
# M = cv2.getRotationMatrix2D(center, angle, 1.0)
# closing = cv2.warpAffine(closing, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

# Apply pytesseract with custom configurations
custom_config = '-l eng --psm 6'
data = pytesseract.image_to_data(closing, output_type=pytesseract.Output.DICT, config=custom_config)

# Post-processing of OCR results
results = []
for i in range(len(data['text'])):
    text = data['text'][i]
    # if len(text) > 0 and text != ' ':
    results.append(text)
final_result = ' '.join(results)


print(final_result)
# Extract lines of text from the OCR data
# lines = []
# for i, line_text in enumerate(data['text']):
#     if data['conf'][i] > 60:
#         line_num = data['line_num'][i]
#         if line_num > len(lines):
#             lines.extend([""] * (line_num - len(lines)))
#         lines[line_num-1] += f"{line_text} "

# # Tokenize the entire text into sentences
# text = ' '.join(lines)
# sentences = sent_tokenize(text)

# # Print the extracted text
# print("--------------------------")
# print(sentences)
# print("--------------------------")
