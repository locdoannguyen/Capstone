import csv
import pytesseract
from PIL import Image

# Open the image
image = Image.open("data/r1.jpg")

# Use pytesseract to extract the text from the image
text = pytesseract.image_to_string(image)

# Split the text into lines and save it in a list
lines = text.split("\n")

# Create a CSV file and write the data
with open("data/ss_data.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows([lines])
