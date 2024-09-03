from flask.views import MethodView
from flask import Flask, jsonify, request
import cv2
import numpy as np
import glob
import os

app = Flask(__name__) 
base_path = os.getcwd()

def translate(image, x, y):
	# Define the translation matrix and perform the translation
	M = np.float32([[1, 0, x], [0, 1, y]])
	shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

	# Return the translated image
	return shifted

def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))

def rotate(image, angle, center = None, scale = 1.0):
	# Grab the dimensions of the image
	(h, w) = image.shape[:2]

	# If the center is None, initialize it as the center of
	# the image
	if center is None:
		center = (w / 2, h / 2)

	# Perform the rotation
	M = cv2.getRotationMatrix2D(center, angle, scale)
	rotated = cv2.warpAffine(image, M, (w, h))

	# Return the rotated image
	return rotated

def resize(image, width = None, height = None, inter = cv2.INTER_AREA):
	# initialize the dimensions of the image to be resized and
	# grab the image size
	dim = None
	(h, w) = image.shape[:2]

	# if both the width and height are None, then return the
	# original image
	if width is None and height is None:
		return image

	# check to see if the width is None
	if width is None:
		# calculate the ratio of the height and construct the
		# dimensions
		r = height / float(h)
		dim = (int(w * r), height)

	# otherwise, the height is None
	else:
		# calculate the ratio of the width and construct the
		# dimensions
		r = width / float(w)
		dim = (width, int(h * r))

	# resize the image
	resized = cv2.resize(image, dim, interpolation = inter)

	# return the resized image
	return resized

def calculate_average_max_values(data_list):
    total_max_values = {}
    max_value_counts = {}

    for data in data_list:
        denomination = data['denomination']
        max_value = data['max_value']

        if denomination not in total_max_values:
            total_max_values[denomination] = 0
            max_value_counts[denomination] = 0
        total_max_values[denomination] += max_value
        max_value_counts[denomination] += 1

    average_max_vals = {
         denomination: total_max_value / max_value_counts[denomination] for denomination, total_max_value in total_max_values.items()
    }

    return average_max_vals

def match_currency():
    # Load template files
    templates = []
    template_path = os.path.join(base_path,'templates' , '*', '*', '*.jpg')
    template_paths = glob.glob(template_path, recursive=True)
    print("Templates loaded: ", template_paths)

    # Prepare templates
    for path in template_paths:
        image = cv2.imread(path)
        image = resize(image, width=int(image.shape[1] * 0.5))  # Scaling
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale        
        normalized_path = os.path.normpath(path)  # Normalize path to eliminate double slashes
        directory, file_name = os.path.split(normalized_path)
        denomination, _ = os.path.split(directory)
        templates.append({
            "image": image,
            "denomination": os.path.basename(denomination),
            "max_value": 0.0
        })
    
    
    # template matching
    for image_path in glob.glob('sample/*.jpg'):
        for template in templates:
            image_read = cv2.imread(image_path)
            resized_image = cv2.resize(image_read, (1200, 1600))# avoid corrupt image cause high resolution
            (template_height, template_width) = template['image'].shape[:2]

            gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

            found = None
            for scale in np.linspace(0.2, 1.0, 20)[::-1]:
                # scalling image
                scaled_image = resize(
                    gray_image, width=int(gray_image.shape[1] * scale))
                ratio = gray_image.shape[1] / float(scaled_image.shape[1])
                if scaled_image.shape[0] < template_height or scaled_image.shape[1] < template_width:
                    break

                # template matching
                result = cv2.matchTemplate(scaled_image, template['image'], cv2.TM_CCOEFF_NORMED)
                (_, max_value, _, max_location) = cv2.minMaxLoc(result)
                if found is None or max_value > found[0]:
                    found = (max_value, max_location, ratio)
            if found is not None:
                (max_value, max_location, ratio) = found
                (start_x, start_y) = (int(max_location[0] * ratio), int(max_location[1] * ratio))
                (endX, endY) = (
                    int((max_location[0] + template_width) * ratio), int((max_location[1] + template_height) * ratio))                 
                template['max_value'] = max_value
                cv2.rectangle(image_read, (start_x, start_y),
                                (endX, endY), (0, 0, 255), 2)

    match_data = calculate_average_max_values(templates)
    return match_data

@app.route('/')
def home():
     return 'Welcome to money detection :)'

@app.route('/api/upload', methods=['POST'])

def upload_image():
    image_file = request.files['image']
    image_file.save('./sample/image.jpg')

    result = {'is_money_detected': False}
    thershold = 0.48
    match_data = match_currency()
    

    max_denomination = max(match_data, key=match_data.get)
    max_average_value = match_data[max_denomination]

    result['denomination'] = ''

    if max_average_value > thershold:
         result['denomination'] = max_denomination
         result['is_money_detected'] = True

    result['max_val'] = max_average_value
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)