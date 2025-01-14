'''
Copyright 2017 TensorFlow Authors and Kent Sommer

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''
from keras import backend as K
import inception_v4
import numpy as np
import cv2
import os
import sys
import train

# If you want to use a GPU set its index here
os.environ['CUDA_VISIBLE_DEVICES'] = ''


# This function comes from Google's ImageNet Preprocessing Script
def central_crop(image, central_fraction):
	"""Crop the central region of the image.
	Remove the outer parts of an image but retain the central region of the image
	along each dimension. If we specify central_fraction = 0.5, this function
	returns the region marked with "X" in the below diagram.
	   --------
	  |        |
	  |  XXXX  |
	  |  XXXX  |
	  |        |   where "X" is the central 50% of the image.
	   --------
	Args:
	image: 3-D array of shape [height, width, depth]
	central_fraction: float (0, 1], fraction of size to crop
	Raises:
	ValueError: if central_crop_fraction is not within (0, 1].
	Returns:
	3-D array
	"""
	if central_fraction <= 0.0 or central_fraction > 1.0:
		raise ValueError('central_fraction must be within (0, 1]')
	if central_fraction == 1.0:
		return image

	img_shape = image.shape
	depth = img_shape[2]
	fraction_offset = int(1 / ((1 - central_fraction) / 2.0))
	bbox_h_start = int(np.divide(img_shape[0], fraction_offset))
	bbox_w_start = int(np.divide(img_shape[1], fraction_offset))

	bbox_h_size = int(img_shape[0] - bbox_h_start * 2)
	bbox_w_size = int(img_shape[1] - bbox_w_start * 2)

	image = image[bbox_h_start:bbox_h_start+bbox_h_size, bbox_w_start:bbox_w_start+bbox_w_size]
	return image


def get_processed_image(img_path):
        # Load image and convert from BGR to RGB
        im = cv2.imread(img_path)
        if im is None:
            return None
        im = np.asarray(im)[:,:,::-1]
        im = central_crop(im, 0.875)
        im = cv2.resize(im, (299, 299))
        im = inception_v4.preprocess_input(im)
        if K.image_data_format() == "channels_first":
            im = np.transpose(im, (2,0,1))
        return im


if __name__ == "__main__":
	# Create model and load pre-trained weights
    model = inception_v4.create_model(num_classes=train.nb_classes, include_top=True)
    model.load_weights(train.weights_file, by_name=True)

    # Load test image!
    imgs = []
    x_ids = []
    y_ids = []
    if sys.argv[1] != "":
        img_file = sys.argv[1]
        with open(img_file, 'r') as id_file:
            ids = id_file.readlines()
            for img in ids:
                x, y = img.strip().split(",")
                img = get_processed_image(os.path.join(train.image_path_prefix, x[0], x[1], x[2], x + '.jpg'))
                if img is not None:
                    imgs.append(img)
                    y_ids.append(y)
                    x_ids.append(x)
    else:
        raise "specify the test image id file!"

    # Run prediction on test image
    preds = model.predict(np.asarray(imgs))

    for idx in range(0, len(imgs)):
        print("%s\t%s\t%f\t%d\t%f" % (x_ids[idx], y_ids[idx], preds[idx][int(y_ids[idx])], np.argmax(preds[idx]), preds[idx][np.argmax(preds[idx])]))
    
#    print("Class is: " + classes[np.argmax(preds)-1])
#    print("Certainty is: " + str(preds[0][np.argmax(preds)]))
