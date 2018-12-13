import xml.etree.ElementTree as ET
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from skimage.io import imsave

OFFSET = 0.3
SCALE = 64

def read_file(filename):
	tree = ET.parse(filename)
	root = tree.getroot()

	raw_points = []

	for point in root.find('object').find('polygon').findall('pt'):
		x = point.find('x').text
		y = point.find('y').text

		raw_points.append([int(x), int(y)])

	# Normalize
	points = []

	raw_points = np.array(raw_points)
	# import pdb; pdb.set_trace()

	min_x = np.min(raw_points[:,0])
	min_y = np.min(raw_points[:,1])
	max_x = np.max(raw_points[:,0])
	max_y = np.max(raw_points[:,1])

	height = max_y - min_y
	width = max_x - min_x

	crop_x = (max_x + min_x)*0.5 - 0.75*(max_x - min_x)
	crop_y = (max_y + min_y)*0.5 - 0.75*(max_y - min_y)

	for x, y in raw_points:
		tx = 1.0*(x - min_x) / (max_x - min_x) - 0.5
		ty = 1.0*(y - min_y) / (max_y - min_y) - 0.5 
		rx = (tx + 0.5) * SCALE
		ry = (ty + 0.5) * SCALE
		points.append([x,y,rx,ry])

	return points


if __name__ == '__main__':
	path = 'Annotations/users/dianchen96/393r_shape'

	for i, file in enumerate(os.listdir(path)):
		points = read_file(os.path.join(path, file))

		polygon = []

		with open('Points/%d.txt'%i, 'w+') as file:
			xs = []
			ys = []
			for x,y, rx, ry in points:
				file.write("%s %s\n"%(x,y))
				polygon.append((rx,ry))
			
		img = Image.new('L', (64, 64), 0)
		ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
		mask = np.array(img)*255

		imsave('Masks/%d.png'%i, mask)

			# plt.imshow(mask)
			# plt.show()
			# plt.scatter(xs, ys)
			# plt.show()