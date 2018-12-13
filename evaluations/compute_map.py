import os
import numpy as np
import xml.etree.ElementTree as ET


import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

GT_PATH = '../evaluations_gt/Annotations/users/dianchen96/393r_eval_logs'

def load_gt_annotation():
	points = {}

	for i in range(89):
		img_num = 2*i
		filename = os.path.join(GT_PATH, '%d_raw.xml'%img_num)
		tree = ET.parse(filename)
		root = tree.getroot()

		# import pdb; pdb.set_trace()

		_points = []

		for obj in root.findall('object'):
			p = obj.find('polygon').findall('pt')
			bbox = []
			for point in p:
				x = point.find('x').text
				y = point.find('y').text

				bbox.append([int(x), int(y)])

			_points.append(bbox)

		points[img_num] = _points

	return points


def load_pred_annotation(path, mask_num):
	points = {}

	for i in range(89):
		img_num = 2*i
		filename = os.path.join('%d_%s'%(mask_num, path), '%d_pred.txt'%img_num)

		_points = []
		with open(filename, 'r') as file:
			lines = file.read().splitlines()
			for l in lines:
				bbox = []
				for xy_str in l.strip().split(';')[:4]:
					x, y = xy_str.split(',')
					bbox.append([int(x), int(y)])
				_points.append(bbox)

		points[img_num] = _points

	return points


def compute_iou(gt_bbox, pred_bbox):
	# print (gt_bbox, pred_bbox)

	[gt_x2, gt_y2], _, [gt_x1, gt_y1], _ = gt_bbox
	[pred_x1, pred_y1], _, _, [pred_x2, pred_y2] = pred_bbox

	x1 = max(gt_x1, pred_x1)
	x2 = min(gt_x2, pred_x2)
	y1 = max(gt_y1, pred_y1)
	y2 = min(gt_y2, pred_y2)

	if x2 < x1 or y2 < y1:
		return 0.0

	intersection = (x2 - x1) * (y2 - y1)
	gt_area = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)
	pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)

	return intersection / float(gt_area + pred_area - intersection)


def get_pr(path, mask_num, iou_t=0.75):

	gt_points = load_gt_annotation()
	pred_points = load_pred_annotation(path, mask_num)

	img_names = gt_points.keys()

	precisions = []
	recalls = []

	for img_name in img_names:

		gt = gt_points[img_name]
		pred = pred_points[img_name]

		correct = 0
		for pred_point in pred:
			for gt_point in gt:
				iou = compute_iou(gt_point, pred_point)
				# print (iou)
				if iou > iou_t:
					correct += 1

		recalls.append(correct/float(len(gt)))
		if len(pred) > 0:
			precisions.append(correct/float(len(pred)))
		

	return np.mean(precisions), np.mean(recalls)



	# return 

	# print (gt_points.keys())

def plot_precision_recall(paths, iou_t=0.75, mask_nums=[3,10], colors=['go-','bo-']):

	gt_points = load_gt_annotation()

	fig, ax = plt.subplots(1,1)


	for mask_num, color in zip(mask_nums, colors):
		precision = []
		recall = []


		for path in paths:
			p, r = get_pr(path, mask_num, iou_t=iou_t)

			precision.append(p)
			recall.append(r)

			## Assuming monotonic
		print (mask_num, "mAP@0.5:%.5f" % np.mean([precision]))

		ax.plot(recall, precision, color)
	
	ax.set_xlabel('Recall at IoU=%.2f'%iou_t)
	ax.set_ylabel('Precision at IoU=%.2f'%iou_t)
	ax.legend(['10 Masks', '3 Masks'])


	plt.show()



if __name__ == '__main__':

	plot_precision_recall([
		'050',
		'052',
		'055',
		'057',
		'060',
	])

	# print (load_gt_annotation()[0])
	# print (load_pred_annotation('050')[0])

	# print (compute_iou(load_gt_annotation()[2][0], load_pred_annotation('050')[2][0]))
