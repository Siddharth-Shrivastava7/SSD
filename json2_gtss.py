import pandas as pd 
from tqdm import tqdm
import json

path = '../scratch/data/MAVI-signboard-new-split/train_annotations.json'
df = pd.read_json(path)

dataset = {} 
dataset["type"] = "instances"
dataset['images'] = []
dataset['categories'] = [{"supercategory": "none","id": 1,"name": "text"}] 
dataset['annotations'] = []


ind = 0
img_id = 0
for item in tqdm(range(len(df))):
	dataset['images'].append({
					'file_name': df['filename'][item], 
					'id': img_id,
					'height': 480,
					'width': 640
 				})
	img_id = img_id + 1
	# anno_lst = []
	for dct in df['regions'][item]:
		if 'type' in dct['region_attributes']: 
			# if dct['region_attributes']['type'] in ['h', 'e', 'a', 'H']:
			if dct['region_attributes']['type'] in ['s','ss']:
				ind = ind + 1 
				# lst_x = dct['shape_attributes']['all_points_x']
				# lst_x = list(map(int, lst_x))
				# lst_y = dct['shape_attributes']['all_points_y']
				# lst_y = list(map(int, lst_y))
				# for maintaining the coco format..annotations are (xmin, ymin, width, height)
				# width = max(lst_x) - min(lst_x)
				# height = max(lst_y) - min(lst_y)
				# area = width * height
				width = dct['shape_attributes']['width']
				height = dct['shape_attributes']['height']
				x = dct['shape_attributes']['x']
				y = dct['shape_attributes']['y']
				area = height * width
				# 'bbox': [min(lst_x), min(lst_y), width, height],
				dataset['annotations'].append({
					'id': ind, 
					'bbox': [x, y, width, height], 
					'image_id': img_id,
					'segmentation': [],
					'ignore': 0,
					'area': area, 
					'iscrowd': 0, 
					'category_id': 1
				})
				# print(anno_lst)
	# dataset['annotations'].append(anno_lst)		

# print(len(dataset['annotations']) == len(dataset['images']))
print('Json file buildup')
with open('../scratch/data/MAVI-signboard-new-split/train_annotations_roundup_sign.json', 'w') as f: 
	json.dump(dataset, f, indent=4)
