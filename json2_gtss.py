import pandas as pd 
from tqdm import tqdm
import json

path = '../scratch/data/MAVI-signboard-new-split/train_annotations.json'
df = pd.read_json(path)

dataset = {} 
dataset['images'] = []
dataset['annotations'] = []
dataset['categories'] = [{'class': 'text'}] 

for item in tqdm(range(len(df))):
	dataset['images'].append({
					'filename': df['filename'][item], 
 				})
	anno_lst = []
	for dct in df['regions'][item]:
		if 'type' in dct['region_attributes']: 
			if dct['region_attributes']['type'] in ['h', 'e', 'a', 'H']:
				lst_x = dct['shape_attributes']['all_points_x']
				lst_x = list(map(int, lst_x))
				lst_y = dct['shape_attributes']['all_points_y']
				lst_y = list(map(int, lst_y))
				# for maintaining the coco format..annotations are (xmin, ymin, width, height)
				width = max(lst_x) - min(lst_x)
				height = max(lst_y) - min(lst_y)
				anno_lst.append({
					'bbox': [min(lst_x), min(lst_y), width, height], 
					'name': dct['shape_attributes']['name']
				})
				# print(anno_lst)
	dataset['annotations'].append(anno_lst)		

print(len(dataset['annotations']) == len(dataset['images']))
print('Json file buildup')
with open('../scratch/data/MAVI-signboard-new-split/train_annotations_roundup.json', 'w') as f: 
	json.dump(dataset, f, indent=4)
