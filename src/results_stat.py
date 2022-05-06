import json
import numpy as np
import copy
import glob
import os


def get_exact_match(resutls):

	api_names = ["search", "adjust_color", "adjust_attr", "image_cutout", "rotate"]
	api_names_em = ["search", "adjust_color", "adjust_attr brightness", "adjust_attr contrast", "image_cutout", "rotate"]
	api_cnt_match = [0,0,0,0,0,0]
	api_cnt_match_only_api = [0,0,0,0,0]
	api_cnt_total = [0,0,0,0,0,0]


	y_true = []
	y_pred = []

	search = 0
	crop = 0
	adjust_color = 0
	adjust_attr = 0
	image_cutout = 0
	rotate = 0

	for img_id, apis_tuple in resutls.items():

		apis = [apis_tuple[0].lower(), apis_tuple[1].lower()]
		api_gt = apis[1].split()[0]


		for i, api_name in enumerate(api_names_em):
			if api_gt == "adjust_attr":
				api_gt_new = api_gt + " " + apis[1].split()[1]
			else: 
				api_gt_new = api_gt
			if api_gt_new == api_name:
				api_cnt_total[i] += 1

		if apis[0] == "":
			continue

		api_gen = apis[0].split()[0]

		if api_gen == "" or (len(apis[0].split()) == 1 and api_gen != "image_cutout"):
			continue


		

		if api_gt == "crop":
			continue

		if api_gt == "adjust_attr":
			y_true.append(api_gt + apis[1].split()[1])
		else:
			y_true.append(api_gt)

		if api_gen == "adjust_attr":
			y_pred.append(api_gen + apis[0].split()[1])
		else:
			y_pred.append(api_gen)
		
		for i, api_name in enumerate(api_names_em):
			if api_gen == "adjust_attr":
				api_gen_new = api_gen + " " + apis[0].split()[1]
			else: 
				api_gen_new = api_gen

		iscolor = False
		if api_gt == "adjust_color" and api_gen == "adjust_color":
			api_gt_color = api_gt + " " + apis[1].split()[1]
			api_gen_color = api_gen + " " + apis[0].split()[1]

			if api_gt_color == api_gen_color:
				iscolor = True

		if False or apis[0].strip() == apis[1].strip() or iscolor:
			for i, api_name in enumerate(api_names_em):
				if api_gen_new == api_name:
					api_cnt_match[i] += 1
					if "search" in api_gen and "_0" not in img_id:
						pass

		if False or api_gen.strip() == api_gt.strip():
			
			for i, api_name in enumerate(api_names):
				if api_gen == api_name:
					api_cnt_match_only_api[i] += 1

	return 1.0 * sum(api_cnt_match) / sum(api_cnt_total)
