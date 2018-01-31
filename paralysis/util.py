from functools import reduce
import itertools
import json

import numpy as np
import pandas as pd


def load_data(d, **kwargs):
	df = None
	if (isinstance(d, pd.DataFrame)):  # Best case, received a pandas DataFrame as input to begin with
		df = d
	elif (isinstance(d, str) and d.endswith(
			'.csv')):  # Expect path to a file containing all observations, with the first line containing the names of the parameters
		df = pd.DataFrame.from_csv(d)
	elif (isinstance(d, str) and d.endswith(
			'.json')):  # Expect a json file (list of dicts really), where each item corresponds to a single observation
		with open(d) as in_file:
			df = pd.DataFrame.from_dict(json.load(in_file))
	elif (isinstance(d, str)):
		data_reader = kwargs.pop('data_reader', np.loadtxt)
		param_names = kwargs.pop('parameter_names')
		param_index = kwargs.pop('parameter_index', None)

		X = data_reader(**kwargs)
		df = pd.DataFrame(data=X, columns=param_names, index=param_index)
	elif (isinstance(d, tuple)):  # Expect 2 numpy arrays (X, y), representing data and labels
		X, y = d
		param_names = kwargs.pop('parameter_names')
		param_index = kwargs.pop('parameter_index', None)

		df = pd.DataFrame(data=np.hstack((X, y.reshape(-1, ))), columns=param_names, index=param_index)
	elif (isinstance(d, np.ndarray)):  # Expect 1 numpy array containing data and labels
		param_names = kwargs.pop('parameter_names')
		param_index = kwargs.pop('parameter_index', None)

		df = pd.DataFrame(data=d, columns=param_names, index=param_index)
	elif (isinstance(d, list)):  # Expect a list of dicts, where every dict is a single observation - i.e. expect the same as from a json file but already loaded
		df = pd.DataFrame.from_dict(d)

	return df


def create_higher_order_feature_interactions(df, feature_interactions=2):
	new_data = []
	columns = df.columns.values.tolist()

	for index, row in df.iterrows():
		item = row.to_dict()
		for feat_combination in itertools.combinations(columns, feature_interactions):
			feat_name = '__'.join(feat_combination)
			feat_val = reduce(lambda x, y: x + '-{}'.format(item[y]), feat_combination, '')[1:]

			item[feat_name] = feat_val
		new_data.append(item)

	return pd.DataFrame.from_dict(new_data)
