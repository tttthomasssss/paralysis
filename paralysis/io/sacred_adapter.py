import json
import os


def load_data_from_file_storage(path, parameters):
	results = []
	for result_dir in os.listdir(path):
		try:
			int(result_dir)
		except ValueError:
			continue

		with open(os.path.join(path, result_dir, 'run.json')) as in_file:
			run = json.load(in_file)

		if (run['status'] == 'COMPLETED'):
			with open(os.path.join(path, result_dir, 'config.json')) as in_file:
				config = json.load(in_file)

			config['result'] = run['result']

			for k in set(config.keys()) - set(parameters):
				del config[k]

			results.append(config)

	return results