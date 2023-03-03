import collections
import itertools

from patsy import LookupFactor
from patsy import ModelDesc
from patsy import Term
from statsmodels.formula import api as smf
from tqdm import tqdm
import pandas as pd

from paralysis import util


class ParameterAnalyser():
	def __init__(self, data, analysis='dominance', feature_interaction_order=1, **kwargs):
		self.label_name_ = kwargs.pop('label_name', 'y')
		self.data_ = util.load_data(d=data, **kwargs)
		self.analysis_ = analysis
		self.model_ = None
		self.parameter_table_ = None

		if feature_interaction_order > 1:
			self.data_ = util.create_higher_order_feature_interactions(
				df=self.data_, feature_interactions=feature_interaction_order
			)

	def fit_ols(self):

		fn_analysis = getattr(self, '_fit_ols_{}'.format(self.analysis_))
		param_ordering = fn_analysis()

		tbl = []
		for k, v in param_ordering.items():
			tbl.append({'parameter': k, 'weight': v})
		self.parameter_table_ = pd.DataFrame.from_dict(tbl).sort_values(by='weight', ascending=False)

		return self.parameter_table_

	def _fit_ols_dominance(self):
		# Who would have thought, but we're actually doing Shapely analysis here: https://www.displayr.com/shapley-value-regression/
		model_subsets = self._build_subsets()

		total_results = {}
		for param in tqdm(model_subsets.keys(), desc="Estimating dominance"):
			results = collections.defaultdict(list)
			for model_size, paired_subsets in tqdm(model_subsets[param].items(), desc=f"\tProcessing paired subsets"):
				for ((terms_1, formula_1), (terms_2, formula_2)) in tqdm(paired_subsets):
					if formula_1 is None:
						model = smf.ols(formula_2, self.data_[terms_2]).fit()

						results[model_size].append(max(model.rsquared_adj, 0))
					else:
						model_1 = smf.ols(formula_1, self.data_[terms_1]).fit()
						model_2 = smf.ols(formula_2, self.data_[terms_2]).fit()

						results[model_size].append(max(model_2.rsquared_adj, 0) - max(model_1.rsquared_adj, 0))

			# Collect total results and average
			c = 0
			for model_size, result_list in results.items():
				c += sum(result_list) / len(result_list)
			total_results[param] = max(c / len(results), 0)

		# Normalise total results
		n = sum(total_results.values())
		for param, value in total_results.items():
			total_results[param] = value / n

		return total_results

	def _build_subsets(self):
		data_columns = [c for c in self.data_.columns.values.tolist() if c != self.label_name_]

		subsets = collections.defaultdict(lambda: collections.defaultdict(list))
		for col in tqdm(data_columns, desc="Building Subsets"):
			remaining_cols = [c for c in data_columns if c != col]
			current_subsets = collections.defaultdict(list)
			for i in range(len(remaining_cols)):
				current_subsets[i].extend(list(itertools.combinations(remaining_cols, i)))

			for k in current_subsets.keys():
				for subset in current_subsets[k]:
					rhs_terms_1 = list(subset)
					formula_1 = None
					if len(rhs_terms_1) > 0:
						formula_1 = ModelDesc(
							[Term([LookupFactor(self.label_name_)])],  # LHS
							[Term([LookupFactor(fn, force_categorical=True)]) for fn in rhs_terms_1]
						)

					rhs_terms_2 = list(subset) + [col]

					formula_2 = ModelDesc(
						[Term([LookupFactor(self.label_name_)])],  # LHS
						[Term([LookupFactor(fn, force_categorical=True)]) for fn in rhs_terms_2]
					)

					pair_1 = (rhs_terms_1 + [self.label_name_], formula_1)
					pair_2 = (rhs_terms_2 + [self.label_name_], formula_2)
					subsets[col][k].append((pair_1, pair_2))
		return subsets


if __name__ == '__main__':
	#filename = '/Users/thomas/DevSandbox/InfiniteSandbox/tag-lab/paralysis/resources/example_data/snli_svm.json'
	filename = '/Users/thomas/DevSandbox/InfiniteSandbox/tag-lab/paralysis/resources/example_data/simlex_word2vec.json'
	pa = ParameterAnalyser(data=filename, label_name='result')
	pa.fit_ols()
	print(pa.parameter_table_)

