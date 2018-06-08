import collections
import itertools

from patsy import dmatrices
from patsy import LookupFactor
from patsy import ModelDesc
from patsy import Term
from statsmodels.formula import api as smf

from paralysis import util


class ParameterAnalyser():
	def __init__(self, data, analysis='dominance', feature_interaction_order=1, **kwargs):
		self.label_name_ = kwargs.pop('label_name', 'y')
		self.data_ = util.load_data(d=data, **kwargs)
		self.analysis_ = analysis
		self.model_ = None
		self.param_ordering_ = None

		if (feature_interaction_order > 1):
			self.data_ = util.create_higher_order_feature_interactions(
				df=self.data_, feature_interactions=feature_interaction_order
			)

	def fit_ols(self):

		fn_analysis = getattr(self, '_fit_ols_{}'.format(self.analysis_))
		self.param_ordering_ = fn_analysis()

		return self.param_ordering_

	'''
	def create_plot(self):
		out_path = os.path.join(path_utils.get_base_path(), 'phd_thesis/_resources/')

		sns.set(style="whitegrid")
		# f, (ax1, ax2) = plt.subplots(ncols=2, sharey=False, sharex=False, figsize=(17.21, 11.86), dpi=900, facecolor='w', edgecolor='k')
		f, ax = plt.subplots(figsize=(17.21, 11.86), dpi=900, facecolor='w', edgecolor='k')
		plt.hold(True)
		plt.grid(True)

		sns.pointplot(x="pct_explained", y="parameter", hue='dataset',
					  data=df_1, join=False, palette="deep",
					  markers=["<", "v", ">", "^"], scale=2.5, ci=None)
		for tick in ax.xaxis.get_major_ticks():
			tick.label.set_fontsize(32)

		for tock in ax.yaxis.get_major_ticks():
			tock.label.set_fontsize(32)
		leg = plt.legend(bbox_to_anchor=(1., 0.37), fancybox=True, fontsize=32)
		plt.xlabel('Variance explained (%)', fontsize=42)
		plt.ylabel('Parameter', fontsize=42)
		plt.savefig(os.path.join(out_path, 'param_ablation_wordsim_2.png'), bbox_extra_artists=(leg,),
					bbox_inches='tight', ncol=3)
	'''
	
	def _fit_ols_dominance(self):
		model_subsets = self._build_subsets()

		total_results = {}
		for param in model_subsets.keys():
			results = collections.defaultdict(list)
			for model_size, paired_subsets in model_subsets[param].items():
				for ((terms_1, formula_1), (terms_2, formula_2)) in paired_subsets:
					if (formula_1 is None):
						y_dm, X_dm = dmatrices(formula_2, self.data_[terms_2], return_type='matrix')
						model = smf.OLS(y_dm, X_dm).fit()

						results[model_size].append(max(model.rsquared_adj, 0))
					else:
						y_dm_1, X_dm_1 = dmatrices(formula_1, self.data_[terms_1], return_type='matrix')
						model_1 = smf.OLS(y_dm_1, X_dm_1).fit()

						y_dm_2, X_dm_2 = dmatrices(formula_2, self.data_[terms_2], return_type='matrix')
						model_2 = smf.OLS(y_dm_2, X_dm_2).fit()

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
		for col in data_columns:
			remaining_cols = [c for c in data_columns if c != col]
			current_subsets = collections.defaultdict(list)
			for i in range(len(remaining_cols)):
				current_subsets[i].extend(list(itertools.combinations(remaining_cols, i)))

			for k in current_subsets.keys():
				for subset in current_subsets[k]:
					rhs_terms_1 = list(subset)
					formula_1 = None
					if (len(rhs_terms_1) > 0):
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


if (__name__ == '__main__'):
	#filename = '/Users/thomas/DevSandbox/InfiniteSandbox/tag-lab/paralysis/resources/example_data/snli_svm.json'
	filename = '/Users/thomas/DevSandbox/InfiniteSandbox/tag-lab/paralysis/resources/example_data/men_word2vec.json'
	pa = ParameterAnalyser(data=filename, label_name='result')
	pa.fit_ols()
	print(pa.param_ordering_)

