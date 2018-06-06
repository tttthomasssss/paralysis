import collections
import itertools

from patsy import dmatrices
from patsy import LookupFactor
from patsy import ModelDesc
from patsy import Term
from sklearn.linear_model import LinearRegression
from statsmodels import api as sm
from statsmodels.formula import api as smf


from paralysis import util


class ParameterAnalyser():
	def __init__(self, data, analysis='dominance', feature_interaction_order=1, **kwargs):
		self.label_name_ = kwargs.pop('label_name', 'y')
		self.data_ = util.load_data(d=data, **kwargs)
		self.analysis_ = analysis
		self.model_ = None

		if (feature_interaction_order > 1):
			self.data_ = util.create_higher_order_feature_interactions(
				df=self.data_, feature_interactions=feature_interaction_order
			)

	def fit_ols(self):
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
			c = 0
			for model_size, result_list in results.items():
				c += sum(result_list) / len(result_list)
			total_results[param] = max(c / len(results), 0)
		'''
			print('PARAM={}'.format(param))
			for terms, formula in model_subsets[param]:

				# Create statsmodels ols
				model = smf.OLS(y_dm, X_dm).fit()

				print('TERMS: {}; RSQ: {}'.format(terms, model.rsquared_adj))


		for param_pair, paired_subsets in model_subsets.items():
			print('PARAM_PAIR={}'.format(param_pair))
			for (formula_1, terms_1), (formula_2, terms_2) in paired_subsets:
				y_dm_1, X_dm_1 = dmatrices(formula_1, self.data_[list(terms_1)], return_type='matrix')
				y_dm_2, X_dm_2 = dmatrices(formula_2, self.data_[list(terms_2)], return_type='matrix')

				# Create statsmodels ols
				model_1 = smf.OLS(y_dm_1, X_dm_1).fit()
				model_2 = smf.OLS(y_dm_2, X_dm_2).fit()

				if (model_1.rsquared_adj > model_2.rsquared_adj):
					results[param_pair[0]] += 1
				if (model_2.rsquared_adj > model_1.rsquared_adj):
					results[param_pair[1]] += 1

				print('MODEL 1[{}] rsq: {} vs. MODEL 2[{}] rsq: {}'.format(terms_1, model_1.rsquared_adj, terms_2, model_2.rsquared_adj))
			print('-------------')
			''
			''
			for formula, formula_terms in subsets:
				print('PARAM_PAIR={}; formula_terms={}'.format(param_pair, formula_terms))
				y_dm, X_dm = dmatrices(formula, self.data_[list(formula_terms)], return_type='matrix')

				# Create statsmodels ols
				model = smf.OLS(y_dm, X_dm).fit()

				results[param_pair][tuple(formula_terms)] = model.rsquared_adj
				print('FORMULA TERMS={}; RSQUARED_ADJ={}'.format(formula_terms, model.rsquared_adj))
				print('----')
			print('RESULTS: {}'.format(results)) # TODO: Create the ordering!!!
			'''

		'''
		If the models is created via a `ModelDesc` formula instead of a string, the design matrix dataframe
		for some reasons does not contain a design_info field. Hence, the quick and easy way out of this shit
		is to simply set an appropriate design_info a priori to calling the anova_lm function in order
		to avoid the whole shit to crash
		'''
		print(total_results)
		model.model.data.design_info = X_dm.design_info

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
		'''
		data_columns = self.data_.columns.values.tolist()

		all_pairs = [p for p in itertools.combinations(data_columns, 2) if self.label_name_ not in p]

		all_subsets = []
		for i in range(1, len(data_columns) + 1):
			all_subsets.extend(list(itertools.combinations(data_columns, i)))

		subsets = collections.defaultdict(list)
		for p in all_pairs:
			remaining_cols = [c for c in data_columns if c not in p and c != self.label_name_]
			current_subsets = []
			for i in range(len(remaining_cols)):
				current_subsets.extend(itertools.combinations(remaining_cols, i))
			for subset in current_subsets:
				rhs_terms_1 = subset + (p[0],)
				rhs_terms_2 = subset + (p[1],)

				formula_1 = ModelDesc(
					[Term([LookupFactor(self.label_name_)])],  # LHS
					[Term([LookupFactor(fn, force_categorical=True)]) for fn in rhs_terms_1]
				)

				formula_2 = ModelDesc(
					[Term([LookupFactor(self.label_name_)])],  # LHS
					[Term([LookupFactor(fn, force_categorical=True)]) for fn in rhs_terms_2]
				)

				subsets[p].append(((formula_1, rhs_terms_1 + (self.label_name_,)),
								   (formula_2, rhs_terms_2 + (self.label_name_,))))
			#''
		
			#''
			for subset in all_subsets:
				if (p[1] not in subset and p[0] in subset): # One has to be absent whereas the other one has to be present (is this consistent with the paper?)
					rhs_terms = [fn for fn in subset if fn != self.label_name_]

					formula = ModelDesc(
						[Term([LookupFactor(self.label_name_)])],  # LHS
						[Term([LookupFactor(fn, force_categorical=True)]) for fn in rhs_terms]
					)
					subsets[(p[0], p[1])].append((formula, rhs_terms + [self.label_name_]))

				if (p[0] not in subset and p[1] in subset):
					rhs_terms = [fn for fn in subset if fn != self.label_name_]

					formula = ModelDesc(
						[Term([LookupFactor(self.label_name_)])],  # LHS
						[Term([LookupFactor(fn, force_categorical=True)]) for fn in rhs_terms]
					)
					subsets[(p[1], p[0])].append((formula, rhs_terms + [self.label_name_]))
			'''
		return subsets




class _ParameterAnalyser():
	def __init__(self, data, feature_interaction_order=1, **kwargs):
		self.label_name_ = kwargs.pop('label_name', 'y')
		self.data_ = util.load_data(d=data, **kwargs)
		self.anova_table_raw_ = None
		self.anova_table_fmt_ = None
		self.anova_result_ = None
		self.model_ = None

		if (feature_interaction_order > 1):
			self.data_ = util.create_higher_order_feature_interactions(
				df=self.data_, feature_interactions=feature_interaction_order
			)

	def fit_ols(self, anova_scale=None, anova_type=2, anova_test='F', anova_robust=None):
		data_columns = self.data_.columns.values.tolist()

		# Create statsmodels formula with patsy --> can't use just a simple string becasue the patsy parser fails after
		# 485 features: https://github.com/pydata/patsy/issues/18
		# http://patsy.readthedocs.io/en/latest/formulas.html
		formula = ModelDesc(
			[Term([LookupFactor(self.label_name_)])],  # LHS
			[Term([LookupFactor(fn, force_categorical=True)]) for fn in data_columns if fn != self.label_name_]  # RHS
		)

		y_dm, X_dm = dmatrices(formula, self.data_, return_type='matrix')

		# Create statsmodels ols
		model = smf.OLS(y_dm, X_dm).fit()

		'''
		If the models is created via a `ModelDesc` formula instead of a string, the design matrix dataframe
		for some reasons does not contain a design_info field. Hence, the quick and easy way out of this shit
		is to simply set an appropriate design_info a priori to calling the anova_lm function in order
		to avoid the whole shit to crash
		'''
		model.model.data.design_info = X_dm.design_info

		# ANOVA ablation
		# ANOVA type explanation: https://mcfromnz.wordpress.com/2011/03/02/anova-type-iiiiii-ss-explained/
		anova_table = sm.stats.anova_lm(model, typ=anova_type, scale=anova_scale, test=anova_test, robust=anova_robust)
		anova_table['partial_sum_sq'] = (anova_table['sum_sq'] / anova_table['sum_sq'].sum())
		anova_table['pct_explained'] = (anova_table['sum_sq'] / anova_table['sum_sq'].sum()) * 100

		self.model_ = model
		self.anova_table_raw_ = anova_table

		# Copy table
		self.anova_table_fmt_ = self.anova_table_raw_.copy(deep=True)
		
		# Drop residual
		self.anova_table_fmt_.drop(self.anova_table_fmt_.tail(1).index, inplace=True)
		
		# Add name to parameter
		#self.anova_table_fmt_.rename(columns={'Unnamed: 0': 'parameter'}, inplace=True)
		#if (self.anova_table_fmt_.index.names[0] is None):
		#	self.anova_table_fmt_.index.names = ['parameter']
		
		# Sort by variance explained
		self.anova_table_fmt_.sort_values(by='pct_explained', ascending=False, inplace=True)

		self.anova_result_ = collections.namedtuple('AnovaResult', ['rsquared_adj', 'rsquared', 'mse'])(
			rsquared_adj=model.rsquared_adj, rsquared=model.rsquared, mse=model.mse_total
		)

		# TODO: Calculate pct_explained as a fraction of r2 and r2_adj, because r2 differs from the raw sum of squares (and therefore pct_explained)
		# http://blog.minitab.com/blog/statistics-and-quality-data-analysis/r-squared-sometimes-a-square-is-just-a-square
		# http://blog.minitab.com/blog/adventures-in-statistics-2/regression-analysis-how-do-i-interpret-r-squared-and-assess-the-goodness-of-fit
		# https://stats.stackexchange.com/questions/32596/what-is-the-difference-between-coefficient-of-determination-and-mean-squared
		# DONT FORGET THE RESIDUAL PLOT

		# Hmmmm....https://www.researchgate.net/post/How_can_I_determine_the_relative_contribution_of_predictors_in_multiple_regression_models
		# http://onlinestatbook.com/2/regression/multiple_regression.html
		# http://sphweb.bumc.bu.edu/otlt/MPH-Modules/BS/BS704_Multivariable/BS704_Multivariable_print.html
		# THIS IS TALKING ABOUT THE METHOD THAT LAPESA AND EVERT USED: https://github.com/statsmodels/statsmodels/issues/2154

		return self.anova_result_

	#def plot_
	# TODO: For the plotting, use bashplotlib when in shell mode and seaborn otherwise


if (__name__ == '__main__'):
	filename = '/Users/thomas/DevSandbox/InfiniteSandbox/tag-lab/paralysis/resources/example_data/snli_svm.json'
	pa = ParameterAnalyser(data=filename, label_name='result')
	pa.fit_ols()