import collections

from patsy import dmatrices
from patsy import LookupFactor
from patsy import ModelDesc
from patsy import Term
from statsmodels import api as sm
from statsmodels.formula import api as smf

from paralysis import util


class ParameterAnalyser():
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
		anova_table['pct_explained'] = (anova_table['sum_sq'] / anova_table['sum_sq'].sum()) * 100

		self.model_ = model
		self.anova_table_raw_ = anova_table

		# TODO: Format the Anova table
		'''
		Copy table
		self.anova_table_fmt_ = self.anova_table_raw_.copy(deep=True)
		
		# Drop residual
		self.anova_table_fmt_.drop(self.anova_table_fmt_.tail(1).index, inplace=True)
		
		# Add name to parameter
		self.anova_table_fmt_.rename(columns={'Unnamed: 0': 'parameter'}, inplace=True)
		
		# Sort by variance explained
		self.anova_table_fmt_.sort_values(by='pct_explained', ascending=False, inplace=True)
		'''

		self.anova_result_ = collections.namedtuple('AnovaResult', ['explained_variance', 'mse'])(
			explained_variance=model.rsquared_adj, mse=model.mse_total
		)

		# Oddly enough, rsquared differs from pct_explained (sum of squares), these blogposts could shed some light on it:
		# http://blog.minitab.com/blog/statistics-and-quality-data-analysis/r-squared-sometimes-a-square-is-just-a-square
		# http://blog.minitab.com/blog/adventures-in-statistics-2/regression-analysis-how-do-i-interpret-r-squared-and-assess-the-goodness-of-fit
		# https://stats.stackexchange.com/questions/32596/what-is-the-difference-between-coefficient-of-determination-and-mean-squared

		return self.anova_result_

	#def plot_
	# TODO: For the plotting, use bashplotlib when in shell mode and seaborn otherwise