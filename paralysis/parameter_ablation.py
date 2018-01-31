from patsy import dmatrices
from patsy import LookupFactor
from patsy import ModelDesc
from patsy import Term
from statsmodels import api as sm
from statsmodels.formula import api as smf

from paralysis import util


class ParameterAnalyser():
	def __init__(self, d, **kwargs):
		self.label_name_ = kwargs.pop('label_name', 'y')
		self.data_ = util.load_data(d=d, **kwargs)

		feat_interactions = kwargs.pop('feature_interaction_order', 1)
		if (feat_interactions > 1):
			self.data_ = util.create_higher_order_feature_interactions(df=self.data_, feature_interactions=feat_interactions)

	def fit_ols(self, anova_type=2):
		data_columns = self.data_.columns.values.tolist()

		# Create statsmodels formula with patsy --> can't use just a simple string becasue the patsy parser fails after 485 features: https://github.com/pydata/patsy/issues/18
		# http://patsy.readthedocs.io/en/latest/formulas.html
		formula = ModelDesc([Term([LookupFactor(self.label_name_)])],  # LHS
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
		anova_table = sm.stats.anova_lm(model, typ=anova_type)  # ANOVA type explanation: https://mcfromnz.wordpress.com/2011/03/02/anova-type-iiiiii-ss-explained/
		anova_table['pct_explained'] = (anova_table['sum_sq'] / anova_table['sum_sq'].sum()) * 100

		result = {
			'explained_variance': model.rsquared_adj,
			'mse': model.mse_total
		}