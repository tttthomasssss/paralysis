from patsy import LookupFactor
from patsy import ModelDesc
from patsy import Term
from statsmodels import api as sm
from statsmodels.formula import api as smf


def why_not_to_use_anova_demo(data, label_name, anova_scale=None, anova_type=2, anova_test='F', anova_robust=None,
							  data_columns=None):
	'''
	ANOVA is sensitive to the ordering of the input variables, thus changing the order creates wildly different results

	DO NOT USE ANOVA FOR ESTIMATING THE EXPLAINED VARIANCE OF EACH PREDICTOR

	:param data:
	:param label_name:
	:param anova_scale:
	:param anova_type:
	:param anova_test:
	:param anova_robust:
	:return:
	'''
	if data_columns is None:
		data_columns = data.columns.values.tolist()

	# Create statsmodels formula with patsy --> can't use just a simple string becasue the patsy parser fails after
	# 485 features: https://github.com/pydata/patsy/issues/18
	# http://patsy.readthedocs.io/en/latest/formulas.html
	formula = ModelDesc(
		[Term([LookupFactor(label_name)])],  # LHS
		[Term([LookupFactor(fn, force_categorical=True)]) for fn in data_columns if fn != label_name]  # RHS
	)

	# Create statsmodels ols
	model = smf.ols(formula, data).fit()

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

	return anova_table