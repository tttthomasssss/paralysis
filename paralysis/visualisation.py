from matplotlib import pyplot as plt
import seaborn as sns


def create_plot(parameter_table, output_file=None, param_dict={}, **kwargs):
	sns.set(style=kwargs.pop('style', 'whitegrid'))
	f, ax = plt.subplots(figsize=kwargs.pop('figsize', (17.21, 11.86)), dpi=kwargs.pop('dpi', 600),
						 facecolor=kwargs.pop('facecolor', 'w'), edgecolor=kwargs.pop('edgecolor', 'k'))
	plt.hold(kwargs.pop('hold', True))
	plt.grid(kwargs.pop('grid', True))

	sns.pointplot(x='weight', y='parameter', data=parameter_table, join=False, palette='deep',
				  markers=['<', 'v', '>', '^'], scale=2.5, ci=None)
	for tick in ax.xaxis.get_major_ticks():
		tick.label.set_fontsize(32)

	for tock in ax.yaxis.get_major_ticks():
		tock.label.set_fontsize(32)
	#leg = plt.legend(bbox_to_anchor=(1., 0.37), fancybox=True, fontsize=32)
	plt.xlabel('Parameter Importance (%)', fontsize=42)
	plt.ylabel('Parameter', fontsize=42)

	if (output_file is not None):
		#plt.savefig(output_file, bbox_extra_artists=(leg,), bbox_inches='tight', ncol=3)
		plt.savefig(output_file)