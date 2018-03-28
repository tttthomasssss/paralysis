# Paralysis

### WARNING - CURRENTLY IN A BAD STATE AND UNDERGOING MAJOR REHAUL (DO NOT CLONE!)

The performance of a machine learning model is - to a substantial extent - dependent on the choice of hyperparameters for a particular model. The influence of different hyperparameter settings on model performance is often even larger than the actual choice of model itself.

Moreover, intuitions about what a "good" default parameterisation is rarely transfers from one type of model to the next - or not even from one dataset to the next when using the same model.

Paralysis is a tool that quantifies the impact of model hyperparameters by using Linear Regression. Given a set of experiments, the model performance (e.g. Accuracy, F1-score, etc.) is treated as the dependent variable and the model hyperparameters are treated as predictors. Paralysis analysis how much of the variance can be explained by a given hyperparameterisation and estimates how much variance each individual predictor contributes.
