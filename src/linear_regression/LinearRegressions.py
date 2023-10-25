import statsmodels.api as sm
import pandas as pd
class LinearRegressionSM:
    def __init__(self, left_hand_side, right_hand_side):
        self.left_hand_side = left_hand_side
        self.right_hand_side=right_hand_side
        self._model = None

    def fit(self):
        X = sm.add_constant(self.right_hand_side)
        model = sm.OLS(self.left_hand_side,X).fit()
        self._model = model

    def get_params(self):
        return pd.Series(self._model.params, name='Beta coefficients')

    def get_pvalues(self):
        return pd.Series(self._model.pvalues, name='P-values for the corresponding coefficients')

    def get_wald_test_result(self, restriction_matrix):
        wald_test_result = self._model.wald_test(restriction_matrix)
        fvalue = '{:.2f}'.format(round(float(wald_test_result.statistic), 2))
        pvalue = '{:.3f}'.format(round(float(wald_test_result.pvalue), 3))
        return f'F-value: {fvalue}, p-value: {pvalue}'

    def get_model_goodness_values(self):
        ars = '{:.3f}'.format(round(self._model.rsquared_adj, 3))
        ak = '{:.3f}'.format(round(self._model.aic, 3))
        by = '{:.3f}'.format(round(self._model.bic, 3))
        return f'Adjusted R-squared: {ars}, Akaike IC: {ak}, Bayes IC: {by}'