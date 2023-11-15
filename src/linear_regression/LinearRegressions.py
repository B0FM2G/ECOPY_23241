import statsmodels.api as sm
import pandas as pd
import numpy as np
from scipy.stats import t
from scipy.stats import f

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
        ak = '{:.2e}'.format(round(self._model.aic, 2))
        by = '{:.2e}'.format(round(self._model.bic, 2))
        return f'Adjusted R-squared: {ars}, Akaike IC: {ak}, Bayes IC: {by}'

class LinearRegressionNP:
    def __init__(self, left_hand_side, right_hand_side):
        self.left_hand_side = left_hand_side
        self.right_hand_side = right_hand_side
        self._model = None

    def fit(self):
        X = np.column_stack((np.ones(len(self.right_hand_side)), self.right_hand_side))
        y = self.left_hand_side
        beta = np.linalg.inv(X.T @ X) @ X.T @ y
        self.beta = beta

    def get_params(self):
        return pd.Series(self.beta, name='Beta coefficients')

    def get_pvalues(self):
        X = np.column_stack((np.ones(len(self.right_hand_side)), self.right_hand_side))
        y = self.left_hand_side
        n, k = X.shape
        beta = self.beta
        H = X @ np.linalg.inv(X.T @ X) @ X.T
        residuals = y - X @ beta
        residual_variance = (residuals @ residuals) / (n - k)
        standard_errors = np.sqrt(np.diagonal(residual_variance * np.linalg.inv(X.T @ X)))
        t_statistics = beta / standard_errors
        df = n - k
        p_values = [2 * (1 - t.cdf(abs(t_stat), df)) for t_stat in t_statistics]
        p_values = pd.Series(p_values, name="P-values for the corresponding coefficients")
        return p_values

    def get_wald_test_result(self, R):
        X = np.column_stack((np.ones(len(self.right_hand_side)), self.right_hand_side))
        y = self.left_hand_side
        beta = self.beta
        residuals = y - X @ beta
        r_matrix = np.array(R)
        r = r_matrix @ beta
        n = len(self.left_hand_side)
        m, k = r_matrix.shape
        sigma_squared = np.sum(residuals ** 2) / (n - k)
        H = r_matrix @ np.linalg.inv(X.T @ X) @ r_matrix.T
        wald = (r.T @ np.linalg.inv(H) @ r) / (m * sigma_squared)
        p_value = 1 - f.cdf(wald, dfn=m, dfd=n - k)
        return f'Wald: {wald:.3f}, p-value: {p_value:.3f}'

    def get_model_goodness_values(self):
        X = np.column_stack((np.ones(len(self.right_hand_side)), self.right_hand_side))
        y = self.left_hand_side
        n, k = X.shape
        beta = self.beta
        y_pred = X @ beta
        ssr = np.sum((y_pred - np.mean(y)) ** 2)
        sst = np.sum((y - np.mean(y)) ** 2)
        centered_r_squared = ssr / sst
        adjusted_r_squared = 1 - (1 - centered_r_squared) * (n - 1) / (n - k)
        result = f"Centered R-squared: {centered_r_squared:.3f}, Adjusted R-squared: {adjusted_r_squared:.3f}"
        return result

import numpy as np
import pandas as pd
from scipy.stats import t, f

class LinearRegressionGLS:
    def __init__(self, left_hand_side, right_hand_side):
        self.left_hand_side = left_hand_side.values
        self.right_hand_side = right_hand_side.values

    def fit(self):

        X_ols = np.column_stack((np.ones(len(self.right_hand_side)), self.right_hand_side))
        beta_ols = np.linalg.inv(X_ols.T @ X_ols) @ X_ols.T @ self.left_hand_side
        residuals = self.left_hand_side - X_ols @ beta_ols

        X_omega = np.column_stack(
                (np.ones(len(self.right_hand_side)), self.right_hand_side))
        y_omega = np.log(residuals ** 2)
        beta_omega = np.linalg.inv(X_omega.T @ X_omega) @ X_omega.T @ y_omega

        V_inv_diag = 1 / np.sqrt(np.exp(X_omega @ beta_omega))
        V_inv = np.diag(V_inv_diag)

        beta_gls = np.linalg.inv(X_ols.T @ V_inv @ X_ols) @ X_ols.T @ V_inv @ self.left_hand_side

        self.beta_params = beta_gls

    def get_params(self):
        return pd.Series(self.beta_params, name='Beta coefficients')

    def get_pvalues(self):
        X_ols = np.column_stack((np.ones(len(self.right_hand_side)), self.right_hand_side))
        beta_ols = np.linalg.inv(X_ols.T @ X_ols) @ X_ols.T @ self.left_hand_side
        residuals = self.left_hand_side - X_ols @ beta_ols

        X_omega = np.column_stack(
            (np.ones(len(self.right_hand_side)), self.right_hand_side))
        y_omega = np.log(residuals ** 2)
        beta_omega = np.linalg.inv(X_omega.T @ X_omega) @ X_omega.T @ y_omega

        V_inv_diag = 1 / np.sqrt(np.exp(X_omega @ beta_omega))
        V_inv = np.diag(V_inv_diag)

        beta_gls = np.linalg.inv(X_ols.T @ V_inv @ X_ols) @ X_ols.T @ V_inv @ self.left_hand_side
        X_gls = np.linalg.inv(np.sqrt(V_inv)) @ X_ols

        dof = len(self.left_hand_side) - X_gls.shape[1]
        t_stat = self.beta_params / np.sqrt(np.diag(np.linalg.inv(X_gls.T @ X_gls)))
        p_values = pd.Series([min(value, 1 - value) * 2 for value in t.cdf(-np.abs(t_stat), df=dof)],
                             name='P-values for the corresponding coefficients')
        return p_values

    def get_wald_test_result(self, R):
        X_ols = np.column_stack((np.ones(len(self.right_hand_side)), self.right_hand_side))
        beta_ols = np.linalg.inv(X_ols.T @ X_ols) @ X_ols.T @ self.left_hand_side
        residuals = self.left_hand_side - X_ols @ beta_ols

        X_omega = np.column_stack(
            (np.ones(len(self.right_hand_side)), self.right_hand_side))
        y_omega = np.log(residuals ** 2)
        beta_omega = np.linalg.inv(X_omega.T @ X_omega) @ X_omega.T @ y_omega

        V_inv_diag = 1 / np.sqrt(np.exp(X_omega @ beta_omega))
        V_inv = np.diag(V_inv_diag)

        beta_gls = np.linalg.inv(X_ols.T @ V_inv @ X_ols) @ X_ols.T @ V_inv @ self.left_hand_side
        residuals_gls = y_gls - X_gls @ beta_gls
        r_matrix = np.array(R)
        r = r_matrix @ beta_gls
        n = len(self.left_hand_side)
        m, k = r_matrix.shape
        sigma_squared = np.sum(residuals_gls ** 2) / (n - k)
        H = r_matrix @ np.linalg.inv(X_ols.T @ X_ols) @ r_matrix.T
        wald = (r.T @ np.linalg.inv(H) @ r) / (m * sigma_squared)
        p_value = 1 - f.cdf(wald, dfn=m, dfd=n - k)
        return f'Wald: {wald:.3f}, p-value: {p_value:.3f}'

    def get_model_goodness_values(self):
        X_ols = np.column_stack((np.ones(len(self.right_hand_side)), self.right_hand_side))
        beta_ols = np.linalg.inv(X_ols.T @ X_ols) @ X_ols.T @ self.left_hand_side
        residuals = self.left_hand_side - X_ols @ beta_ols

        X_omega = np.column_stack(
            (np.ones(len(self.right_hand_side)), self.right_hand_side))
        y_omega = np.log(residuals ** 2)
        beta_omega = np.linalg.inv(X_omega.T @ X_omega) @ X_omega.T @ y_omega

        V_inv_diag = 1 / np.sqrt(np.exp(X_omega @ beta_omega))
        V_inv = np.diag(V_inv_diag)

        beta_gls = np.linalg.inv(X_ols.T @ V_inv @ X_ols) @ X_ols.T @ V_inv @ self.left_hand_side
        y_mean = np.mean(self.left_hand_side)
        y_pred = X_ols @ self.beta_params
        total_sum_of_squares = np.sum((self.left_hand_side - y_mean) ** 2)
        residual_sum_of_squares = np.sum((self.left_hand_side - y_pred) ** 2)
        centered_r_squared = 1 - (residual_sum_of_squares / total_sum_of_squares)
        adjusted_r_squared = 1 - (residual_sum_of_squares / (len(self.left_hand_side) - X_ols.shape[1])) * (
                    len(self.left_hand_side) - 1) / total_sum_of_squares
        return f"Centered R-squared: {centered_r_squared:.3f}, Adjusted R-squared: {adjusted_r_squared:.3f}"
