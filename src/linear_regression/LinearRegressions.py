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

    def get_paired_se_and_percentile_ci(self, number_of_bootstrap_samples, alpha, random_seed):
        np.random.seed(random_seed)
        X = np.column_stack((np.ones(len(self.right_hand_side)), self.right_hand_side))
        y = self.left_hand_side
        n, k = X.shape
        beta = self.beta

        bootstrap_betas = []
        for _ in range(number_of_bootstrap_samples):
            indices = np.random.choice(n, size=n, replace=True)
            bootstrap_X = X[indices, :]
            bootstrap_y = y[indices]
            bootstrap_beta = np.linalg.inv(bootstrap_X.T @ bootstrap_X) @ bootstrap_X.T @ bootstrap_y
            bootstrap_betas.append(bootstrap_beta[1])

        bse = np.std(bootstrap_betas, ddof=1)

        lb = np.percentile(bootstrap_betas, 100 * alpha / 2)
        ub = np.percentile(bootstrap_betas, 100 * (1 - alpha / 2))

        return f'Paired Bootstraped SE: {bse:.3f}, CI: [{lb:.3f}, {ub:.3f}]'

    def get_wild_se_and_normal_ci(self, number_of_bootstrap_samples, alpha, random_seed):
        np.random.seed(random_seed)
        X = np.column_stack((np.ones(len(self.right_hand_side)), self.right_hand_side))
        y = self.left_hand_side
        n, k = X.shape
        beta = self.beta

        bootstrap_betas = []
        for _ in range(number_of_bootstrap_samples):
            indices = np.random.choice(n, size=n, replace=True)
            bootstrap_X = X[indices, :]
            bootstrap_y = y[indices]
            bootstrap_residuals = (bootstrap_y - bootstrap_X @ beta)* np.random.normal(0, 1, size=n)
            bootstrap_y = X @ beta + bootstrap_residuals
            bootstrap_beta = np.linalg.inv(X.T @ X) @ X.T @ bootstrap_y
            bootstrap_betas.append(bootstrap_beta[1])

        bse = np.std(bootstrap_betas, ddof=1)

        lb = np.percentile(bootstrap_betas, 100 * alpha / 2)
        ub = np.percentile(bootstrap_betas, 100 * (1 - alpha / 2))

        return f'Wild Bootstraped SE: {bse:.3f}, CI: [{lb:.3f}, {ub:.3f}]'





import numpy as np
import pandas as pd
from scipy.stats import t, f

class LinearRegressionGLS:
    def __init__(self, left_hand_side, right_hand_side):
        self.left_hand_side = left_hand_side.values
        self.right_hand_side = right_hand_side.values

    def fit(self):
        self.X = np.column_stack((np.ones(len(self.right_hand_side)), self.right_hand_side))
        self.Y = self.left_hand_side
        beta_ols = np.linalg.inv(self.X.T @ self.X) @ self.X.T @self.Y
        resid_ols = self.Y - self.X @ beta_ols
        log_resid_ols = np.log(resid_ols ** 2)
        beta_omega = np.linalg.inv(self.X.T @ self.X) @ self.X.T @ log_resid_ols
        V_inv_diag = 1 / np.sqrt(np.exp(self.X @ beta_omega))
        self.V_inv = np.diag(V_inv_diag)
        beta_gls = np.linalg.inv(self.X.T @ self.V_inv @ self.X) @ self.X.T @ self.V_inv @ self.Y
        self.beta_params = beta_gls

    def get_params(self):
        return pd.Series(self.beta_params, name='Beta coefficients')

    def get_pvalues(self):
        self.fit()
        dof = len(self.Y) - self.X.shape[1]
        residuals = self.Y - self.X @ self.beta_params
        residual_variance = (residuals @ residuals) / dof
        t_stat = self.beta_params / np.sqrt(np.diag(residual_variance*np.linalg.inv(self.X.T @ self.V_inv @ self.X)))
        p_values = pd.Series([min(value, 1 - value) * 2 for value in t.cdf(-np.abs(t_stat), df=dof)],
                             name='P-values for the corresponding coefficients')
        return p_values

    def get_wald_test_result(self, R):
        self.fit()
        r_matrix = np.array(R)
        r = r_matrix @ self.beta_params
        n = len(self.Y)
        m, k = r_matrix.shape
        residuals = self.Y - self.X @ self.beta_params
        residual_variance = (residuals @ residuals)/ (n - k)
        H = r_matrix @ np.linalg.inv(self.X.T @ self.V_inv @ self.X) @ r_matrix.T
        wald_value = (r.T @ np.linalg.inv(H) @ r) / (m * residual_variance)
        p_value = 1 - f.cdf(wald_value, dfn=m, dfd=n - k)
        return f'Wald: {wald_value:.3f}, p-value: {p_value:.3f}'

    def get_model_goodness_values(self):
        self.fit()
        total_sum_of_squares = self.Y.T @ self.V_inv @ self.Y
        residual_sum_of_squares = self.Y.T @ self.V_inv @ self.X @ np.linalg.inv(self.X.T @ self.V_inv @ self.X) @ self.X.T @ self.V_inv @ self.Y
        crs = 1 - (residual_sum_of_squares / total_sum_of_squares)
        ars = 1 - (residual_sum_of_squares / (len(self.Y) - self.X.shape[1])) * (
                len(self.Y) - 1) / total_sum_of_squares
        return f"Centered R-squared: {crs:.3f}, Adjusted R-squared: {ars:.3f}"


import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

class LinearRegressionML:
    def __init__(self, left_hand_side, right_hand_side):
        self.left_hand_side = left_hand_side
        self.right_hand_side = right_hand_side

    def neg_loglikelihood(self, params):
        self.X = np.column_stack((np.ones(len(self.right_hand_side)), self.right_hand_side))
        self.Y = self.left_hand_side
        beta0, beta1, beta2, beta3, sig = params
        betak = np.array([beta0, beta1, beta2, beta3])
        pred = self.X @ betak
        LL = np.sum(norm.logpdf(self.Y, pred, sig))
        return -LL

    def fit(self):
        result = minimize(self.neg_loglikelihood, np.array([0.1, 0.1, 0.1, 0.1, 0.1]), method='L-BFGS-B')
        beta0, beta1, beta2, beta3, sig = result.x
        self.beta = np.array([beta0, beta1, beta2, beta3])

    def get_params(self):
        return pd.Series(self.beta, name='Beta coefficients')

    def get_pvalues(self):
        self.fit()
        n, k = self.X.shape
        self.residuals = self.Y - self.X @ self.beta
        sigma_sq = np.sum(self.residuals ** 2) / (n-k)
        variance_beta = np.linalg.inv(self.X.T @ self.X) * sigma_sq
        t_statistic = self.beta / np.sqrt(np.diag(variance_beta))
        p_values = 2 * (1 - t.cdf(np.abs(t_statistic), df=n-k))
        return pd.Series(p_values, name='P-values for the corresponding coefficients')

    def get_model_goodness_values(self):
        self.fit()
        n, k = self.X.shape
        beta = self.beta
        y_pred = self.X @ beta
        ssr = np.sum((y_pred - np.mean(self.Y)) ** 2)
        sst = np.sum((self.Y - np.mean(self.Y)) ** 2)
        crs = ssr / sst
        ars = 1 - (1 - crs) * (n - 1) / (n - k)
        return f"Centered R-squared: {crs:.3f}, Adjusted R-squared: {ars:.3f}"
