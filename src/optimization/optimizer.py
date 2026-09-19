from typing import Optional, Dict
import cvxpy as cp
import numpy as np
import pandas as pd

from src.utils.config import Config
from src.utils.logger import get_logger

logger = get_logger("PortfolioOptimizer")

class PortfolioOptimizer:
    def __init__(
        self,
        risk_aversion: float = 5.0,
        max_weight: float = 0.10,
        fee_rate: float = 0.002,
        beta_tol: float = 0.02
    ):
        self.risk_aversion = risk_aversion if risk_aversion else Config.RISK_AVERSION_LAMBDA
        self.max_weight = max_weight if max_weight else Config.MAX_WEIGHT_PER_ASSET
        self.fee_rate = fee_rate if fee_rate else Config.TRANSACTION_FEE
        self.beta_tol = beta_tol if beta_tol else Config.BETA_TOLERANCE

    def optimize(self, mu_series: pd.Series, cov_mat: pd.DataFrame, betas: pd.Series) -> pd.Series:
        symbols = mu_series.index.tolist()
        mu = mu_series.values
        Sigma = cov_mat.loc[symbols, symbols].values
        beta_vec = betas.loc[symbols].values

        weights = self._solve_cvxpy(mu, Sigma, beta_vec)
        if weights is None:
            logger.warning("CVXPY optimization infeasible or failed. Using heuristic fallback.")
            weights = self._heuristic_fallback(mu_series, betas.loc[symbols])

        w_series = pd.Series(weights, index=symbols)
        # Clean up very small numerical weights
        w_series[w_series.abs() < 1e-5] = 0.0
        return w_series

    def _solve_cvxpy(self, mu: np.ndarray, Sigma: np.ndarray, beta_vec: np.ndarray) -> Optional[np.ndarray]:
        n = len(mu)
        w = cp.Variable(n)
        
        # Quadratic risk and linear return objective
        risk = cp.quad_form(w, cp.psd_wrap(Sigma))
        ret = mu @ w
        objective = cp.Maximize(ret - (self.risk_aversion / 2.0) * risk)

        constraints = [
            cp.norm(w, 1) <= 1.0001,             
            cp.sum(w) == 0.0,                    
            cp.abs(beta_vec @ w) <= self.beta_tol, 
            w <= self.max_weight,                
            w >= -self.max_weight                
        ]

        prob = cp.Problem(objective, constraints)
        try:
            prob.solve(solver=cp.SCS, verbose=False)
            if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE] and w.value is not None:
                return w.value
            
            # Try ECO solver as secondary
            prob.solve(solver=cp.ECOS, verbose=False)
            if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE] and w.value is not None:
                return w.value
        except Exception as e:
            logger.debug(f"Solver exception: {e}")
        return None

    def _heuristic_fallback(self, mu_series: pd.Series, betas: pd.Series) -> np.ndarray:
        n = len(mu_series)
        ranks = mu_series.rank()
        w = np.where(ranks > n / 2, 1.0, -1.0)
        w = w / np.sum(np.abs(w))
        return w
