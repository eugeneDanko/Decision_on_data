import numpy as np 
from interfaces import LossFunction, LossFunctionClosedFormMixin, LinearRegressionInterface, AbstractOptimizer
from descents import AnalyticSolutionOptimizer
from typing import Dict, Type, Optional, Callable
from abc import abstractmethod, ABC
from scipy.sparse.linalg import ArpackError, svds



class MSELoss(LossFunction, LossFunctionClosedFormMixin):

    def __init__(self, analytic_solution_func: Callable[[np.ndarray, np.ndarray], np.ndarray] = None):

        if analytic_solution_func is None:
            self.analytic_solution_func = self._plain_analytic_solution
        else:
            self.analytic_solution_func = analytic_solution_func

        

    def loss(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
        error = X @ w - y
        return np.mean(error ** 2)

    def gradient(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
        l = len(y)
        error = X @ w - y
        return (2 / l) * X.T @ error

    def analytic_solution(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Возвращает решение по явной формуле (closed-form solution)

        X: np.ndarray, матрица регрессоров 
        y: np.ndarray, вектор таргета

        returns: np.ndarray, оптимальный по MSE вектор весов, вычисленный при помощи аналитического решения для данных X, y
        """
        # Функция-диспатчер в одну из истинных функций для вычисления решения по явной формуле (closed-form)
        # Необходима в связи c наличием интерфейса analytic_solution у любого лосса; 
        # self-injection даёт возможность выбирать, какое именно closed-form решение использовать
        return self.analytic_solution_func(X, y)
        
    
    @classmethod
    def _plain_analytic_solution(cls, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.linalg.inv(X.T @ X) @ X.T @ y
    
    @classmethod
    def _svd_analytic_solution(cls, X: np.ndarray, y: np.ndarray, eps: float = 1e-10) -> np.ndarray:
        if min(X.shape) == 1:
            return np.linalg.lstsq(X, y, rcond=None)[0]

        try:
            u, s, vt = svds(X, k=min(X.shape) - 1, solver="arpack", tol=0)
        except ArpackError:
            return np.linalg.lstsq(X, y, rcond=None)[0]

        if X.shape[0] >= X.shape[1]:
            missing_v = np.linalg.svd(vt, full_matrices=True)[2][-1]
            missing_u = X @ missing_v
            missing_s = np.linalg.norm(missing_u)
            if missing_s > eps:
                u = np.column_stack((u, missing_u / missing_s))
                s = np.append(s, missing_s)
                vt = np.vstack((vt, missing_v))
        else:
            missing_u = np.linalg.svd(u.T, full_matrices=True)[2][-1]
            missing_v = X.T @ missing_u
            missing_s = np.linalg.norm(missing_v)
            if missing_s > eps:
                u = np.column_stack((u, missing_u))
                s = np.append(s, missing_s)
                vt = np.vstack((vt, missing_v / missing_s))

        nonzero = s > eps
        
        u = u[:, nonzero]
        s = s[nonzero]
        vt = vt[nonzero, :]
        
        return vt.T @ ((u.T @ y) / s)

class LogCoshLoss(LossFunction):

    def loss(
        self,
        X: np.ndarray,
        y: np.ndarray,
        w: np.ndarray
    ) -> float:

        error = X @ w - y

        # Численно устойчивая запись log(cosh(error))
        log_cosh = (
            np.logaddexp(error, -error)
            - np.log(2)
        )

        return np.mean(log_cosh)


    def gradient(
        self,
        X: np.ndarray,
        y: np.ndarray,
        w: np.ndarray
    ) -> np.ndarray:

        error = X @ w - y

        return (
            X.T @ np.tanh(error)
        ) / len(y)

class HuberLoss(LossFunction):

    def __init__(self, delta: float = 1.0):
        self.delta = delta


    def loss(
        self,
        X: np.ndarray,
        y: np.ndarray,
        w: np.ndarray
    ) -> float:

        error = X @ w - y
        absolute_error = np.abs(error)

        object_losses = np.where(
            absolute_error < self.delta,
            0.5 * error ** 2,
            (
                self.delta * absolute_error
                - 0.5 * self.delta ** 2
            )
        )

        return np.mean(object_losses)


    def gradient(
        self,
        X: np.ndarray,
        y: np.ndarray,
        w: np.ndarray
    ) -> np.ndarray:

        error = X @ w - y
        absolute_error = np.abs(error)

        derivative = np.where(
            absolute_error < self.delta,
            error,
            self.delta * np.sign(error)
        )

        return (
            X.T @ derivative
        ) / len(y)

class L2Regularization(LossFunction):

    def __init__(self, core_loss: LossFunction, mu_rate: float = 1.0, analytic_solution_func: Callable[[np.ndarray, np.ndarray], np.ndarray] = None):
        self.core_loss = core_loss
        self.mu_rate = mu_rate

    def loss(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:

        core_part = self.core_loss.loss(X, y, w)

        penalty_part = self.mu_rate / 2 * np.sum(w ** 2)

        return core_part + penalty_part
    

    def gradient(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:

        core_part = self.core_loss.gradient(X, y, w)

        penalty_part = self.mu_rate * w

        return core_part + penalty_part



class CustomLinearRegression(LinearRegressionInterface):
    def __init__(
        self,
        optimizer: AbstractOptimizer,
        # l2_coef: float = 0.0,
        loss_function: LossFunction = MSELoss()
    ):
        self.optimizer = optimizer
        self.optimizer.set_model(self)

        # self.l2_coef = l2_coef
        self.loss_function = loss_function
        self.loss_history = []
        self.w = None
        self.X_train = None
        self.y_train = None
        

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.w

    def compute_gradients(self, X_batch: np.ndarray | None = None, y_batch: np.ndarray | None = None) -> np.ndarray:
        if X_batch is None:
            X_batch = self.X_train
            y_batch = self.y_train

        return self.loss_function.gradient(X_batch, y_batch, self.w)


    def compute_loss(self, X_batch: np.ndarray | None = None, y_batch: np.ndarray | None = None) -> float:
        if X_batch is None:
            X_batch = self.X_train
            y_batch = self.y_train
    
        return self.loss_function.loss(X_batch, y_batch, self.w)


    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X_train = X
        self.y_train = y
        self.w = np.zeros(X.shape[1])

        self.optimizer.optimize()
