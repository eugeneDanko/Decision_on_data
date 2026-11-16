import numpy as np
from abc import ABC, abstractmethod
from scipy.sparse import lil_matrix
from interfaces import LearningRateSchedule, AbstractOptimizer, LinearRegressionInterface


# ===== Learning Rate Schedules =====
class ConstantLR(LearningRateSchedule):
    def __init__(self, lr: float):
        self.lr = lr

    def get_lr(self, iteration: int) -> float:
        return self.lr


class TimeDecayLR(LearningRateSchedule):
    def __init__(self, lambda_: float = 1.0):
        self.s0 = 1
        self.p = 0.5
        self.lambda_ = lambda_

    def get_lr(self, iteration: int) -> float:
        return self.lambda_ * (self.s0 / (self.s0 + iteration)) ** self.p


# ===== Base Optimizer =====
class BaseDescent(AbstractOptimizer, ABC):
    """
    Оптимизатор, имплементирующий градиентный спуск.
    Ответственен только за имплементацию общего алгоритма спуска.
    Все его составные части (learning rate, loss function+regularization) находятся вне зоны ответственности этого класса (см. Single Responsibility Principle).
    """
    def __init__(self, 
                 lr_schedule: LearningRateSchedule = TimeDecayLR(), 
                 tolerance: float = 1e-6,
                 max_iter: int = 1000
                ):
        self.lr_schedule = lr_schedule
        self.tolerance = tolerance
        self.max_iter = max_iter

        self.iteration = 0
        self.model: LinearRegressionInterface = None

    @abstractmethod
    def _update_weights(self) -> np.ndarray:
        """
        Вычисляет обновление согласно конкретному алгоритму и обновляет веса модели, перезаписывая её атрибут.
        Не имеет прямого доступа к вычислению градиента в точке, для подсчета вызывает model.compute_gradients.

        returns: np.ndarray, w_{k+1} - w_k
        """
        pass

    def _step(self) -> np.ndarray:
        """
        Проводит один полный шаг интеративного алгоритма градиентного спуска

        returns: np.ndarray, w_{k+1} - w_k
        """
        delta = self._update_weights()
        self.iteration += 1
        return delta

    def optimize(self) -> None:
        self.iteration = 0
        loss_history = []
    
        while self.iteration < self.max_iter:
            loss_history.append(self.model.compute_loss())
    
            delta = self._step()
    
            delta_norm_squared = np.sum(delta ** 2)
    
            if delta_norm_squared < self.tolerance or np.isnan(delta).any():
                break

        loss_history.append(self.model.compute_loss())
        self.model.loss_history = loss_history

# ===== Specific Optimizers =====
class VanillaGradientDescent(BaseDescent):
    def _update_weights(self) -> np.ndarray:
        gradient = self.model.compute_gradients()
        learning_rate = self.lr_schedule.get_lr(self.iteration)

        delta = -learning_rate * gradient
        self.model.w += delta

        return delta


class StochasticGradientDescent(BaseDescent):
    def __init__(self, *args, batch_size=32, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size

    def _update_weights(self) -> np.ndarray:
        X_train = self.model.X_train
        y_train = self.model.y_train
    
        num_objects = X_train.shape[0]
    
        batch_indx = np.random.randint(0, num_objects, size=self.batch_size)
    
        X_batch = X_train[batch_indx]
        y_batch = y_train[batch_indx]
    
        gradient = self.model.compute_gradients(X_batch, y_batch)
        learning_rate = self.lr_schedule.get_lr(self.iteration)
    
        delta = -learning_rate * gradient
        self.model.w += delta
    
        return delta


class SAGDescent(BaseDescent):
    def __init__(self, *args, batch_size=32, **kwargs):
        super().__init__(*args, **kwargs)
        self.grad_memory = None
        self.avg_grad = None
        self.batch_size = batch_size

    def _update_weights(self) -> np.ndarray:
        X_train = self.model.X_train
        y_train = self.model.y_train
        num_objects, num_features = X_train.shape
    
        if self.grad_memory is None:
            self.grad_memory = lil_matrix((num_objects, num_features),dtype=np.float32)
            self.avg_grad = np.zeros(num_features)
    
        batch_indx = np.random.choice(num_objects, size=self.batch_size, replace=False)
    
        new_gradients = np.array([self.model.compute_gradients(X_train[i:i + 1], y_train[i:i + 1]) for i in batch_indx])
    
        old_gradients = self.grad_memory[batch_indx].toarray()
        self.grad_memory[batch_indx] = new_gradients
    
        self.avg_grad += np.sum(new_gradients - old_gradients, axis=0) / num_objects
    
        learning_rate = self.lr_schedule.get_lr(self.iteration)
        delta = -learning_rate * self.avg_grad
    
        self.model.w += delta
        return delta


class MomentumDescent(BaseDescent):
    def __init__(self,  *args, beta=0.9, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta
        self.velocity = None

    def _update_weights(self) -> np.ndarray:
        if self.velocity is None:
            self.velocity = np.zeros_like(self.model.w)

        gradient = self.model.compute_gradients()
        learning_rate = self.lr_schedule.get_lr(self.iteration)
    
        self.velocity = self.beta * self.velocity + learning_rate * gradient
    
        delta = -self.velocity
        self.model.w += delta
    
        return delta


class Adam(BaseDescent):
    def __init__(self, *args, beta1=0.9, beta2=0.999, eps=1e-8, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None
        self.v = None

    def _update_weights(self) -> np.ndarray:
        if self.m is None:
            self.m = np.zeros_like(self.model.w)
            self.v = np.zeros_like(self.model.w)
    
        gradient = self.model.compute_gradients()
    
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
    
        self.v = self.beta2 * self.v + (1 - self.beta2) * gradient ** 2
    
        step_number = self.iteration + 1
    
        m_hat = self.m / (1 - self.beta1 ** step_number)
        v_hat = self.v / (1 - self.beta2 ** step_number)
    
        learning_rate = self.lr_schedule.get_lr(self.iteration)
    
        delta = -learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)
    
        self.model.w += delta
        return delta


# ===== Non-iterative Algorithms ====
class AnalyticSolutionOptimizer(AbstractOptimizer):
    """
    Универсальный дамми-класс для вызова аналитических решений 
    """
    def __init__(self):
        self.model = None
    

    def optimize(self) -> None:
        self.model.w = self.model.loss_function.analytic_solution(self.model.X_train, self.model.y_train)