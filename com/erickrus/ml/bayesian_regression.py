import numpy as np
import itertools
import functools
from scipy.stats import multivariate_normal



class PolynomialFeature(object):
    """
    polynomial features
    transforms input array with polynomial features
    Example
    =======
    x =
    [[a, b],
    [c, d]]
    y = PolynomialFeatures(degree=2).transform(x)
    y =
    [[1, a, b, a^2, a * b, b^2],
    [1, c, d, c^2, c * d, d^2]]
    """

    def __init__(self, degree=2):
        """
        construct polynomial features
        Parameters
        ----------
        degree : int
            degree of polynomial
        """
        assert isinstance(degree, int)
        self.degree = degree

    def transform(self, x):
        """
        transforms input array with polynomial features
        Parameters
        ----------
        x : (sample_size, n) ndarray
            input array
        Returns
        -------
        output : (sample_size, 1 + nC1 + ... + nCd) ndarray
            polynomial features
        """
        if x.ndim == 1:
            x = x[:, None]
        x_t = x.transpose()
        features = [np.ones(len(x))]
        for degree in range(1, self.degree + 1):
            for items in itertools.combinations_with_replacement(x_t, degree):
                features.append(functools.reduce(lambda x, y: x * y, items))
        return np.asarray(features).transpose()

class BayesianRegression:
    """
    Bayesian regression model

    w ~ N(w|0, alpha^(-1)I)
    y = X @ w
    t ~ N(t|X @ w, beta^(-1))
    """
    # 此处 @ 操作符 X @ w 相当于np.matmul(X, w), 1D时是向量乘法

    def __init__(self, alpha:float=1., beta:float=1.):
        # 只在初始化 precision_prev 时使用alhpa
        self.alpha = alpha
        self.beta = beta
        self.w_mean = None
        self.w_precision = None

    def _is_prior_defined(self) -> bool:
        return self.w_mean is not None and self.w_precision is not None

    def _get_prior(self, ndim:int) -> tuple:
        if self._is_prior_defined():
            return self.w_mean, self.w_precision
        else:
            return np.zeros(ndim), self.alpha * np.eye(ndim)

    def fit(self, X:np.ndarray, t:np.ndarray):
        """
        bayesian update of parameters given training dataset

        Parameters
        ----------
        X : (N, n_features) np.ndarray
            training data independent variable
        t : (N,) np.ndarray
            training data dependent variable
        """


        # w ~ N(0, alpha^(-1)I)
        # y = np.matmul(X, w)
        # t ~ N(y, beta^(-1))
        mean_prev, precision_prev = self._get_prior(np.size(X, 1))

        # 此处增量为: beta * np.matmul(X.T, X)
        w_precision = precision_prev + self.beta * X.T @ X

        # 为方程: np.matmul(w_precision, w_mean) = np.matmul(precision_prev, mean_prev) + beta * np.matmul(X.T, t)
        # 求解 => w_mean
        # 此处增量为: beta * np.matmul(X.T, t)
        w_mean = np.linalg.solve(
            w_precision,
            precision_prev @ mean_prev + self.beta * X.T @ t
        )
        self.w_mean = w_mean
        self.w_precision = w_precision

        # w_cov = w_precision^(-1)
        self.w_cov = np.linalg.inv(self.w_precision)

    def predict(self, X:np.ndarray, return_std:bool=False, sample_size:int=None):
        """
        return mean (and standard deviation) of predictive distribution

        Parameters
        ----------
        X : (N, n_features) np.ndarray
            independent variable
        return_std : bool, optional
            flag to return standard deviation (the default is False)
        sample_size : int, optional
            number of samples to draw from the predictive distribution
            (the default is None, no sampling from the distribution)

        Returns
        -------
        y : (N,) np.ndarray
            mean of the predictive distribution
        y_std : (N,) np.ndarray
            standard deviation of the predictive distribution
        y_sample : (N, sample_size) np.ndarray
            samples from the predictive distribution
        """

        # 从N(w_mean, w_cov)分布中采样sample_size次, 得到w_sample
        # 根据np.matmul(X, w_sample)计算出y_sample 并返回
        if sample_size is not None:
            w_sample = np.random.multivariate_normal(
                self.w_mean, self.w_cov, size=sample_size
            )
            y_sample = X @ w_sample.T
            return y_sample

        # 此处可得出y的标准差y_std, 用来作为置信度参数
        y = X @ self.w_mean
        if return_std:
            y_var = 1 / self.beta + np.sum(X @ self.w_cov * X, axis=1)
            y_std = np.sqrt(y_var)
            return y, y_std
        return y

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    #%matplotlib inline
    # from com.erickrus.ml.bayesian_regression import BayesianRegression

    # 构造一个玩具数据集, 事实上贝叶斯线性回归也只需要很少的数据即可
    def create_toy_data(func, sample_size, std, domain=[0, 1]):
        x = np.linspace(domain[0], domain[1], sample_size)
        np.random.shuffle(x)
        t = func(x) + np.random.normal(scale=std, size=x.shape)
        return x, t

    # 构造一个简单线性函数
    def linear(x):
        return -0.3 + 0.5 * x

    def main():
        x_train, y_train = create_toy_data(linear, 20, 0.1, [-1, 1])
        x = np.linspace(-1, 1, 100)

        # 构造一个 二维空间mesh 从 w0:(-1,1), w1:(-1,1) 两个不同维度组合的空间点集
        w0, w1 = np.meshgrid(
            np.linspace(-1, 1, 100),
            np.linspace(-1, 1, 100))
        w = np.array([w0, w1]).transpose(1, 2, 0)

        feature = PolynomialFeature(degree=1)
        X_train = feature.transform(x_train)
        X = feature.transform(x)
        model = BayesianRegression(alpha=1., beta=100.)

        # 依次切片X_train, y_train数据, 送入model.fit(), 然后绘制图形
        for begin, end in [[0, 0], [0, 1], [1, 2], [2, 3], [3, 20]]:
            model.fit(X_train[begin: end], y_train[begin: end])
            plt.subplot(1, 2, 1)
            plt.scatter(-0.3, 0.5, s=200, marker="x")
            # 以 model里的 mean, cov 作为参数, 绘制等高线图, 范围取值 w0, w1, w
            plt.contour(
                w0, 
                w1, 
                multivariate_normal.pdf(
                    w, 
                    mean=model.w_mean, 
                    cov=model.w_cov
                )
            )
            plt.gca().set_aspect('equal')
            plt.xlabel("$w_0$")
            plt.ylabel("$w_1$")
            plt.title("prior/posterior")

            plt.subplot(1, 2, 2)
            plt.scatter(x_train[:end], y_train[:end], s=100, facecolor="none", edgecolor="steelblue", lw=1)
            plt.plot(x, model.predict(X, sample_size=6), c="orange")
            plt.xlim(-1, 1)
            plt.ylim(-1, 1)
            plt.gca().set_aspect('equal', adjustable='box')
            plt.show()