import math
import random
class LaplaceDistribution:
    def __init__(self, rand, loc, scale):
        self.rand = rand
        self.loc = loc
        self.scale = scale

    def pdf(self, x):
        return (1 / (2 * self.scale)) * math.exp(-abs(x - self.loc) / self.scale)

    def cdf(self, x):
        if x < self.loc:
            return 0.5 * math.exp((x - self.loc) / self.scale)
        else:
            return 1 - 0.5 * math.exp(-(x - self.loc) / self.scale)

    def ppf(self, p):
        if p < 0 or p > 1:
            raise ValueError("p értékének 0 és 1 között kell lennie")
        if p < 0.5:
            return self.loc + self.scale * math.log(2 * p)
        else:
            return self.loc - self.scale * math.log(2 - 2 * p)

    def gen_rand(self):
        u = random.uniform(0, 1)
        if u < 0.5:
            return self.loc + self.scale * math.log(2 * u)
        else:
            return self.loc - self.scale * math.log(2 - 2 * u)

    def mean(self):
        if self.scale == 0:
            raise Exception("Moment undefined")
        else:
            return self.loc

    def variance(self):
        if self.scale == 0:
            raise Exception("Moment undefined")
        return 2 * (self.scale ** 2)

    def skewness(self):
        if self.scale == 0:
            raise Exception("Moment undefined")
        return 0

    def ex_kurtosis(self):
        if self.scale == 0:
            raise Exception("Moment undefined")
        return 3.0

    def mvsk(self):
        if self.scale == 0:
            raise Exception("Moment undefined")
        return [self.mean(), self.variance(), self.skewness(), self.ex_kurtosis()]

class ParetoDistribution:
    def __init__(self, rand, scale, shape):
        self.rand = rand
        self.scale = scale
        self.shape = shape

    def pdf(self, x):
        if x < self.scale:
            return 0
        else:
            return self.shape * (self.scale ** self.shape) / (x ** (self.shape + 1))

    def cdf(self, x):
        if x < self.scale:
            return 0
        else:
            return 1 - (self.scale / x) ** self.shape

    def ppf(self, p):
        if p < 0 or p > 1:
            raise ValueError("p értékének 0 és 1 között kell lennie")
        else:
            return self.scale / ((1 - p) ** (1 / self.shape))

    def gen_rand(self):
        u = random.uniform(0, 1)
        return self.ppf(u)

    def mean(self):
        if self.shape <= 1:
            raise Exception("Moment undefined")
        else:
            return (self.shape * self.scale) / (self.shape - 1)

    def variance(self):
        if self.shape <= 2:
            raise Exception("Moment undefined")
        else:
            return (self.scale ** 2 * self.shape) / ((self.shape - 1) ** 2 * (self.shape - 2))

    def skewness(self):
        if self.shape <= 3:
            raise Exception("Moment undefined")
        else:
            return (2 * (1 + self.shape)) / (self.shape - 3) * math.sqrt((self.shape - 2) / self.shape)

    def ex_kurtosis(self):
        if self.shape <= 4:
            raise Exception("Moment undefined")
        else:
            return 6 * ((self.shape ** 3 + self.shape ** 2 - 6 * self.shape - 2) / (
                        self.shape * (self.shape - 3) * (self.shape - 4)))

    def mvsk(self):
        if self.shape <= 1:
            raise Exception("Moment undefined")
        else:
            return [self.mean(), self.variance(), self.skewness(), self.ex_kurtosis()]