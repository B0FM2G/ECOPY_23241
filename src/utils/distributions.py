import random
class Secondclass:
    def __init__(self, rand):
        self.random = rand

class UniformDistribution:
    def __init__(self, rand, a, b):
        self.rand = rand
        self.a = a
        self.b = b

    def pdf(self, x):
        if x >= self.a and x <= self.b:
            return 1 / (self.b - self.a)
        else:
            return 0

    def cdf(self, x):
        if x < self.a:
            return 0
        elif x > self.b:
            return 1
        else:
            return (x - self.a) / (self.b - self.a)

    def ppf(self, p):
        if p < 0 or p > 1:
            raise ValueError("p must be between 0 and 1")
        return self.a + p * (self.b - self.a)

    def gen_rand(self):
        return self.rand.uniform(self.a, self.b)

    def mean(self):
        if self.a == self.b:
            raise Exception("Moment undefined")
        return (self.a + self.b) / 2

    def median(self):
        return (self.a + self.b) / 2

    def variance(self):
        if self.a == self.b:
            raise Exception("Moment undefined")
        return ((self.b - self.a)) / 12

    def skewness(self):
        if self.a == self.b:
            raise Exception("Moment undefined")
        return 0

    def ex_kurtosis(self):
        if self.a == self.b:
            raise Exception("Moment undefined")
        return 1.8-3

    def mvsk(self):
        if self.a == self.b:
            raise Exception("Moments undefined")
        return [self.mean(), self.variance(), self.skewness(), self.ex_kurtosis()]

import math
import random
from pyerf import pyerf
class NormalDistribution:
    def __init__(self, rand, loc, scale):
        self.rand = rand
        self.loc = loc
        self.scale = scale

    def pdf(self, x):
        return (1 / (self.scale**0.5 * ((2 * math.pi) ** 0.5))) * (math.e ** (-0.5 * (((x - self.loc) / self.scale**0.5) ** 2)))

    def cdf(self, x):
        return 0.5 * (1 + math.erf((x - self.loc) / (self.scale**0.5 * (2 ** 0.5))))

    def ppf(self, p):
        return self.loc + self.scale**0.5 * (2 ** 0.5) * pyerf.erfinv(2 * p - 1)

    def gen_rand(self):
        return random.normalvariate(self.loc,self.scale)

    def mean(self):
        return self.loc

    def median(self):
        return self.loc

    def variance(self):
            return self.scale

    def skewness(self):
        return 0

    def ex_kurtosis(self):
        return 0

    def mvsk(self):
        return [self.mean(), self.variance(), self.skewness(), self.ex_kurtosis()]

class CauchyDistribution:
    def __init__(self, rand, x0, gamma):
        self.rand = rand
        self.loc = x0
        self.scale = gamma

    def pdf(self, x):
        return (1 / (math.pi * self.scale * (1 + ((x - self.loc) / self.scale) ** 2)))

    def cdf(self, x):
        return 0.5 + (1 / math.pi) * math.atan((x - self.loc) / self.scale)

    def ppf(self, p):
        return self.loc + self.scale * math.tan(math.pi * (p - 0.5))

    def gen_rand(self):
        return self.loc + self.scale * math.tan(math.pi * (self.rand.random() - 0.5))

    def mean(self):
        raise Exception("Moment undefined")

    def median(self):
        return self.loc

    def variance(self):
        raise Exception("Moment undefined")

    def skewness(self):
        raise Exception("Moment undefined")

    def ex_kurtosis(self):
        raise Exception("Moment undefined")

    def mvsk(self):
        raise Exception("Moments undefined")