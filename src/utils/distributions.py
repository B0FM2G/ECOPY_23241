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

class LogisticDistribution:
    def __init__(self, rand, location, scale):
        self.rand = rand
        self.location = location
        self.scale = scale

    def pdf(self,x):
        exponent = -(x - self.location) / self.scale
        pdf_value = math.exp(exponent)/(self.scale*(1 + math.exp(exponent)) ** 2)
        return pdf_value

    def cdf(self, x):
        exponent = -(x - self.location) / self.scale
        cdf_value = 1 / (1 + math.exp(exponent))
        return cdf_value

    def ppf(self, p):
        if 0 < p < 1:
            ppf_value = self.location - self.scale * math.log(1 / p - 1)
            return ppf_value
        else:
            raise ValueError("p-nek 0 és 1 között kell lennie")

    def gen_rand(self):
        u = self.rand.random()
        rand_value = self.location - self.scale * math.log(1 / u - 1)
        return rand_value

    def mean(self):
        if self.scale == 0:
            raise Exception("Moment undefined")
        return self.location

    def variance(self):
        if self.scale == 0:
            raise Exception("Moment undefined")
        variance = (math.pi ** 2 * self.scale ** 2) / 3
        return variance

    def skewness(self):
        if self.scale == 0:
            raise Exception("Moment undefined")
        skewness = 0
        return skewness

    def ex_kurtosis(self):
        if self.scale == 0:
            raise Exception("Moment undefined")
        ex_kurtosis = 1.2
        return ex_kurtosis

    def mvsk(self):
        if self.scale == 0:
            raise Exception("Moment undefined")
        return [self.mean(), self.variance(), self.skewness(), self.ex_kurtosis()]

import scipy.special as scp
class ChiSquaredDistribution:
    def __init__(self, rand, dof):
        self.rand = rand
        self.dof = dof

    def pdf(self,x):
        if x >= 0:
            pdf_value = (1 / (2 ** (self.dof / 2) * math.gamma(self.dof / 2))) * x ** (self.dof / 2 - 1) * math.exp(
                -x / 2)
            return pdf_value
        else:
            return 0

    def cdf(self, x):
        if x >= 0:
            cdf_value = scp.gammainc(self.dof/2, x/2)
            return cdf_value
        else:
            return 0
    def ppf(self, p):
        if 0 <= p <= 1:
            ppf_value = 2 * scp.gammaincinv(self.dof/2, p)
            return ppf_value
        else:
            raise ValueError("p-nek 0 és 1 között kell lennie")

    def gen_rand(self):
        return sum(random.gauss(0, 1) ** 2 for _ in range(int(self.dof)))

    def mean(self):
        return self.dof

    def variance(self):
        return 2 * self.dof

    def skewness(self):
        return (8 / self.dof) ** 0.5
    def ex_kurtosis(self):
        return 12 / self.dof

    def mvsk(self):
        return [self.mean(), self.variance(), self.skewness(), self.ex_kurtosis()]
