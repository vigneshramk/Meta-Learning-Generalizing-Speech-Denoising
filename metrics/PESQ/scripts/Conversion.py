#!/usr/bin/env python

from __future__ import division
import sys, os
from math import sqrt, pi, atan2, log, pow, cos, log, exp

class Conversion(object):
    def mos2r(self, mos):
        """ With given MOS LQO return R-factor  (1 < MOS < 4.5) """
        D = -903522 + 1113960 * mos - 202500 * mos * mos
        if D < 0:
            D = 0
        h = 1/3 * atan2(15*sqrt(D), 18556-6750*mos)
        R = 20/3 * (8 - sqrt(226) * cos(h+pi/3))
        return R > 100 and 100.0 or R

    def r2mos(self, r):
        """ With given R-factor return MOS """
        if r < 0:
            return 1
        if r > 100:
            return 4.5
        return 1 + 0.035 * r  + r * (r - 60) * (100 - r) * 7e-6

    def delay2id(self, Ta):
        """ Delay Ta (ms) render to Id penalty according to ITU-T G.107 and G.108
        recommendations. """
        if Ta < 100:
            Id = 0
        else:
            X = log(Ta/100) / log(2)
            Id = 25.0 * (
                pow((1 + pow(X, 6)), 1.0/6) - \
            3 * pow(1+ pow(X/3, 6), 1.0/6 ) + 2
            )
        return Id

    def pesq2mos(self, pesq):
        """ Return MOS LQO value (within 1..4.5) on PESQ value (within -0.5..4.5).
        Mapping function given from P.862.1 (11/2003) """
        return 0.999 + (4.999-0.999) / (1+exp(-1.4945*pesq+4.6607))

    def mos2pesq(self, mos):
        """ Return PESQ value (within -0.5..4.5) on MOS LQO value (within 1..4.5).
        Mapping function given from P.862.1 (11/2003) """
        inlog =(4.999-mos)/(mos-0.999)
        return (4.6607-log(inlog)) / 1.4945


if __name__ == '__main__':
    test1 = Conversion()
    mos_score = test1.pesq2mos(2.3)