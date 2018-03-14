#!/usr/bin/env python

"""
"""

import numpy.random as rnd

def randInRange(max): # returns integer, max: integer
    return rnd.randint(max)

def rand_un(): # returns floating point
    return rnd.uniform()

def randn (mu, sigma): # returns floating point, mu: floating point, sigma: floating point
    return rnd.normal(mu, sigma)
