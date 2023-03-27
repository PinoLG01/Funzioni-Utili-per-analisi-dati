from fupad import *
import string
import uncertainties
from pprint import pprint
from uncertainties import unumpy as unp
import pandas as pd
import numpy as np
from lmfit import Parameters, fit_report, minimize, Parameter
import matplotlib.pyplot as plt
from uncertainties.unumpy import std_devs, nominal_values
import zfit
from zfit import z
import tensorflow as tf
import zfit.z.numpy as znp

df = pescadati("../SpettroAmericio.xlsx", colonne = "A:B", listaRighe = range(15,2063))

canali = df[0].to_numpy()
conteggi = df[1].to_numpy()
err_canali = np.ones(len(canali))
err_conteggi = np.sqrt(conteggi)

_, ax1 = plt.subplots()


def SingleCrystalBall(pars, x):
    vals = pars.valuesdict()
    a=vals['a']
    b=vals['b']
    c=vals['c']
    d=vals['d']
    e=vals['e']
    return e*CrystalBall(x, mu = a, sigma = b, alpha = c, n = d)

def TripleCrystalBall(pars, x):
    vals = pars.valuesdict()
    a=vals['a']
    b=vals['b']
    c=vals['c']
    d=vals['d']
    e=vals['e']
    f=vals['f']
    g=vals['g']
    h=vals['h']
    i=vals['i']
    return d*CrystalBall(x, mu = a, sigma = g, alpha = h, n = i) + e*CrystalBall(x, mu = b, sigma = g, alpha = h, n = i) + f*CrystalBall(x, mu = c, sigma = g, alpha = h, n = i)


modelloCrystalBall = ModelloConParametri(TripleCrystalBall, ValNames=["a","b","c","d","e","f","g","h","i"],
                                         ValStart ={"a": 1430, "b": 1385, "c": 1330, "d": 5e4, "e":1e4, "f":1e3,
                                                          "g": 5, "h":2, "i":1},
                                        ValMin = {"a":2, "b":2, "c":3, "d":4, "e":5, "f":6, "g":3, "h":2, "i":2})

dati = XY(canali, err_canali, conteggi, err_conteggi)
out = fit(modelloCrystalBall, dati)

plotta(ax1, dati, FunzioneModello = modelloCrystalBall, parametri = out.params)

plt.show()