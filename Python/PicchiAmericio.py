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

df = pescadati("../SpettroAmericio.xlsx", colonne = "A:B", listaRighe = range(15,2063))

canali = df[0].to_numpy()
conteggi = np.maximum(df[1].to_numpy(), np.ones(len(canali)))
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
    a=vals['a'] # Mu1
    b=vals['b'] # Mu2
    c=vals['c'] # Mu3
    d=vals['d'] # Amp1
    e=vals['e'] # Amp2
    f=vals['f'] # Amp3
    g=vals['g'] # sigma
    h=vals['h'] # alfa
    i=vals['i'] # n
    return d*CrystalBall(x, mu = a, sigma = g, alpha = h, n = i) + e*CrystalBall(x, mu = b, sigma = g, alpha = h, n = i) + f*CrystalBall(x, mu = c, sigma = g, alpha = h, n = i)


modelloCrystalBall = ModelloConParametri(TripleCrystalBall, ValNames=["a","b","c","d","e","f","g","h","i"],
                                         ValStart ={"a": 1430, "b": 1385, "c": 1330, "d": 5e4, "e":1e4, "f":1e3,
                                                          "g": 5, "h":2, "i":1},
                                        ValMin = {"a":2, "b":2, "c":3, "d":4, "e":5, "f":6, "g":3, "h":1, "i":0})

dati = XY(canali, err_canali, conteggi, err_conteggi)
out = fit(modelloCrystalBall, dati, debug = False)

plotta(ax1, dati, FunzioneModello = modelloCrystalBall, parametri = out.params)

val_param = [out.params[par].value for par in out.params]

print(val_param)

A, B, C, D, E, F, G, H, I = uncertainties.correlated_values(val_param, out.covar)

def integrale_crystal(x, mu, sigma, alpha, n, amp):
    somma = uncertainties.ufloat(0, 0)
    for i in x:
        u_x = uncertainties.ufloat(i+1, 1)
        somma += amp*CrystalBall(u_x, mu, sigma, alpha, n)
    return somma

print(D*u_CrystalBall(x = uncertainties.ufloat(1300, 1), mu = A, sigma = G, alpha = H, n = I))

def integrate_u_crystal(x, mu, sigma, alpha, n, amp):
    somma = 0.
    for i in x: # Se x è un uarray, considera anche le incertezze, altrimenti no
        somma += amp*u_CrystalBall(i, mu, sigma, alpha, n)
    return somma

listacanali = range(2048)

I1, I2, I3 = (
          integrate_u_crystal(listacanali, mu = A, sigma = G, alpha = H, n = I, amp = D), 
          integrate_u_crystal(listacanali, mu = B, sigma = G, alpha = H, n = I, amp = E),
          integrate_u_crystal(listacanali, mu = C, sigma = G, alpha = H, n = I, amp = F))

tot = I1 + I2 + I3

print(I1, I2, I3, f'Totale: {tot}', sep = "\t\t")

print(f"BR1: {I1/tot}", f'BR2: {I2/tot}', f'BR3: {I3/tot}')

i1, ui1, i2, ui2, i3, ui3 = I1.nominal_value, I1.std_dev, I2.nominal_value, I2.std_dev, I3.nominal_value, I3.std_dev

err_sbagliato = (1/(i1+i2+i3)**2) * np.sqrt( ((i2+i3)**2)* ui1**2 + (i1**2) * (ui2**2 + ui3**2) )

print(err_sbagliato)

print('Result = {:f}'.format(err_sbagliato))

plt.show()