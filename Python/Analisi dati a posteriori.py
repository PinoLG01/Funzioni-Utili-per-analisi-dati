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


Energie = {"Am": 5.486, "Np": 4.788, "Cm": 5.805}
energies_of_peaks = sorted([5.443, 5.388, 5.763, 4.771, 4.639])

df = pescadati("../excelfinto.xlsx", colonne = 'A:J', listaRighe = range(11,21))
df = df.dropna(axis = 1)
df.columns = ["ch1", "ch2", "delta", "CHN", "CNT", "errore_ch"]

print(df)

ydata = (df["ch1"].to_numpy() + df["ch2"].to_numpy())/2
yerr = df["delta"].to_numpy()/500
#yerr = np.ones(len(ydata))*0.08
xdata = df["CHN"].to_numpy()
xerr = np.ones(len(xdata))

def retta(pars, x): #Define a line. Pars is a dict of parameters and x an array of data
    vals = pars.valuesdict()
    a = vals['a']
    b = vals['b']
    return a*x + b

modelloRetta = ModelloConParametri(retta, ValNames = ["a", "b"])

xy = XY(xdata, xerr, ydata, yerr)
print(ydata)
out = fit(modelloRetta, xy)

_,(ax1, ax2)=plt.subplots(1, 2)

plotta(ax1, xy, FunzioneModello = modelloRetta, parametri = out.params)


df = pescadati("../excelfinto.xlsx", colonne = 'B:F', listaRighe = range(26,29))
df.columns = ["Picco", "CNT", "MezzoPicco", "FWHM", "err"]

ydata = np.array(sorted(list(Energie.values())))
yerr = np.ones(len(ydata))*0.001
xdata = df["Picco"].to_numpy()
xerr = np.ones(len(xdata))

dati = XY(xdata, xerr, ydata, yerr)

out = fit(modelloRetta, data2D = dati)

plotta(ax2, dati, FunzioneModello = modelloRetta, parametri = out.params)
plt.show()

channels = np.array([912, 939, 1062, 1072, 1136])
#                    Np3, Np2, Am3,  Am2,  Cm2
u_channels = unp.uarray(channels, np.ones(len(channels))*1.702)

a, b = uncertainties.correlated_values([out.params["a"].value, out.params["b"].value], out.covar)

u_energies = a*u_channels + b*np.ones(len(channels))

actual_u_energies = unp.uarray(energies_of_peaks, np.ones(len(energies_of_peaks))*0.001)

Delta = actual_u_energies - u_energies

for x,y in zip(Delta, unp.nominal_values(Delta)/unp.std_devs(Delta)):
    print(x,y,sep="\t")

