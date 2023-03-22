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

#PARTE 1: VARIABILI DEL MODULO, INCLUSI I MODELLI

Energie = {"Am": 5.486, "Np": 4.788, "Cm": 5.805}
energies_of_peaks = sorted([5.443, 5.388, 5.763, 4.771, 4.639])
_, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
Energie_mylar = pd.DataFrame({"Energie": [4.0, 4.5, 5.0, 5.5],
                              "Range": [2.90e-3, 3.44e-3, 4.04e-3, 4.67e-3]})


def retta(pars, x):  # Define a line. Pars is a dict of parameters and x an array of data
    vals = pars.valuesdict()
    a = vals['a']
    b = vals['b']
    return a*x + b

modelloRetta = ModelloConParametri(retta, ValNames=["a", "b"])

def Poly(pars, x):
    vals = pars.valuesdict()
    a = vals['a']
    b = vals['b']
    c = vals['c']
    return a*(x**2) + b * x + c

Pol = ModelloConParametri(Poly, ValNames = ["a", "b", "c"])


#PARTE 2: CALIBRAZIONE DELL'ELETTRONICA TRAMITE SEGNALI DI AMPIEZZA NOTA

#Pesca i dati e li ripulisce
df = pescadati("../excelfinto.xlsx", colonne='A:J', listaRighe=range(11, 21))
df.dropna(axis = 1, inplace = True)
df.columns = ["ch1", "ch2", "delta", "CHN", "CNT", "errore_ch"]

#Assegna ad x e y i valori che verranno usati nel fit, cioè V(CHN)
ydata = (df["ch1"].to_numpy() + df["ch2"].to_numpy())/2
yerr = df["delta"].to_numpy()/500
xdata = df["CHN"].to_numpy()
xerr = np.ones(len(xdata))
xy = XY(xdata, xerr, ydata, yerr)

#Fa il fit e plotta usando i parametri ottenuti
out = fit(modelloRetta, xy)
plotta(ax1, xy, FunzioneModello = modelloRetta, parametri = out.params)


#PARTE 3: CALIBRAZIONE DEL RIVELATORE UTILIZZANDO I PICCHI PRIMARI DELLA SORGENTE TRIPLA

#Pesca i dati e li ripulisce
df = pescadati("../excelfinto.xlsx", colonne = 'B:F', listaRighe = range(26, 29))
df.columns = ["Picco", "CNT", "MezzoPicco", "FWHM", "err"]

#Assegna i dati per un fit E(CHN_picco)
ydata = np.array(sorted(list(Energie.values())))
yerr = np.ones(len(ydata))*0.001
xdata = df["Picco"].to_numpy()
xerr = np.ones(len(xdata))
dati = XY(xdata, xerr, ydata, yerr)

#Fa il fit e plotta usando i parametri ottenuti
out = fit(modelloRetta, data2D = dati)
plotta(ax2, dati, FunzioneModello=modelloRetta, parametri=out.params)


#PARTE 4: COSTRUZIONE DELLA RETTA DI CALIBRAZIONE

#Definizione delle costanti
channels = np.array([912, 939, 1062, 1081, 1136])
                #    Np3, Np2, Am3,  Am2,  Cm2
u_channels = unp.uarray(channels, np.ones(len(channels))*1.702)

A_calibr, B_calibr = uncertainties.correlated_values(
    [out.params["a"].value, out.params["b"].value], out.covar)

#Definizione retta di calibrazione
def E(Ch):
    return A_calibr * Ch + B_calibr * np.ones(len(Ch))

#Calcolo delle energie dei picchi secondari e test z degli array
u_energies = E(u_channels)
actual_u_energies = unp.uarray(energies_of_peaks, np.ones(len(energies_of_peaks))*0.001)

Delta = actual_u_energies - u_energies

for x, y in zip(Delta, unp.nominal_values(Delta)/unp.std_devs(Delta)):
    print(x, y)


#PARTE 5: STUDIO DELL'ANDAMENTO DEI CONTEGGI TOTALI IN FUNZIONE DELLA PRESSIONE

#Pesca i dati e li pulisce
df = pescadati("../AlphaInAria.xlsx", colonne='J:K', listaRighe=range(4, 19))
df = df.dropna(axis = 1)  # elimina ogni colonna che abbia almeno un valore vuoto
df.columns = ["Pressione", "Conteggi"] # associa ad ogni colonna del file excel un nome

df_per_estrapolazione = df.loc[df["Pressione"].isin([695.0, 702.0])]  #Tiene solo i punti corrispondenti alle
                                                                    #    pressioni della regione di linearità

#Assegna dati per un fit di due punti: Pressione(Cnt)
xdata = df_per_estrapolazione["Conteggi"].to_numpy()
ydata = df_per_estrapolazione["Pressione"].to_numpy()
xerr = unp.sqrt(df_per_estrapolazione["Conteggi"].to_numpy())
yerr = np.ones(len(xdata))
xy = XY(xdata, xerr, ydata, yerr)

#Fa un fit e plotta usando i parametri ottenuti
out = fit(modelloRetta, xy)
plotta(ax3, xy, FunzioneModello = modelloRetta, parametri = out.params)

#Scrittura della retta come funzione python usando i parametri trovati
A_pressione, B_pressione = uncertainties.correlated_values(
    [out.params["a"].value, out.params["b"].value], out.covar)

def pressione(Cnt):
    return A_pressione * Cnt + B_pressione

metà_conteggio = max(df["Conteggi"])/2
u_metà_conteggio = uncertainties.ufloat(metà_conteggio, np.sqrt(metà_conteggio))


#PARTE 5.1: CALCOLO DEL RANGE TOTALE DELLE PARTICELLE IN ARIA

Pdimezz = pressione(u_metà_conteggio)
D = uncertainties.ufloat(61.2, 1.0)
Pstandard = uncertainties.ufloat(1013.2, 0.0)
Tstandard = uncertainties.ufloat(293.15, 0.0)
Tamb = uncertainties.ufloat(296.95, 1.0)

Ddimezz = (D * Pdimezz * Tstandard) / (Pstandard * Tamb) # D*(p1/2)/pstandar * Tstandard /Tamb
print(Ddimezz)

RResiduo = uncertainties.ufloat(1.385892, 0.1)
RangeTot = Ddimezz + RResiduo
print(RangeTot * 1.205e-3)


#PARTE 6: STUDIO DELL'ANDAMENTO IN FUNZIONE DELLO SPESSORE DI MYLAR ATTRAVERSATO


xdata = Energie_mylar["Energie"].to_numpy()
ydata = Energie_mylar["Range"].to_numpy()
xy = XY(xdata, np.ones(len(xdata))*0.0001, ydata, np.ones(len(ydata))*0.001e-3)

#Fit quadratico per estrapolare dopo 
out = fit(Pol, xy)
plotta(ax4, xy, FunzioneModello = Pol, parametri = out.params)







plt.show()