import uncertainties
import math
import os
import string
from uncertainties import unumpy as unp
import pandas as pd
import numpy as np
from lmfit import Parameters, fit_report, minimize, Parameter
import matplotlib.pyplot as plt

Energie = {"Am": 5.486, "Np": 4.788, "Cm": 5.805}

def derivata(FunzioneModello, parametri, x0):
    """
    Dato il modello(che è una funzione da R in R), trova la derivata al punto x0 di esso, per calcolare l'errore indotto.
    Il secondo input è una struttura di lmfit che viene restituita dalla funzione fit(). Quindi questa funzione può essere utilizzata
    solo dopo il primo fit per farne un secondo utilizzando l'errore indotto
    """
    h = 1e-8

    # Restituisce un numero, la derivata del modello a parametri fissati
    return (FunzioneModello(parametri, x0+h) - FunzioneModello(parametri, x0-h)) / (2*h)


def pescadati(file_name='my_file.xlsx', foglio=0, colonne='A:E', listaRighe=[0, 1, 2], header=None):
    """
    Prende i dati dal file_name. Mantenere la forma 'X:Y' per le colonne. 
    Restituisce una tabella (DataFrame) che contiene le colonne del file excel specificato, e che ha solo le righe della lista specificata.
    Utilizzare range(a,b) per selezionare solo le righe [a+1 ,..., b]:
            range(0,5) -> righe dalla 1 alla 5 del file (le prime 5)
            range(3,7) -> righe dalla 4 alla 7 del file
    Utilizzare header per inserire l'intero della riga che si vuole usare per i titoli. Ad es usare header=5 vuol
            dire che la riga 6 del file sarà usata per i titoli (0-indexed)
    """

    df = pd.read_excel(file_name, sheet_name=foglio, header=header,

                       usecols=colonne, skiprows=lambda x: x not in listaRighe,)

    return df

df = pescadati("./excelfinto.xlsx", colonne = 'A:J', listaRighe = range(11,21))
df = df.dropna(axis = 1)
df.columns = ["ch1", "ch2", "delta", "CHN", "CNT", "errore_ch"]

def model(pars,x):
    vals = pars.valuesdict()
    a=vals['a']
    b=vals['b']
    model = a*x+b
    return model

def residual(pars, x,data):
    residuo=(model(pars,x)-data)/yerr
    return residuo

ydata = (df["ch1"].to_numpy() + df["ch2"].to_numpy())/2
yerr = df["delta"].to_numpy()/500
#yerr = np.ones(len(ydata))*0.08
xdata = df["CHN"].to_numpy()

fit_params = Parameters()
fit_params.add('a', value=1.)
fit_params.add('b', value=1.)

out = minimize(residual, fit_params, args=(xdata,), kws={'data': ydata},scale_covar=True)

print("ppp",fit_report(out))

spazio = np.linspace(min(xdata)-1,max(xdata)+1,100)

fig, ax = plt.subplots()
ax.set_xlabel('variable x - u.m.')
ax.set_ylabel('variable y - u.m.')
plt.plot(xdata,ydata,'o')
plt.plot(spazio,model(out.params,spazio))
plt.errorbar(xdata,ydata,yerr=yerr,ecolor='black', ls=" ")
plt.show()

df = pescadati("./excelfinto.xlsx", colonne = 'B:F', listaRighe = range(26,29))
df.columns = ["Picco", "CNT", "MezzoPicco", "FWHM", "err"]
print(df)

ydata = np.array(sorted(list(Energie.values())))
yerr = np.ones(len(ydata))*0.001

xdata = df["Picco"].to_numpy()

out = minimize(residual, fit_params, args=(xdata,), kws={'data': ydata},scale_covar=True)

print(fit_report(out))

spazio = np.linspace(min(xdata)-1,max(xdata)+1,100)

fig, ax = plt.subplots()
ax.set_xlabel('variable x - u.m.')
ax.set_ylabel('variable y - u.m.')
plt.plot(xdata,ydata,'o')
plt.plot(spazio,model(out.params,spazio))
plt.errorbar(xdata,ydata,yerr=yerr,ecolor='black', ls=" ")
#plt.show()

print(out.covar)
print(np.sqrt(out.covar[0,0]))
print(out.params)
def sigma_E(ch, sigma_ch, cov, params):
    a = params["a"].value
    b = params["b"].value
    sigma_a = params["a"].stderr
    sigma_b = params["b"].stderr
    sigma_ab = cov[1,0]

    termine1 = (ch*sigma_a)**2
    termine2 = (sigma_b)**2
    termine3 = (a*sigma_ch)**2
    termine4 = 2*ch*sigma_ab
    print(termine1, termine2, termine3)
    return a*ch + b, np.sqrt(termine1 + termine2 + termine3 + termine4)

print(sigma_E(1072, 1.702, out.covar, out.params))
