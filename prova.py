import uncertainties
import math
import os
import string
from uncertainties import unumpy as unp
import pandas as pd
import numpy as np
from lmfit import Parameters, fit_report, minimize, Parameter
import matplotlib.pyplot as plt

Alfabeto = list(string.ascii_lowercase)

x = np.array([41., 42., 43., 44., 45., 46., 47., 48., 49., 50.])
y = np.array([33.9, 34.8, 34.7, 35.6, 36.4, 37.7, 36.8, 38.9, 38.6, 40.2])
sx = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

arr = unp.uarray(x, sx)


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


def make_starting_pardict(n):
    """
    Crea un dizionario degli n parametri che verranno usati(in ordine alfabetico), costruito in modo che
            ad ogni parametro corrispondano un valore minimo ed un valore massimo
    """
    final_dict = {}
    for i in range(n):
        param_dict = {"ValMin": None, "ValMax": None, "ValStart": None}
        final_dict[Alfabeto[i]] = param_dict
    return final_dict


class modelclass():

    def __init__(self, func, numParams=1, Vals=None, ValMin=None, ValMax=None):
        self.func = func
        self.numParams = numParams
		self.pardict = make_starting_pardict()
		


class xy():

    def __init__(self, varx, vary, errx, erry):
        self.varx = varx
        self.vary = vary
        self.errx = errx
        self.erry = erry


def derivata(FunzioneModello, parametri, x0):
    """
    Dato il modello(che è una funzione da R in R), trova la derivata al punto x0 di esso, per calcolare l'errore indotto.
    Il secondo input è una struttura di lmfit che viene restituita dalla funzione fit(). Quindi questa funzione può essere utilizzata
    solo dopo il primo fit per farne un secondo utilizzando l'errore indotto
    """
    h = 1e-8

    # Restituisce un numero, la derivata del modello a parametri fissati
    return (FunzioneModello(parametri, x0+h) - FunzioneModello(parametri, x0-h)) / (2*h)


def Assegnaparam(dict, num):
    lettere = ['a', 'b', 'c', 'd', 'e']
    for i in range(5 - num):
        del dict[lettere[-1 - i]]
    return dict


def fit(FunzioneModello, Classexy, stampa=True):
    """
    Il fit vero e proprio. Di solito si chiama fit(residual) dal momento che la funzione del residuo si chiama "residual".
    Si può però chiamare fit più volte utilizzando residui diversi. xdata e ydata SONO ARRAY e lo devono rimanere! Se il programma dà errore,
    è facile che sia colpa del fatto che non si stanno usando array: a meno di casi particolari, non lavorare mai con liste. Siccome le liste
    hanno un metodo comodo che gli array non hanno, e cioè list.append(elemento) che permette di aggiungere ad una lista un elemento ulteriore,
    si consiglia di trasformare l'array in lista, chiamare append() e poi ritrasformare in array.
    """
    val, mins, maxs = FunzioneModello.Vals, FunzioneModello.ValMin, FunzioneModello.ValMax
    fit_params = Parameters()
    fit_params.add_many(('a', val['a'], True, mins['a'], maxs['a'], None, None),
                        ('b', val['b'], True, mins['b'],
                         maxs['b'], None, None),
                        ('c', val['c'], True, mins['c'],
                         maxs['c'], None, None),
                        ('d', val['d'], True, mins['d'],
                         maxs['d'], None, None),
                        ('e', val['e'], True, mins['e'], maxs['e'], None, None))
    fit_params = Assegnaparam(fit_params, FunzioneModello.numParams)

    def residual(pars, x, y):  # NON MODIFICARE GLI INPUT DELLA FUNZIONE, E NEMMENO IL LORO ORDINE

        return (Funzionemodello.func(pars, x)-y)/(Classexy.erry)
    """
	Siccome noi minimizziamo somma((yi-f(xi))/xerri) bisogna definire (yi-f(xi))/xerri perché poi la somma 
	la fa il programma da solo. Il residuo è definito come (yi-f(xi))/xerri ed è un numero per un dato i.
	x e data sono quelli che sotto chiamo xdata e ydata. SONO ARRAY e il programma funziona solo se lo sono.
	"""

    out = minimize(residual, fit_params, args=(xdata,),
                   kws={'data': ydata}, scale_covar=True)

    if stampa:
        print(fit_report(out), "\n\n===============================\n\n")

    return out.params


def plotta(ax, Classexy, i,

           nome=None, colore=None,

           FunzioneModello=None, parametri=None, DatiGrezzi=True, AncheCurvaFit=True):

    nom = nome[i] if type(nome) == type([]) else nome
    col = colore[i % len(colore)] if type(colore) == type([]) else colore

    if Datigrezzi == True:
        ax.plot(Classexy.varx, Classexy.vary, 'o', color=col)
        ax.errorbar(Classexy.varx, Classexy.vary, yerr=Classexy.erry,
                    xerr=Classexy.errx, ecolor='black', ls='')

    if AncheCurvaFit == True:
        spazio = np.linspace(min(Classexy.varx), max(Classexy.vary), 1000)
        ax.plot(spazio, FunzioneModello.func(
            parametri, Classexy.varx), color=col, label=nom)
