import string
import uncertainties
from pprint import pprint
from uncertainties import unumpy as unp
import pandas as pd
import numpy as np
from lmfit import Parameters, fit_report, minimize, Parameter
import matplotlib.pyplot as plt
from uncertainties.unumpy import std_devs, nominal_values
from mdutils import MdUtils


Alfabeto = list(string.ascii_lowercase)

Redefine_Print = False

Massimo_Parametri = 26
Colori = ['blue','green','red','yellow','violet']

def pescadati(file_name = 'my_file.xlsx', foglio = 0, colonne = 'A:E', listaRighe = [0, 1, 2], header = None):
    """
    Prende i dati dal file_name. Mantenere la forma 'X:Y' per le colonne. 
    Restituisce una tabella (DataFrame) che contiene le colonne del file excel specificato, e che ha solo le righe della lista specificata.
    Utilizzare range(a,b) per selezionare solo le righe [a+1 ,..., b]:
            range(0,5) -> righe dalla 1 alla 5 del file (le prime 5)
            range(3,7) -> righe dalla 4 alla 7 del file
    Utilizzare header per inserire l'intero della riga che si vuole usare per i titoli. Ad es usare header=5 vuol
            dire che la riga 6 del file sarà usata per i titoli (0-indexed)
    """

    df = pd.read_excel(file_name, sheet_name = foglio, header = header,

                       usecols = colonne, skiprows = lambda x: x not in listaRighe,)

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


class ModelloConParametri():

    def __init__(self, func, ValNames=None,  # Lista lettere per nomi param
                 ValStart=None,  # 3 Dizionari del tipo {'a' : float}
                 ValMin=None,
                 ValMax=None):

        self.func = func

        pardict = make_starting_pardict(Massimo_Parametri)

        for key in list(pardict):
            if key in ValNames:
                pardict[key]["ValStart"] = 1.0
            else:
                pardict.pop(key)

        if ValStart is not None:
            for key in ValStart:
                pardict[key]["ValStart"] = ValStart[key]

        if ValMin is not None:
            for key in ValMin:
                pardict[key]["ValMin"] = ValMin[key]

        if ValMax is not None:
            for key in ValMax:
                pardict[key]["ValMax"] = ValMax[key]

        self.pardict = make_actual_pardict(pardict)


def make_actual_pardict(pardict):
    fit_params = Parameters()
    for key, dict_of_values in pardict.items():
        fit_params.add(key, value=dict_of_values["ValStart"],
                       min=dict_of_values["ValMin"] if (
                           dict_of_values["ValMin"] is not None) else -np.inf,
                       max=dict_of_values["ValMax"] if (
                           dict_of_values["ValMax"] is not None) else np.inf
                       )
    return fit_params


class XY():

    def __init__(self, x, xerr, y, yerr):  # Both x and y are uarrays, provided by unumpy package
        self.x = unp.uarray(x, xerr)
        self.y = unp.uarray(y, yerr)


def derivata(Modello, parametri, x0):
    """
    Dato il modello(che è una funzione da R in R), trova la derivata al punto x0 di esso, per calcolare l'errore indotto.
    Il secondo input è una struttura di lmfit che viene restituita dalla funzione fit(). Quindi questa funzione può essere utilizzata
    solo dopo il primo fit per farne un secondo utilizzando l'errore indotto
    """
    h = 1e-15

    # Restituisce un numero, la derivata del modello a parametri fissati
    return (Modello.func(parametri, x0+h) - Modello.func(parametri, x0-h)) / (2*h)


def fit(FunzioneModello, data2D, verbose=True):
    """
    Il fit vero e proprio. Di solito si chiama fit(residual) dal momento che la funzione del residuo si chiama "residual".
    Si può però chiamare fit più volte utilizzando residui diversi. xdata e ydata SONO ARRAY e lo devono rimanere! Se il programma dà errore,
    è facile che sia colpa del fatto che non si stanno usando array: a meno di casi particolari, non lavorare mai con liste. Siccome le liste
    hanno un metodo comodo che gli array non hanno, e cioè list.append(elemento) che permette di aggiungere ad una lista un elemento ulteriore,
    si consiglia di trasformare l'array in lista, chiamare append() e poi ritrasformare in array.
    """
    fit_params = FunzioneModello.pardict
    x = nominal_values(data2D.x)
    y = nominal_values(data2D.y)
    yerr = std_devs(data2D.y)
    def residual(pars, x, y):  # NON MODIFICARE GLI INPUT DELLA FUNZIONE, E NEMMENO IL LORO ORDINE
        return (FunzioneModello.func(pars, x) - y) / yerr
    """
	Siccome noi minimizziamo somma((yi-f(xi))/xerri) bisogna definire (yi-f(xi))/xerri perché poi la somma 
	la fa il programma da solo. Il residuo è definito come (yi-f(xi))/xerri ed è un numero per un dato i.
	x e data sono quelli che sotto chiamo xdata e ydata. SONO ARRAY e il programma funziona solo se lo sono.
	"""

    out = minimize(residual, fit_params, args=(x,),
                   kws={'y': y}, scale_covar=True)

    if verbose:
        print(fit_report(out), "\n\n===============================\n\n")

    return out


def plotta(ax, data2D, i = None,

           nome = None, colore = None,

           FunzioneModello = None, parametri = None, DatiGrezzi = True, AncheCurvaFit = True):


    #Sia nome che colore possono esser sia liste che stringhe. Se liste, viene utilizzato l'i-esimo elemento
    #   se stringhe, viene utilizzata la stringa stessa. Se sono liste, ricordarsi di passare il parametro i
    title = nome[i] if type(nome) == type([]) else nome
    color = colore[i % len(colore)] if type(colore) == type([]) else colore

    xdata, ydata = nominal_values(data2D.x), nominal_values(data2D.y)
    xerr, yerr = std_devs(data2D.x), std_devs(data2D.y)

    if DatiGrezzi:
        ax.plot(xdata, ydata, 'o', color = color)
        ax.errorbar(xdata, ydata, yerr = yerr, xerr = xerr, ecolor = 'black', ls = "")

    if AncheCurvaFit:
        spazio = np.linspace(min(xdata), max(xdata), 1000)
        ax.plot(spazio, FunzioneModello.func(parametri, spazio), color = color, label = title)


md = MdUtils(file_name = "../Output")

def write(*args, **kwargs):
    output = ""
    for x in args:
        output += str(x)
    md.new_paragraph(output)

    print(args, k)



# MWE OF HOW TO USE THESE FUNCTIONS:
def main():

    def retta(pars, x): #Define a line. Pars is a dict of parameters and x an array of data
        vals = pars.valuesdict()
        a = vals['a']
        b = vals['b']
        return a*x + b

    #Define a ModelloConParametri of the line function, by inserting both the function
    #   itself and the names of the values. One could also add minimum/maximum values
    #   of parameters, or specify starting values if 1.0 gives wrong results
    modelloRetta = ModelloConParametri(retta, ValNames = ["a", "b"])

    x = np.array([41., 42., 43., 44., 45., 46., 47., 48., 49., 50.])
    y = np.array([33.9, 34.8, 34.7, 35.6, 36.4, 37.7, 36.8, 38.9, 38.6, 40.2])
    sx = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    sy = np.ones(len(x))*0.1

    xy = XY(x, sx, y, sy)

    #Actual fit
    out = fit(modelloRetta, xy)


    _,ax1=plt.subplots()

    plotta(ax1, xy, FunzioneModello = modelloRetta, parametri = out.params)
    plt.show()

    cov = out.covar

    a, b = uncertainties.correlated_values([out.params["a"].value, out.params["b"].value], cov)

    print(a,b)

    print(a*45.5+b) #Gives correct result of 36.76 \pm 0.17 so it keeps the covariance into account


    md.create_md_file()

if __name__ == "__main__":
    main()