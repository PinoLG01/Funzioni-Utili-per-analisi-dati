import math
import pandas as pd
import numpy as np
from lmfit import Parameters, fit_report, minimize, Parameter
import matplotlib.pyplot as plt


def pescadati(file_name='my_file.xlsx',foglio=0,colonne='A:E',listaRighe=[0,1,2],matrice=False):
	"""
	Prende i dati dal file_name. Le colonne vanno modificate per utilizzare quelle corrette (mantenendo la forma di una stringa). 
	Restituisce una tabella(DataFrame) che contiene le colonne del file excel specificato, e che ha solo le righe della lista specificata.
	Utilizzare range(a,b) per selezionare solo le righe [a+1 ,..., b+1]
	"""

	df = pd.read_excel(file_name, sheet_name=foglio, usecols=colonne)
	num_righe = df.shape[0]
	df.drop(list(set(range(num_righe)) - set(listaRighe)), inplace=True)

	if matrice:
		mat = df.to_numpy() #Passando il parametro "True" al fondo, pescadati restituisce una matrice di numpy invece che un DataFrame di Pandas
		return mat

	return df   #La funzione ritorna un DataFrame se non si specifica nulla, altrimenti una matrice di numpy


def assegnaerroretens(arrtens):
    """
	Assegna l'errore ad una lista/array di tensioni SCRITTA IN VOLT. Non lavora correttamente per tensioni superiori a 100V o inferiori a 100mV,
	ma controllare ciò che fa prima di utilizzarla.
	"""
    arrtens = list(arrtens)
    err = []

    for x in arrtens:
        if x < 1:
            err1 = x * 0.001 + 0.0001
        if x >= 10:
            err1 = x * 0.001 + 0.01
        if 1 <= x < 10:
            err1 = x * 0.001 + 0.001

    err.append(err1)
    return np.array(err) #Restituisce un array di errori indipendentemente che le tensioni siano in una lista o in un array, quindi attenzione
							#al fatto che arrtens ed err potrebbero essere cose diverse. Si consiglia di convertire arrtens ad array.


class modelclass():

    def __init__(self, func, numParams=1, Vals=None, ValMin=None, ValMax=None):
        self.func = func
        self.numParams = numParams

        if Vals is None:
            self.Vals = {'a': 1.0, 'b': 1.0, 'c': 1.0, 'd': 1.0, 'e': 1.0}
        else:
            self.Vals = Vals

        if ValMin is None:
            self.ValMin = {'a': None, 'b': None, 'c': None, 'd': None, 'e': None}
        else:
            self.ValMin = ValMin

        if ValMax is None:
            self.ValMax = {'a': None, 'b': None, 'c': None, 'd': None, 'e': None}
        else:
            self.ValMax = ValMax


class xy():

    def __init__(self, varx, vary, errx, erry):
        self.varx = varx
        self.vary = vary
        self.errx = errx
        self.erry = erry

#ESEMPIO DI UTILIZZO DELLE CLASSI:
"""
def somma(a,b):
	return a+b

Summa=modelclass(somma)

print(Summa.func(3,2))  #Stampa 5

x,y,sigmax,sigmay=[0,1],[1,2],[0.1,0.2],[0.1,0.2]

misure=xy(x,y,sigmax,sigmay)

print(misure.varx) #Stampa x, cioè stampa [0,1]
"""


"""
Il modello vero e proprio, va modificato da qui dentro per cambiare il numero di parametri in gioco e
la funzione di best fit (scrivendo model=a*x**2+b*x+c si ha una parabola. In tal caso, bisogna de-commentare la riga corrispondente alla 'c').
Se servono più di 4 parametri, aggiungerne altri utilizzando la stessa formula. Ad esempio e=vals['e']. model è una funzione da R in R
a parte i parametri che vanno fissati: x è un numero. Una volta fatto il fit, si può plottare model passando in input i parametri dati in
output dal fit. Se questo sembra complicato c'è una funzione "plotta" che semplifica il procedimento, ma è molto meno personalizzabile

NOTA 1: MODIFICARE PER MODIFICARE LA FUNZIONE DI BEST FIT. RICORDARE DI MODIFICARE IL NUMERO DI PARAMETRI: il numero di parametri indipendenti deve essere pari al numero di parametri che python usa, altrimenti
il fit non funziona correttamente

"""
def modelloesempio(pars, x):
    vals = pars.valuesdict()
    a = vals['a']
    b = vals['b']
    model = a / (np.sqrt(1 + (x / b)**2))  #Nota 1
    return model

"""#ESEMPIO DI CLASSE COMPLETA (cancellare le virgolette in cima e in fondo per eseguire)
lettere = ['a','b','c','d','e']
val_iniz = [1.0, 1.0, 1.0, 1.0, 1.0]       #I par. vanno definiti anche se poi non li si usa: vengono eliminati dalla funzione "Assegnaparam" dentro la funzione "fit"
val_min = [None, None, 1.0, 1.0, None]
val_max = [None, None, 0.0, None, None]		#Queste due liste dovrebbero essere composte di soli None se non si vogliono fornire vincoli ai parametri
											#In questo caso i vincoli sono che c deve essere compreso tra 0 e 1, e d deve essere maggiore di 1
											#Oss: questi vincoli non hanno senso pratico perché il modello non utilizza nè c nè d
dic_val_iniz = dict(zip(lettere, val_iniz))
dic_val_min = dict(zip(lettere, val_min))
dic_val_max = dict(zip(lettere, val_max))
print(dic_val_iniz,dic_val_min,dic_val_max)
classeEsempio = modelclass(modelloesempio, 2, dic_val_iniz, dic_val_min, dic_val_max)
#Contiene il modello con il numero di parametri in uso, il loro valore iniziale e i vincoli su di essi
"""


#Suggerimento: se si vogliono usare più modelli diversi nello stesso programma consiglio di mettere le prime 3 righe della
#	funzione in un "if True:", in modo da poter collassare questa parte, per una più chiara leggibilità


def derivata(FunzioneModello,parametri,x0):
	"""
	Dato il modello(che è una funzione da R in R), trova la derivata al punto x0 di esso, per calcolare l'errore indotto.
	Il secondo input è una struttura di lmfit che viene restituita dalla funzione fit(). Quindi questa funzione può essere utilizzata
	solo dopo il primo fit per farne un secondo utilizzando l'errore indotto
	"""
	h = 1e-8

	return (FunzioneModello(parametri, x0+h) - FunzioneModello(parametri, x0-h)) / (2*h)
	#Restituisce un numero, la derivata del modello a parametri fissati

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
                		('b', val['b'], True, mins['b'], maxs['b'], None, None),
                		('c', val['c'], True, mins['c'], maxs['c'], None, None),
                		('d', val['d'], True, mins['d'], maxs['d'], None, None),
						('e', val['e'], True, mins['e'], maxs['e'], None, None))
	fit_params = Assegnaparam(fit_params, FunzioneModello.numParams)

	def residual(pars,x,y):  			#NON MODIFICARE GLI INPUT DELLA FUNZIONE, E NEMMENO IL LORO ORDINE

		return (Funzionemodello.func(pars,x)-y)/(Classexy.erry)
	"""
	Siccome noi minimizziamo somma((yi-f(xi))/xerri) bisogna definire (yi-f(xi))/xerri perché poi la somma 
	la fa il programma da solo. Il residuo è definito come (yi-f(xi))/xerri ed è un numero per un dato i.
	x e data sono quelli che sotto chiamo xdata e ydata. SONO ARRAY e il programma funziona solo se lo sono.
	"""

	out = minimize(residual, fit_params, args=(xdata,), kws={'data': ydata},scale_covar=True)

	if stampa:
		print(fit_report(out),"\n\n===============================\n\n")

	return out.params

def plotta(ax, Classexy, i, 

		   nome = None, colore = None, 

		   FunzioneModello =None, parametri=None, DatiGrezzi = True, AncheCurvaFit = True):

	nom = nome[i] if type(nome) == type([]) else nome
	col = colore[i%len(colore)] if type(colore) == type([]) else colore

	if Datigrezzi == True:
		ax.plot(Classexy.varx, Classexy.vary, 'o', color=col)
		ax.errorbar(Classexy.varx, Classexy.vary, yerr = Classexy.erry, xerr = Classexy.errx, ecolor = 'black', ls = '')

	if AncheCurvaFit == True:
		spazio=np.linspace(min(Classexy.varx),max(Classexy.vary),1000)
		ax.plot(spazio,FunzioneModello.func(parametri,Classexy.varx),color=col,label=nom)


def testz(val1,val2,err1,err2):
	"""
	Dati due valori e due errori resituisce il valore di z(test z). Funziona anche con gli array, utile per fare molteplici test z
	tra dati: in questo caso restituisce un array che contiene le z corrispondenti
	"""
	return np.absolute(val1-val2)/np.sqrt(err1**2+err2**2)


def gradiente(func,x):
	h=1e-10
	res=np.zeros(len(x))
	l=len(x)
	for i in range(l):
		xh=x.copy()
		xh[i]=xh[i]+h
		xhh=x.copy()
		xhh[i]=xhh[i]-h
		dfi=(func(xh)-func(xhh))/(2*h)
		res[i]=dfi
	return res


def assegnaerrore(g,arr,err):
	"""
	Data una funzione di python g, che va definita con 'def g(arr)', restituisce l'errore gaussiano propagato dell'input sull'output
	"""
	grad=gradiente(g,arr)
	grad,err=np.array(grad),np.array(err)
	#print(grad,err)
	prod=np.dot(grad**2,err**2)
	#print(np.sqrt(prod))
	return np.sqrt(prod)
