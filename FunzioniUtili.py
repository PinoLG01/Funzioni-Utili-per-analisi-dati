global yerr#,parametri, xdata,ydata 

#LASCIARE LA PRIMA RIGA COMMENTATA SE NON SI USA IL main(), altrimenti  probabilmente andrà decommentata

import math
import pandas as pd
import numpy as np
from lmfit import Parameters, fit_report, minimize, Parameter
import matplotlib.pyplot as plt

def pescadati(file_name='my_file.xlsx',colonne='A:E',listaRighe=[0,1,2],matrice=False):
	"""
	Prende i dati dal file_name. Le colonne vanno modificate per utilizzare quelle corrette (mantenendo la forma di una stringa). 
	Restituisce una tabella(DataFrame) che contiene le colonne del file excel specificato, e che ha solo le righe della lista specificata.
	Utilizzare range(a,b) per selezionare solo le righe [a+1 ,..., b]. Ad esempio se listaRighe=range(3,5), la funzione utilizza le righe da 4 a 5,
	cioè 4 e 5.
	"""
	df = pd.read_excel(file_name,usecols=colonne)
	num_righe = df.shape[0]
	df.drop(list(set(range(num_righe))-set(listaRighe)),inplace=True)

	if matrice:
		mat=df.to_numpy() #Passando il parametro "True" al fondo, pescadati restituisce una matrice di numpy invece che un DataFrame di Pandas
		return mat

	return df   #La funzione ritorna un DataFrame se non si specifica nulla, altrimenti una matrice di numpy

def assegnaerroretens(arrtens):
	"""
	Assegna l'errore ad una lista/array di tensioni SCRITTA IN VOLT. Non lavora correttamente per tensioni superiori a 100V o inferiori a 100mV,
	ma controllare ciò che fa prima di utilizzarla.
	"""
	arrtens=list(arrtens)
	err=[]
	for x in arrtens:
		if x<1:
			err1=x*0.001+0.0001
		if x>=10:
			err1=x*0.001+0.01
		if 1<=x<10:
			err1=x*0.001+0.001
	err.append(err1)
	return np.array(err) #Restituisce un array di errori indipendentemente che le tensioni siano in una lista o in un array, quindi attenzione
						 #al fatto che arrtens ed err potrebbero essere cose diverse. Si consiglia di convertire arrtens ad array.


def model(pars,x):
	"""
	Il modello vero e proprio, va modificato da qui dentro per cambiare il numero di parametri in gioco e
	la funzione di best fit (scrivendo model=a*x**2+b*x+c si ha una parabola. In tal caso, bisogna de-commentare la riga corrispondente alla 'c').
	Se servono più di 4 parametri, aggiungerne altri utilizzando la stessa formula. Ad esempio e=vals['e']. model è una funzione da R in R
	a parte i parametri che vanno fissati: x è un numero. Una volta fatto il fit, si può plottare model passando in input i parametri dati in
	output dal fit. Se questo sembra complicato c'è una funzione "plotta" che semplifica il procedimento, ma è molto meno personalizzabile
	"""
	vals = pars.valuesdict()
	a=vals['a']
	b=vals['b']
	#c=vals['c']
	#d=vals['d']
	model = a*x+b  #MODIFICARE PER MODIFICARE LA FUNZIONE DI BEST FIT. RICORDARE DI MODIFICARE IL NUMERO DI PARAMETRI commentando o decommentando 
					#quelli non utilizzati: il numero di parametri indipendenti deve essere pari al numero di parametri che python usa, altrimenti
					#il fit non funziona correttamente

	return model #Restituisce un numero

def d_model(x0, parametri):
	"""
	Dato il modello(che è una funzione da R in R), trova la derivata al punto x0 di esso, per calcolare l'errore indotto.
	Il secondo input è una struttura di lmfit che viene restituita dalla funzione fit(). Quindi questa funzione può essere utilizzata
	solo dopo il primo fit per farne un secondo utilizzando l'errore indotto
	"""
	h = 1e-5

	return (model(parametri,x0+h)-model(parametri,x0-h))/(2*h) #Restituisce un numero, la derivata del modello a parametri fissati

def residual(pars, x,data):  #NON MODIFICARE GLI INPUT DELLA FUNZIONE, E NEMMENO IL LORO ORDINE
	"""
	Siccome noi minimizziamo somma((yi-f(xi))/xerri) bisogna definire (yi-f(xi))/xerri perché poi la somma 
	la fa il programma da solo. Il residuo è definito come (yi-f(xi))/xerri ed è un numero per un dato i.
	x e data sono quelli che sotto chiamo xdata e ydata. SONO ARRAY e il programma funziona solo se lo sono.
	"""
	residuo=(model(pars,x)-data)/yerr
	return residuo

def fit(residuo,xdata,ydata):
	"""
	Il fit vero e proprio. Di solito si chiama fit(residual) dal momento che la funzione del residuo si chiama "residual".
	Si può però chiamare fit più volte utilizzando residui diversi. xdata e ydata SONO ARRAY e lo devono rimanere! Se il programma dà errore,
	è facile che sia colpa del fatto che non si stanno usando array: a meno di casi particolari, non lavorare mai con liste. Siccome le liste
	hanno un metodo comodo che gli array non hanno, e cioè list.append(elemento) che permette di aggiungere ad una lista un elemento ulteriore,
	si consiglia di trasformare l'array in lista, chiamare append() e poi ritrasformare in array.
	"""
	fit_params = Parameters()
	fit_params.add('a', value=4)#Valori iniziali di a e b(bisogna aggiungere c,d se serve)
	fit_params.add('b', value=4)
	#fit_params.add('c',value=50,min=0)

	out = minimize(residuo, fit_params, args=(xdata,), kws={'data': ydata},scale_covar=True)
	print(fit_report(out),"\n\n===============================\n\n")
	#stampa la tabella con tutti i valori utili(tra cui X^2)
	parametri=out.params
	return out.params

#ESEMPIO(decommentare le righe successive a questa):
#yerr=np.array([0.1,0.1,0.1])
#data=np.array([0,1,2])
#ydata=np.array([1,4,9])
#fit(residual, xdata, ydata)


def plotta(parametri,ax,num,xdata,ydata,xerr):
	"""
	Stampa dati e curva di best fit usando rispettivamente i parametri(ottenuti da fit()), la figura da utilizzare*, e il colore da utilizzare,
	che è un numero, da scegliere dalla lista sotto. Ad esempio se num=3, il grafico sarà rosso(le liste sono numerate da 0).
	I dati e la curva hanno lo stesso colore, rendendo facile plottare più set di dati con più curve di best fit corrispondenti, semplicemente
	variando il parametro num.
	"""
	colori=['grey','blue','green','red','yellow','violet']
	spazio=np.linspace(min(xdata),max(xdata),1000)
	ax.plot(xdata,ydata,'o',color=colori[num])
	ax.legend(str('Titolo')) #Sostituire 'Titolo' con il titolo che si vuole dare al grafico. 
							  #Utile specialmente se si lavora con più grafici diversi, per distinguerli
	ax.plot(spazio,model(parametri,spazio),color=colori[num],label='nome')  #Modificare 'nome' per dare un nome alla singola serie
																# del grafico(infatti 'Titolo' dà il nome all'intero piano cartesiano, mentre 'nome' solo
																# ad una serie di dati)
	ax.errorbar(xdata,ydata,yerr=yerr,ecolor='black', ls=" ",xerr=xerr)
	#ax.legend(loc='best')

def plottadatigrezzi(ax,arr1,arr2,err1,err2,num):
	"""
	Inserisce nel grafico ax solo i dati x e y con i loro errori, ma senza fit o altro. arr1 e err1 sono gli array delle x
	"""
	ax.plot(arr1,arr2,'o',color=colori[num],label=lista_label_1[num])
	ax.errorbar(arr1,arr2,xerr=err1,yerr=err2,ecolor='black', ls=" ")

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

#Esempio:
def somma(numeri):
	return numeri[0]+numeri[1] #Questa funzione restituisce il rapporto tra il primo ed il secondo elemento dell'array in input
numeri=[1,2]
errori=[0.1,0.2]
print("La somma tra i numeri è ",rapporto(numeri),"+-",assegnaerrore(rapporto,numeri,errori))