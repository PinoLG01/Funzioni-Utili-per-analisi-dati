global yerr#,parametri, xdata,ydata 

#LASCIARE LA PRIMA RIGA COMMENTATA SE NON SI USA IL main(), altrimenti  probabilmente andrà decommentata

import math
import pandas as pd
import numpy as np
from lmfit import Parameters, fit_report, minimize, Parameter
import matplotlib.pyplot as plt
yerr=np.zeros(5)
def main():
	global yerr

	#Pesca i dati:

	Inv1=pescadati('operazionali.ods','Foglio1','B:H',range(9,55)) #Crea una tabella coi dati
	Inv1.reset_index(drop=True, inplace=True)
	Inv1.columns=['Vin','errVin','Vout','errVout','f','G','errG'] #Dà i nomi alle colonne della tabella

	Inv10=pescadati('operazionali.ods','Foglio1','B:H',range(60,94))
	Inv10.reset_index(drop=True, inplace=True)
	Inv10.columns=['Vin','errVin','Vout','errVout','f','G','errG']

	NonInv=pescadati('operazionali.ods','Foglio1','B:H',range(100,131))
	NonInv.reset_index(drop=True, inplace=True)
	NonInv.columns=['Vin','errVin','Vout','errVout','f','G','errG']

	Int=pescadati('operazionali.ods','Foglio_2','B:H',range(8,39)) #Crea una tabella coi dati
	Int.reset_index(drop=True, inplace=True)
	Int.columns=['Vin','errVin','Vout','errVout','f','G','errG'] #Dà i nomi alle colonne della tabella

	Logar=pescadati('operazionali.ods','Foglio_2','B:E',range(45,74))
	Logar.reset_index(drop=True, inplace=True)
	Logar.columns=['V0','errV0','V_s','errV_s']
	#print(Logar)
	#print(Inv1,Inv10,NonInv)  #Consiglio di decommentare la riga per vedere come sono fatte le tabelle
								# però sconsiglio di lasciarlo decommentato quando si va avanti
								# perché i print sono già lunghi
								# quindi si affolla solo la schermata del print con cose inutili

	#Cose utili per il ciclo for(Lista dei nomi, Lista dei dati, Dichiarazione delle figure, Lista delle figure)

	label=['Invertente 1','Invertente 10','Non invertente','Integratore']
	series=[Inv1,Inv10,NonInv] #Ogni elemento della lista è una tabella(una porzione del foglio excel). Iterando su
								# lla lista si itera sulle tabelle. Se non si fa mai esplicito riferimento al numero di righe
								# della tabella, non ci sono problemi
	#fig,((ax1,ax2),(ax3,ax4))=plt.subplots(2,2)
	fig,ax1=plt.subplots()
	fig,ax2=plt.subplots()
	fig,ax3=plt.subplots()
	fig,ax4=plt.subplots()
	figure=[ax1,ax2,ax3]

	for i in range(len(series)):
		valy,yerr=np.array(list(series[i].loc[:,'G'])),np.array(list(series[i].loc[:,'errG'])) #Assegna Y e Yerr(li prende dal)
		valx,errx=np.array(list(series[i].loc[:,'f'])),np.ones(len(yerr))  #Assegna X e Xerr
		print('FIT 1 serie ',i)
		parametri=fit(residual,valx,valy)   #Fa il fit, e restituisce i parametri del best fit
		plotta(parametri,ax4,i,valx,valy,errx,label)	#Plotta usando i parametri appena trovati sul quarto grafico(Tutte le serie assieme)
		plottadatigrezzi(figure[i],valx,valy,errx,yerr,i,label)  #Plotta i dati grezzi su uno dei primi tre grafici(Ogni serie su uno diverso)
		#print(d_model(valx,parametri)) #Stampa la derivata del modello
		yerr=np.sqrt(yerr**2+np.dot(errx,d_model(valx,parametri))**2)  #Calcola l'errore indotto usando l'errore sulle
		print('FIT 2 serie ',i)																# X, quello sulle Y e la derivata della curva di best fit
		parametri1=fit(residual,valx,valy)  #Ripete il fit col nuovo errore creando nuovi parametri(il nome è diverso dai precedenti)
		
		plotta(parametri1,ax4,i+3,valx,valy,errx,label,anche_dati_grezzi=False)  #Plotta il nuovo fit senza ripetere i dati grezzi, che sarebbero ripetuti

	fig1,ax5=plt.subplots()
	fig2,ax6=plt.subplots()

	#FIT AMPLIFICATORE INTEGRATORE

	valy,yerr=np.array(list(Int.loc[:,'G'])),np.array(list(Int.loc[:,'errG'])) #Assegna Y e Yerr(li prende dal)
	valx,errx=np.array(list(Int.loc[:,'f'])),np.ones(len(yerr))/1000  #Assegna X e Xerr
	print('FIT 1 INTEGRATORE')
	parametri=fit(residual,valx,valy)   #Fa il fit, e restituisce i parametri del best fit
	plotta(parametri,ax5,3,valx,valy,errx,label)	#Plotta usando i parametri appena trovati sul quarto grafico(Tutte le serie assieme)
	plottadatigrezzi(ax5,valx,valy,errx,yerr,3,label)  #Plotta i dati grezzi su uno dei primi tre grafici(Ogni serie su uno diverso)
	#print(d_model(valx,parametri)) #Stampa la derivata del modello
	yerr=np.sqrt(yerr**2+np.dot(errx,d_model(valx,parametri))**2)  #Calcola l'errore indotto usando l'errore sulle
	print('FIT 2 INTEGRATORE')																# X, quello sulle Y e la derivata della curva di best fit
	parametri1=fit(residual,valx,valy)  #Ripete il fit col nuovo errore creando nuovi parametri(il nome è diverso dai precedenti)
	
	plotta(parametri1,ax5,3,valx,valy,errx,label,anche_dati_grezzi=False)

	#FIT AMPLIFICATORE LOGARITMICO:

	valy=np.array(list(Logar.loc[:,'V0']))/1000 #Assegna Y e Yerr(li prende dal)
	valx=np.array(list(Logar.loc[:,'V_s']))
	yerr=valy*0.001+0.0001
	errx=np.array(list(Logar.loc[:,'errV_s']))
	#print(valy,valx,yerr,errx)
	print('FIT 1 LOGAR')
	parametri=fit1(residual1,valx,valy)   #Fa il fit, e restituisce i parametri del best fit
	plotta1(parametri,ax6,3,valx,valy,errx,label)	#Plotta usando i parametri appena trovati sul quarto grafico(Tutte le serie assieme)
	plottadatigrezzi(ax6,valx,valy,errx,yerr,3,label)  #Plotta i dati grezzi su uno dei primi tre grafici(Ogni serie su uno diverso)
	#print(d_model(valx,parametri)) #Stampa la derivata del modello
	yerr=np.sqrt(yerr**2+np.dot(errx,d_model(valx,parametri))**2)  #Calcola l'errore indotto usando l'errore sulle
	print('FIT 2 LOGAR')																# X, quello sulle Y e la derivata della curva di best fit
	parametri1=fit1(residual1,valx,valy)  #Ripete il fit col nuovo errore creando nuovi parametri(il nome è diverso dai precedenti)
	
	plotta1(parametri1,ax6,3,valx,valy,errx,label,anche_dati_grezzi=False)


	if True:	#serve solo per poter nascondere il codice mantenendolo attivo
		ax1.set_xscale('log')
		ax5.set_xscale('log')
		ax5.set_xlabel('f(kHz)')
		ax5.set_ylabel('A')
		"""
		ax1.set_xscale('log')
		ax2.set_xscale('log')
		ax3.set_xscale('log')
		ax1.set_yscale('log')
		ax2.set_yscale('log')
		ax3.set_yscale('log')
		ax1.legend(loc='best')
		ax2.legend(loc='best')
		ax3.legend(loc='best')
		ax4.legend(loc='best')
		"""
		plt.show()
	
	"""
	print(Inv1)
	yerr=np.array(list(Inv1.loc[:,'errG']))
	fit(residual,np.array(list(Inv1.loc[:,'f'])),np.array(list(Inv1.loc[:,'G'])))
	"""

def pescadati(file_name='my_file.xlsx',foglio=0,colonne='A:E',listaRighe=[0,1,2],matrice=False):
	"""
	Prende i dati dal file_name. Le colonne vanno modificate per utilizzare quelle corrette (mantenendo la forma di una stringa). 
	Restituisce una tabella(DataFrame) che contiene le colonne del file excel specificato, e che ha solo le righe della lista specificata.
	Utilizzare range(a,b) per selezionare solo le righe [a+1 ,..., b+1]
	"""
	df = pd.read_excel(file_name,sheet_name=foglio,usecols=colonne)
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
	model = a/(np.sqrt(1+(x/b)**2))  #MODIFICARE PER MODIFICARE LA FUNZIONE DI BEST FIT. RICORDARE DI MODIFICARE IL NUMERO DI PARAMETRI commentando o decommentando 
					#quelli non utilizzati: il numero di parametri indipendenti deve essere pari al numero di parametri che python usa, altrimenti
					#il fit non funziona correttamente

	return model #Restituisce un numero

def model1(pars,x):
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
	#print(type(b*x))
	#c=vals['c']
	#d=vals['d']
	model = a*np.ones(len(x))-b*np.log(x)  #MODIFICARE PER MODIFICARE LA FUNZIONE DI BEST FIT. RICORDARE DI MODIFICARE IL NUMERO DI PARAMETRI commentando o decommentando 
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
	
def d_model1(x0, parametri):
	"""
	Dato il modello(che è una funzione da R in R), trova la derivata al punto x0 di esso, per calcolare l'errore indotto.
	Il secondo input è una struttura di lmfit che viene restituita dalla funzione fit(). Quindi questa funzione può essere utilizzata
	solo dopo il primo fit per farne un secondo utilizzando l'errore indotto
	"""
	h = 1e-5

	return (model1(parametri,x0+h)-model1(parametri,x0-h))/(2*h) #Restituisce un numero, la derivata del modello a parametri fissati

def residual(pars, x,data):  #NON MODIFICARE GLI INPUT DELLA FUNZIONE, E NEMMENO IL LORO ORDINE
	"""
	Siccome noi minimizziamo somma((yi-f(xi))/xerri) bisogna definire (yi-f(xi))/xerri perché poi la somma 
	la fa il programma da solo. Il residuo è definito come (yi-f(xi))/xerri ed è un numero per un dato i.
	x e data sono quelli che sotto chiamo xdata e ydata. SONO ARRAY e il programma funziona solo se lo sono.
	"""
	#print(len(x),len(data),len(yerr))
	residuo=(model(pars,x)-data)/yerr
	return residuo

def residual1(pars, x,data):  #NON MODIFICARE GLI INPUT DELLA FUNZIONE, E NEMMENO IL LORO ORDINE
	"""
	Siccome noi minimizziamo somma((yi-f(xi))/xerri) bisogna definire (yi-f(xi))/xerri perché poi la somma 
	la fa il programma da solo. Il residuo è definito come (yi-f(xi))/xerri ed è un numero per un dato i.
	x e data sono quelli che sotto chiamo xdata e ydata. SONO ARRAY e il programma funziona solo se lo sono.
	"""
	#print(len(x),len(data),len(yerr))
	residuo=(model1(pars,x)-data)/yerr
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
	fit_params.add('a', value=4.0)#Valori iniziali di a e b(bisogna aggiungere c,d se serve)
	fit_params.add('b', value=4.0)
	#fit_params.add('c',value=50,min=0)

	out = minimize(residuo, fit_params, args=(xdata,), kws={'data': ydata},scale_covar=True)
	print(fit_report(out),"\n\n===============================\n\n")
	#stampa la tabella con tutti i valori utili(tra cui X^2)
	parametri=out.params
	return out.params

def fit1(residuo,xdata,ydata):
	"""
	Il fit vero e proprio. Di solito si chiama fit(residual) dal momento che la funzione del residuo si chiama "residual".
	Si può però chiamare fit più volte utilizzando residui diversi. xdata e ydata SONO ARRAY e lo devono rimanere! Se il programma dà errore,
	è facile che sia colpa del fatto che non si stanno usando array: a meno di casi particolari, non lavorare mai con liste. Siccome le liste
	hanno un metodo comodo che gli array non hanno, e cioè list.append(elemento) che permette di aggiungere ad una lista un elemento ulteriore,
	si consiglia di trasformare l'array in lista, chiamare append() e poi ritrasformare in array.
	"""
	fit_params = Parameters()
	fit_params.add('a', value=1)#Valori iniziali di a e b(bisogna aggiungere c,d se serve)
	fit_params.add('b', value=0.001)
	#fit_params.add('c',value=-10,max=0.15)
	#print(model(fit_params,xdata))
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


def plotta(parametri,ax,num,xdata,ydata,xerr,nomi,titolo='titolo',anche_dati_grezzi=True):
	"""
	Stampa dati e curva di best fit usando rispettivamente i parametri(ottenuti da fit()), la figura da utilizzare*, e il colore da utilizzare,
	che è un numero, da scegliere dalla lista sotto. Ad esempio se num=3, il grafico sarà rosso(le liste sono numerate da 0).
	I dati e la curva hanno lo stesso colore, rendendo facile plottare più set di dati con più curve di best fit corrispondenti, semplicemente
	variando il parametro num.
	"""
	colori=['blue','green','red','yellow','violet','grey']
	spazio=np.linspace(min(xdata),max(xdata),1000)
	if anche_dati_grezzi:
		ax.plot(xdata,ydata,'o',color=colori[num])
									#ax.legend(str('Titolo')) #Sostituire 'Titolo' con il titolo che si vuole dare al grafico. 
							  				#Utile specialmente se si lavora con più grafici diversi, per distinguerli
		ax.errorbar(xdata,ydata,yerr=yerr,ecolor='black', ls=" ",xerr=xerr)

	ax.plot(spazio,model(parametri,spazio),color=colori[num],label=nomi[num%len(nomi)])  #Modificare 'nome' per dare un nome alla singola serie
																# del grafico(infatti 'Titolo' dà il nome all'intero piano cartesiano, mentre 'nome' solo
																# ad una serie di dati)
	
def plotta1(parametri,ax,num,xdata,ydata,xerr,nomi,titolo='titolo',anche_dati_grezzi=True):
	"""
	Stampa dati e curva di best fit usando rispettivamente i parametri(ottenuti da fit()), la figura da utilizzare*, e il colore da utilizzare,
	che è un numero, da scegliere dalla lista sotto. Ad esempio se num=3, il grafico sarà rosso(le liste sono numerate da 0).
	I dati e la curva hanno lo stesso colore, rendendo facile plottare più set di dati con più curve di best fit corrispondenti, semplicemente
	variando il parametro num.
	"""
	colori=['blue','green','red','yellow','violet','grey']
	spazio=np.linspace(min(xdata),max(xdata),1000)
	if anche_dati_grezzi:
		ax.plot(xdata,ydata,'o',color=colori[num])
									#ax.legend(str('Titolo')) #Sostituire 'Titolo' con il titolo che si vuole dare al grafico. 
							  				#Utile specialmente se si lavora con più grafici diversi, per distinguerli
		ax.errorbar(xdata,ydata,yerr=yerr,ecolor='black', ls=" ",xerr=xerr)

	ax.plot(spazio,model1(parametri,spazio),color=colori[num],label=nomi[num%len(nomi)])  #Modificare 'nome' per dare un nome alla singola serie
																# del grafico(infatti 'Titolo' dà il nome all'intero piano cartesiano, mentre 'nome' solo
																# ad una serie di dati)
	


def plottadatigrezzi(ax,arr1,arr2,err1,err2,num,lista_label_1):
	"""
	Inserisce nel grafico ax solo i dati x e y con i loro errori, ma senza fit o altro. arr1 e err1 sono gli array delle x
	"""
	colori=['grey','blue','green','red','yellow','violet']
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

main()

"""
#Esempio:
def somma(numeri):
	return numeri[0]+numeri[1] #Questa funzione restituisce il rapporto tra il primo ed il secondo elemento dell'array in input
numeri=[1,2]
errori=[0.1,0.2]
print("La somma tra i numeri è ",rapporto(numeri),"+-",assegnaerrore(rapporto,numeri,errori))
"""
