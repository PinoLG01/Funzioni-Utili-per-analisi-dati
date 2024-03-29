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
from mdutils.mdutils import MdUtils

#PARTE 1: VARIABILI DEL MODULO, INCLUSI I MODELLI

Energie = {"Am": 5.486, "Np": 4.788, "Cm": 5.805}
energies_of_peaks = sorted([5.443, 5.388, 5.763, 4.771, 4.639])
Energie_mylar = pd.DataFrame({"Energie": [4.0, 4.5, 5.0, 5.5],
                              "Range": [2.90e-3, 3.44e-3, 4.04e-3, 4.67e-3]})
pi = 3.1415926535
md = MdUtils(file_name = "AnalisiOut")

fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
fig3, ax3 = plt.subplots()
fig3inv, ax3inv = plt.subplots()
fig4, ax4 = plt.subplots()
fig5, ax5 = plt.subplots()
fig6, ax6 = plt.subplots()
fig7, ax7 = plt.subplots()
fig8, ax8 = plt.subplots()

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

def Coulomb(pars, x):
    vals = pars.valuesdict()
    a = vals['a']
    b = vals['b']
    return a/(x**2) + b

modelloCoulomb = ModelloConParametri(Coulomb, ValNames = ["a", "b"])

def G_giusto(pars, x):
    vals = pars.valuesdict()
    a = vals['a']
    b = vals['b']
    primaparte = np.arcsin(1/np.sqrt(2 + 0.5*(a/x)**2))
    return b*0.5*(1-(4/pi)*primaparte)

modelloG_giusto = ModelloConParametri(G_giusto, ValNames = ["a","b"],
                                      ValStart = {"a": 40, "b": 1e11}
                                      )

#PARTE 2: CALIBRAZIONE DELL'ELETTRONICA TRAMITE SEGNALI DI AMPIEZZA NOTA

#Pesca i dati e li ripulisce
df = pescadati("../excelfinto.xlsx", colonne='A:J', listaRighe=range(11, 21))
df.dropna(axis = 1, inplace = True)
df.columns = ["ch1", "ch2", "delta", "CHN", "CNT", "errore_ch"]

md.new_header(title = "Calibrazione dell'elettronica tramite segnali di ampiezza nota", level = 1)
md.write(f"{df.to_markdown(index = False)}\n")

#Assegna ad x e y i valori che verranno usati nel fit, cioè V(CHN)
ydata = (df["ch1"].to_numpy() + df["ch2"].to_numpy())/2
yerr = df["delta"].to_numpy()/500
xdata = df["CHN"].to_numpy()
xerr = np.ones(len(xdata))
xy = XY(xdata, xerr, ydata, yerr)

#Fa il fit e plotta usando i parametri ottenuti
out = fit(modelloRetta, xy)
plotta(ax1, xy, FunzioneModello = modelloRetta, parametri = out.params)

md.write("\n Il fit ha forma V = a*Ch+b. Il grafico è AX1 \n")
for l in out.params:
    md.write(f"${l} = {out.params[l].value:.5f} \pm {out.params[l].stderr:.5f} $ \n")
md.insert_code(f"{fit_report(out)} \n")

#PARTE 3: CALIBRAZIONE DEL RIVELATORE UTILIZZANDO I PICCHI PRIMARI DELLA SORGENTE TRIPLA

#Pesca i dati e li ripulisce
df = pescadati("../excelfinto.xlsx", colonne = 'B:F', listaRighe = range(26, 29))
df.columns = ["Picco", "CNT", "MezzoPicco", "FWHM", "err"]

md.new_header(title = "Calibrazione dell'elettronica tramite segnali di ampiezza nota", level = 1)
md.write(f"{df.to_markdown(index = False)}\n")

#Assegna i dati per un fit E(CHN_picco)
ydata = np.array(sorted(list(Energie.values())))
yerr = np.ones(len(ydata))*0.001
xdata = df["Picco"].to_numpy()
xerr = np.ones(len(xdata))
dati = XY(xdata, xerr, ydata, yerr)

#Fa il fit e plotta usando i parametri ottenuti
out = fit(modelloRetta, data2D = dati)
plotta(ax2, dati, FunzioneModello=modelloRetta, parametri=out.params)

md.write("\n Il fit ha forma E_note = a*Ch_picco+b. Il grafico è AX2 \n")
for l in out.params:
    md.write(f"${l} = {out.params[l].value:.5f} \pm {out.params[l].stderr:.5f} $ \n")
md.insert_code(f"{fit_report(out)} \n")

#PARTE 4: COSTRUZIONE DELLA RETTA DI CALIBRAZIONE

#Definizione delle costanti
channels = np.array([912, 939, 1062, 1072, 1136])
                #    Np3, Np2, Am3,  Am2,  Cm2
u_channels = unp.uarray(channels, np.ones(len(channels))*1.702)

A_calibr, B_calibr = uncertainties.correlated_values(
    [out.params["a"].value, out.params["b"].value], out.covar)

#Definizione retta di calibrazione
def E(Ch):
    return A_calibr * Ch + B_calibr

#Calcolo delle energie dei picchi secondari e test z degli array
u_energies = E(u_channels)
actual_u_energies = unp.uarray(energies_of_peaks, np.ones(len(energies_of_peaks))*0.001)

Delta = actual_u_energies - u_energies

for x, y in zip(Delta, unp.nominal_values(Delta)/unp.std_devs(Delta)):
    print(x, y)

md.new_header(title = "Verifica della correttezza della retta di calibrazione tramite studio dei picchi secondari", level = 1)

df = pd.DataFrame({
    "Elemento e numero picco": ["Np III", "Np II", "Am III",  "Am II",  "Cm II"],
    "Canali picchi secondari": u_channels,
    "Energie picchi secondari": u_energies,
    "Energie picchi secondari teoriche": actual_u_energies,
    "z = (Eteo - Esperim)/sqrt(u_teo^2 + u_esp^2)": unp.nominal_values(Delta)/unp.std_devs(Delta)})
df.reset_index(drop=True, inplace=True)

md.write(f"{df.to_markdown(index = False)}\n\n")
md.write("Dove la terza riga è stata ottenuta passando la seconda dentro la retta di calibrazione \n")


#PARTE 5: STUDIO DELL'ANDAMENTO DEI CONTEGGI TOTALI IN FUNZIONE DELLA PRESSIONE

#Pesca i dati e li pulisce
df = pescadati("../AlphaInAria.xlsx", colonne='J:K', listaRighe=range(4, 19))
df = df.dropna(axis = 1)  # elimina ogni colonna che abbia almeno un valore vuoto
df.columns = ["Pressione", "Conteggi"] # associa ad ogni colonna del file excel un nome

press_tutti = df["Pressione"].to_numpy()
cont_tutti = df["Conteggi"].to_numpy()

ax3.plot(cont_tutti, press_tutti, 'o', color = "blue")
ax3.errorbar(cont_tutti, press_tutti, yerr = np.ones(len(press_tutti)), xerr = np.sqrt(cont_tutti), ecolor = 'black', ls = "")

md.new_header(title = "Conteggi totali in funzione della pressione" , level = 1)
md.write(f"{df.to_markdown(index = False)}\n\n")

df_per_estrapolazione = df.loc[df["Pressione"].isin([695.0, 702.0])]  #Tiene solo i punti corrispondenti alle
                                                                    #    pressioni della regione di linearità

#Assegna dati per un fit di due punti: Pressione(Cnt)
xdata = df_per_estrapolazione["Conteggi"].to_numpy()
ydata = df_per_estrapolazione["Pressione"].to_numpy()
xerr = unp.sqrt(df_per_estrapolazione["Conteggi"].to_numpy())
yerr = np.ones(len(xdata))
xy = XY(xdata, xerr, ydata, yerr)

md.write(f"Si esegue il fit utilizzando solo i punti: \n{df_per_estrapolazione.to_markdown(index = False)}\n\n")
#Fa un fit e plotta usando i parametri ottenuti
out = fit(modelloRetta, xy)
plotta(ax3, xy, FunzioneModello = modelloRetta, parametri = out.params)

md.write("\n Il fit ha forma Press = a*Conteggi+b. Il grafico è AX3 \n")
for l in out.params:
    md.write(f"${l} = {out.params[l].value:.5f} \pm {out.params[l].stderr:.5f} $ \n")
md.insert_code(f"{fit_report(out)} \n")


#Scrittura della retta come funzione python usando i parametri trovati
A_pressione, B_pressione = uncertainties.correlated_values(
    [out.params["a"].value, out.params["b"].value], out.covar)

def pressione(Cnt):
    return A_pressione * Cnt + B_pressione

def pressioneInv(press):
    return (press - B_pressione.n)/A_pressione.n

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

md.write(f"\n P di dimezzamento: {Pdimezz}, con conteggio {u_metà_conteggio}\n"
         f"D = {D} \t P standard = {Pstandard} \t Tstandard = {Tstandard} \t"
         f"Tlab = {Tamb}\n D di dimezzamento = {Ddimezz} \n"
         f"Rtot = Ddimezzamento + (R residuo) {RResiduo} = {RangeTot*1.205e-3}")

#PARTE 6: STUDIO DELL'ANDAMENTO IN FUNZIONE DELLO SPESSORE DI MYLAR ATTRAVERSATO
md.new_header(title = "Range nel Mylar", level = 1)

xdata = Energie_mylar["Energie"].to_numpy()
ydata = Energie_mylar["Range"].to_numpy()
xy = XY(xdata, np.ones(len(xdata))*0.0001, ydata, np.ones(len(ydata))*0.02e-3)

#Fit quadratico per estrapolare dopo 
out = fit(Pol, xy)
plotta(ax4, xy, FunzioneModello = Pol, parametri = out.params)

md.write("\n Il fit ha forma Range = a*E^2+b*E+c. Il grafico è AX4 \n")
for l in out.params:
    md.write(f"${l} = {out.params[l].value:.5f} \pm {out.params[l].stderr:.5f} $ \n")
md.insert_code(f"{fit_report(out)} \n\n")

a, b, c = uncertainties.correlated_values([out.params["a"].value, out.params["b"].value, out.params["c"].value], out.covar)

def Parabola(x, a, b, c):
    return a*(x**2)+b*x+c

def Parabola_Inv(y0, a, b, c):
    return (-b + unp.sqrt(b**2 - 4*a*(c-y0)) ) / (2*a)

def E1(E0, spess, a, b, c):
    R0 = Parabola(E0, a, b, c)
    R1 = R0 - spess
    E1 = Parabola_Inv(R1, a, b, c)
    return E1

df = pescadati("../RateDistanza e LossMylar.xlsx", foglio = 1, colonne = "B:D", listaRighe = range(6,11))
df.columns = ["spess", "chn", "u_chn"]
md.new_paragraph(f"{df.to_markdown(index = False)}\n\n")
         

spess = df["spess"].to_numpy()
u_spess = np.array([0.05, 0.07, 0.2, 0.4, 0.5])
chn = df["chn"].to_numpy()
u_chn = df["u_chn"].to_numpy()/1.175

spess, chn = unp.uarray(spess, u_spess)*1e-4*1.39, unp.uarray(chn, u_chn)

E1_misurate = E(chn)

E1_attese = [E1(5.486, x, a, b, c) for x in spess]
print("ENERGIE")
for x, y in zip(E1_misurate, E1_attese):
    print(x, y, sep = "\t")

print([(x-y).n/(x-y).s for x,y in zip(E1_misurate, E1_attese)])

df = pd.DataFrame({
    "Energie misurate": E1_misurate,
    "Energie attese": E1_attese
})
md.write(f"{df.to_markdown(index = False)}\n\n")

# PARTE 7: RATE IN FUNZIONE DELLA DISTANZA

df = pescadati("../RateDistanza e LossMylar.xlsx", foglio = 0, colonne = "A:E", listaRighe = range(1,9))
df.columns = ["d", "u_d", "cnt", "t", "u_t"] # Con cnt si intendono tutte le alfa arrivate
d, u_d = df["d"].to_numpy(), df["u_d"].to_numpy()
cnt, t, u_t = df["cnt"].to_numpy(), df["t"].to_numpy(), df["u_t"].to_numpy()

md.new_header(title = "Rate in funzione della distanza", level = 1)
md.write(f"{df.to_markdown(index = False)}\n\n")

d_soffitto_pcb = uncertainties.ufloat(5, 1)
d_pcb_rivelatore = uncertainties.ufloat(6.3, 0.1)
d_pavimento_sorg = uncertainties.ufloat(4.5, 0.1)

d = unp.uarray(d, u_d) - d_soffitto_pcb - d_pcb_rivelatore - d_pavimento_sorg
t = unp.uarray(t, u_t)
cnt = unp.uarray(cnt, np.sqrt(cnt))

rate = cnt / t

df = pd.DataFrame({
    "Distanze": d,
    "Tempi": t,
    "Counts": cnt,
    "Rate": rate
})
md.write(f"{df.to_markdown(index = False)}\n\n")

# Fit 1/r^2
dati = XY(x = d, y = rate, uarr = True)

out = fit(modelloCoulomb, data2D = dati)

md.write("\n Il fit ha forma Range = a/x^2 + b. Il grafico è AX5 \n")
for l in out.params:
    md.write(f"${l} = {out.params[l].value:.5f} \pm {out.params[l].stderr:.5f} $ \n")
md.insert_code(f"{fit_report(out)} \n")

plotta(ax5, data2D = dati, FunzioneModello = modelloCoulomb, parametri = out.params)

# Fit corretto

out = fit(modelloG_giusto, data2D = dati)

md.write("\n Il fit ha forma Range = b*0.5*(1-(4/pi)*np.arcsin(1/np.sqrt(2 + 0.5*(a/x)**2))). Il grafico è AX6 \n")
for l in out.params:
    md.write(f"${l} = {out.params[l].value:.5f} \pm {out.params[l].stderr:.5f} $ \n")
md.insert_code(f"{fit_report(out)} \n")

plotta(ax6, data2D = dati, FunzioneModello = modelloG_giusto, parametri = out.params)

def G_giusto_c(pars, x):
    vals = pars.valuesdict()
    a = vals['a']
    b = vals['b']
    c = vals['c']
    primaparte = np.arcsin(1/np.sqrt(2 + 0.5*(a/(x-c))**2))
    return b*0.5*(1-(4/pi)*primaparte)

modelloG_giusto_c = ModelloConParametri(G_giusto_c, ValNames = ["a","b","c"],
                                      ValStart = {"a": 40, "b": 1e11}
                                      )


out = fit(modelloG_giusto_c, data2D = dati)
plotta(ax7, data2D = dati, FunzioneModello = modelloG_giusto_c, parametri = out.params)

md.write("\n Il fit ha forma Range = b*0.5*(1-(4/pi)*np.arcsin(1/np.sqrt(2 + 0.5*(a/(x-c))**2))). Il grafico è AX7 \n")
for l in out.params:
    md.write(f"${l} = {out.params[l].value:.5f} \pm {out.params[l].stderr:.5f} $ \n")
md.insert_code(f"{fit_report(out)} \n")

uy2, y2 = [], []
for x, y in zip(dati.x, dati.y):
    uy2.append(np.sqrt(y.std_dev**2 + (x.std_dev * derivata(modelloG_giusto_c, out.params, x.n))**2))
    y2.append(y.nominal_value)
    print(derivata(modelloG_giusto_c, out.params, x.n))
print("FIT FINALE")
print(dati.y, y2, uy2)
dati.y = unp.uarray(y2, uy2)
out = fit(modelloG_giusto, data2D = dati)
plotta(ax8, data2D = dati, FunzioneModello = modelloG_giusto, parametri = out.params)
print(df_per_estrapolazione)
ax3inv.plot(press_tutti, cont_tutti, 'o', color = "blue")
ax3inv.errorbar(press_tutti, cont_tutti, yerr = np.sqrt(cont_tutti), xerr = np.ones(len(press_tutti)), ecolor = 'black', ls = "")
ax3inv.plot(df_per_estrapolazione["Pressione"].to_numpy(), pressioneInv(df_per_estrapolazione["Pressione"]))

ax1.set_xlabel('Canale')
ax1.set_ylabel('Tensione [V]')
ax1.title.set_text("Linearità della tensione")

ax2.set_xlabel('Canale')
ax2.set_ylabel('Energia [MeV]')
ax2.title.set_text("Linearità dell'energia")

ax3.set_xlabel('Conteggi')
ax3.set_ylabel('Pressione [mbar]')
ax3.title.set_text("Estrapolazione della pressione di dimezzamento")

ax3inv.set_xlabel('Pressione [mbar]')
ax3inv.set_ylabel('Conteggi')
ax3inv.title.set_text("Estrapolazione della pressione di dimezzamento")

ax4.set_xlabel('Energia [MeV]')
ax4.set_ylabel('Range equivalente [g/cm^2]')
ax4.title.set_text("Fit quadratico sui dati tabulati del Mylar")

ax5.set_xlabel('Distanza tra la sorgente e il rivelatore [mm]')
ax5.set_ylabel('Rate [cnt/s]')
ax5.title.set_text("Rate in funzione della distanza (1/r^2)")

ax6.set_xlabel('Distanza tra la sorgente e il rivelatore [mm]')
ax6.set_ylabel('Rate [cnt/s]')
ax6.title.set_text("Rate in funzione della distanza (arcsin)")

ax7.set_xlabel('Distanza tra la sorgente e il rivelatore [mm]')
ax7.set_ylabel('Rate [cnt/s]')
ax7.title.set_text("Rate in funzione della distanza (arcsin traslato)")

ax8.set_xlabel('Distanza tra la sorgente e il rivelatore [mm]')
ax8.set_ylabel('Rate')
ax8.title.set_text("Rate in funzione della distanza (arcsin con errori indotti)")

plt.show()
for i, fig in enumerate([fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8]):
    fig.savefig(f"../Figure/ax{i+1}", bbox_inches="tight")

fig3inv.savefig(f"../Figure/ax3inv", bbox_inches="tight")

md.create_md_file()
