



# Calibrazione dell'elettronica tramite segnali di ampiezza nota
|   ch1 |   ch2 |   delta |   CHN |   CNT |   errore_ch |
|------:|------:|--------:|------:|------:|------------:|
| 0.968 | 0.992 |      24 |   195 |  1237 |           1 |
| 1.98  | 2.06  |      80 |   401 |   996 |           1 |
| 2.92  | 3.02  |     100 |   596 |   651 |           1 |
| 3.92  | 4.04  |     120 |   793 |   768 |           1 |
| 4.84  | 5     |     160 |   981 |   980 |           1 |
| 5.88  | 6.04  |     160 |  1191 |   896 |           1 |
| 6.92  | 7.08  |     160 |  1403 |   755 |           1 |
| 7.76  | 8.08  |     320 |  1587 |   607 |           1 |
| 8.8   | 9.12  |     320 |  1807 |   522 |           1 |
| 9.2   | 9.52  |     320 |  1889 |   839 |           1 |

 Il fit ha forma V = a*Ch+b. Il grafico è AX1 
$a = 0.00498 \pm 0.00001 $ 
$b = 0.00953 \pm 0.00429 $ 


```
[[Fit Statistics]]
    # fitting method   = leastsq
    # function evals   = 7
    # data points      = 10
    # variables        = 2
    chi-square         = 0.03872345
    reduced chi-square = 0.00484043
    Akaike info crit   = -51.5389496
    Bayesian info crit = -50.9337795
[[Variables]]
    a:  0.00498374 +/- 9.7152e-06 (0.19%) (init = 1)
    b:  0.00953102 +/- 0.00429480 (45.06%) (init = 1)
[[Correlations]] (unreported correlations are < 0.100)
    C(a, b) = -0.724 

```
# Calibrazione dell'elettronica tramite segnali di ampiezza nota
|   Picco |   CNT |   MezzoPicco |   FWHM |   err |
|--------:|------:|-------------:|-------:|------:|
|     942 |   276 |          944 |      4 | 1.702 |
|    1081 |   357 |         1083 |      4 | 1.702 |
|    1144 |   312 |         1146 |      4 | 1.702 |

 Il fit ha forma E_note = a*Ch_picco+b. Il grafico è AX2 
$a = 0.00503 \pm 0.00001 $ 
$b = 0.04703 \pm 0.01050 $ 


```
[[Fit Statistics]]
    # fitting method   = leastsq
    # function evals   = 7
    # data points      = 3
    # variables        = 2
    chi-square         = 2.10142915
    reduced chi-square = 2.10142915
    Akaike info crit   = 2.93201612
    Bayesian info crit = 1.12924070
[[Variables]]
    a:  0.00503250 +/- 9.9177e-06 (0.20%) (init = 1)
    b:  0.04702507 +/- 0.01050314 (22.34%) (init = 1)
[[Correlations]] (unreported correlations are < 0.100)
    C(a, b) = -0.997 

```
# Verifica della correttezza della retta di calibrazione tramite studio dei picchi secondari
| Elemento e numero picco   | Canali picchi secondari   | Energie picchi secondari   | Energie picchi secondari teoriche   |   z = (Eteo - Esperim)/sqrt(u_teo^2 + u_esp^2) |
|:--------------------------|:--------------------------|:---------------------------|:------------------------------------|-----------------------------------------------:|
| Np III                    | 912.0+/-1.7               | 4.637+/-0.009              | 4.6390+/-0.0010                     |                                       0.266014 |
| Np II                     | 939.0+/-1.7               | 4.773+/-0.009              | 4.7710+/-0.0010                     |                                      -0.176385 |
| Am III                    | 1062.0+/-1.7              | 5.392+/-0.009              | 5.3880+/-0.0010                     |                                      -0.408479 |
| Am II                     | 1072.0+/-1.7              | 5.442+/-0.009              | 5.4430+/-0.0010                     |                                       0.131077 |
| Cm II                     | 1136.0+/-1.7              | 5.764+/-0.009              | 5.7630+/-0.0010                     |                                      -0.10851  |

Dove la terza riga è stata ottenuta passando la seconda dentro la retta di calibrazione 

# Conteggi totali in funzione della pressione
|   Pressione |   Conteggi |
|------------:|-----------:|
|           0 |       4967 |
|         205 |       4884 |
|         403 |       4782 |
|         598 |       4741 |
|         612 |       4626 |
|         621 |       4538 |
|         631 |       4583 |
|         641 |       4437 |
|         650 |       4447 |
|         670 |       4310 |
|         685 |       3922 |
|         690 |       3409 |
|         695 |       2790 |
|         702 |       1105 |
|         750 |         15 |

Si esegue il fit utilizzando solo i punti: 
|   Pressione |   Conteggi |
|------------:|-----------:|
|         695 |       2790 |
|         702 |       1105 |


 Il fit ha forma Press = a*Conteggi+b. Il grafico è AX3 
$a = -0.00415 \pm 0.00000 $ 
$b = 706.59050 \pm 0.00000 $ 


```
[[Fit Statistics]]
    # fitting method   = leastsq
    # function evals   = 7
    # data points      = 2
    # variables        = 2
    chi-square         = 2.000e-250
    reduced chi-square = 0.00000000
    Akaike info crit   = -1147.29255
    Bayesian info crit = -1149.90625
##  Warning: uncertainties could not be estimated:
[[Variables]]
    a: -0.00415430 +/- 0.00000000 (0.00%) (init = 1)
    b:  706.590504 +/- 0.00000000 (0.00%) (init = 1) 

```
 P di dimezzamento: 696.27+/-0.21, con conteggio (2.48+/-0.05)e+03
D = 61.2+/-1.0 	 P standard = 1013.2+/-0 	 Tstandard = 293.15+/-0 	Tlab = 296.9+/-1.0
 D di dimezzamento = 41.5+/-0.7 
Rtot = Ddimezzamento + (R residuo) 1.39+/-0.10 = 0.0517+/-0.0008
# Range nel Mylar

 Il fit ha forma Range = a*E^2+b*E+c. Il grafico è AX4 
$a = 0.00009 \pm 0.00001 $ 
$b = 0.00033 \pm 0.00013 $ 
$c = 0.00015 \pm 0.00030 $ 


```
[[Fit Statistics]]
    # fitting method   = leastsq
    # function evals   = 9
    # data points      = 4
    # variables        = 3
    chi-square         = 45.0000000
    reduced chi-square = 45.0000000
    Akaike info crit   = 15.6814725
    Bayesian info crit = 13.8403556
[[Variables]]
    a:  9.0000e-05 +/- 1.3416e-05 (14.91%) (init = 1)
    b:  3.2700e-04 +/- 1.2760e-04 (39.02%) (init = 1)
    c:  1.5050e-04 +/- 2.9989e-04 (199.27%) (init = 1)
[[Correlations]] (unreported correlations are < 0.100)
    C(a, b) = -0.999
    C(b, c) = -0.999
    C(a, c) = 0.995 


```

|   spess |   chn |   u_chn |
|--------:|------:|--------:|
|     0.9 |  1061 |       2 |
|     1.4 |  1049 |       3 |
|     2.8 |  1016 |       3 |
|     4.2 |   981 |       8 |
|     5.1 |   963 |       8 |

| Energie misurate   | Energie attese   |
|:-------------------|:-----------------|
| 5.387+/-0.009      | 5.390+/-0.006    |
| 5.326+/-0.013      | 5.336+/-0.008    |
| 5.160+/-0.013      | 5.184+/-0.022    |
| 4.984+/-0.034      | 5.03+/-0.05      |
| 4.893+/-0.034      | 4.93+/-0.06      |


# Rate in funzione della distanza
|     d |   u_d |   cnt |      t |   u_t |
|------:|------:|------:|-------:|------:|
| 49.8  |  0.05 | 43746 | 200    |  0.01 |
| 46.65 |  0.05 | 43724 | 150.43 |  0.01 |
| 39    |  0.05 | 48153 | 101.93 |  0.01 |
| 31.45 |  0.05 | 45742 |  51.29 |  0.01 |
| 27.3  |  0.05 | 47219 |  35.76 |  0.01 |
| 23.65 |  0.05 | 48594 |  22.37 |  0.01 |
| 19.75 |  0.05 | 47525 |  11.91 |  0.01 |
| 16.15 |  0.05 | 43125 |   4.95 |  0.01 |

| Distanze   | Tempi           | Counts              | Rate              |
|:-----------|:----------------|:--------------------|:------------------|
| 34.0+/-1.0 | 200.000+/-0.010 | (4.375+/-0.021)e+04 | 218.7+/-1.0       |
| 30.9+/-1.0 | 150.430+/-0.010 | (4.372+/-0.021)e+04 | 290.7+/-1.4       |
| 23.2+/-1.0 | 101.930+/-0.010 | (4.815+/-0.022)e+04 | 472.4+/-2.2       |
| 15.6+/-1.0 | 51.290+/-0.010  | (4.574+/-0.021)e+04 | 892+/-4           |
| 11.5+/-1.0 | 35.760+/-0.010  | (4.722+/-0.022)e+04 | 1320+/-6          |
| 7.8+/-1.0  | 22.370+/-0.010  | (4.859+/-0.022)e+04 | 2172+/-10         |
| 3.9+/-1.0  | 11.910+/-0.010  | (4.753+/-0.022)e+04 | 3990+/-19         |
| 0.3+/-1.0  | 4.950+/-0.010   | (4.312+/-0.021)e+04 | (8.71+/-0.05)e+03 |


 Il fit ha forma Range = a/x^2 + b. Il grafico è AX5 
$a = 1066.11126 \pm 825.90998 $ 
$b = 326.63097 \pm 112.20730 $ 


```
[[Fit Statistics]]
    # fitting method   = leastsq
    # function evals   = 12
    # data points      = 8
    # variables        = 2
    chi-square         = 131750.160
    reduced chi-square = 21958.3601
    Akaike info crit   = 81.6737691
    Bayesian info crit = 81.8326522
[[Variables]]
    a:  1066.11126 +/- 825.909981 (77.47%) (init = 1)
    b:  326.630972 +/- 112.207304 (34.35%) (init = 1) 

```
 Il fit ha forma Range = b*0.5*(1-(4/pi)*np.arcsin(1/np.sqrt(2 + 0.5*(a/x)**2))). Il grafico è AX6 
$a = 14.80192 \pm 0.91625 $ 
$b = 15108.72430 \pm 1367.26672 $ 


```
[[Fit Statistics]]
    # fitting method   = leastsq
    # function evals   = 24
    # data points      = 8
    # variables        = 2
    chi-square         = 2579.64832
    reduced chi-square = 429.941387
    Akaike info crit   = 50.2077345
    Bayesian info crit = 50.3666176
[[Variables]]
    a:  14.8019238 +/- 0.91624927 (6.19%) (init = 40)
    b:  15108.7243 +/- 1367.26672 (9.05%) (init = 1e+11)
[[Correlations]] (unreported correlations are < 0.100)
    C(a, b) = -0.923 

```
 Il fit ha forma Range = b*0.5*(1-(4/pi)*np.arcsin(1/np.sqrt(2 + 0.5*(a/(x-c))**2))). Il grafico non c'è 
$a = 10.54729 \pm 1.48403 $ 
$b = 38281.21989 \pm 12748.57322 $ 
$c = -3.54370 \pm 0.84905 $ 


```
[[Fit Statistics]]
    # fitting method   = leastsq
    # function evals   = 68
    # data points      = 8
    # variables        = 3
    chi-square         = 516.697013
    reduced chi-square = 103.339403
    Akaike info crit   = 39.3441209
    Bayesian info crit = 39.5824455
[[Variables]]
    a:  10.5472857 +/- 1.48402813 (14.07%) (init = 40)
    b:  38281.2199 +/- 12748.5732 (33.30%) (init = 1e+11)
    c: -3.54369879 +/- 0.84904713 (23.96%) (init = 1)
[[Correlations]] (unreported correlations are < 0.100)
    C(a, b) = -0.993
    C(b, c) = -0.965
    C(a, c) = 0.933 

```