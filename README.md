# Progetto Machine Learning: Riconoscimento Pulsanti

## Team di sviluppo
[Andrea Busacca](https://github.com/busaccandrea)  
[Andrea Sequenzia](https://github.com/Boros93)  
[Valerio Tosto](https://github.com/ValerioTosto)  

----
## Indice
**1. Introduzione**  
**2. Dataset**  
**3. Metodo**  
Squeezenet  
VGG16  
Implementazione  
**4. Esperimenti e Risultati**  
**5. Conclusioni**  
**6. Citazioni**  

----
## 1. Introduzione

Il task affrontato nel periodo di lavoro consiste nella costruzione di un algoritmo che, una volta acquisita tramite wearable (ad esempio Microsoft HoloLens) una foto di una pulsantiera, sia in grado di capire se è stato premuto un tasto e se sì quale di essi è stato premuto. Il problema è quindi riducibile ad una classificazione multiclasse a 4 classi.

La pulsantiera è stata costruita ad hoc con un foglio di carta e tre tappi di bottiglia: uno rosso, uno verde e uno blu, posizionati in quest&#39;ordine da sinistra a destra, come riporta lo schema seguente.

![Alt text](doc/img/pulsantiera.png?raw=true "Pulsantiera")

Quando l&#39;utente preme il tasto rosso, l&#39;algoritmo deve dare in output **&quot;Classe A&quot;​**, quando viene premuto il tasto verde l&#39;output dev&#39;essere **&quot;Classe B&quot;​** ,se viene premuto il tasto blu dev&#39;essere corrisposto l&#39;output **&quot;Classe C&quot;** e infine **&quot;Classe NonPremuto&quot;** se non viene rilevato alcun tasto premuto.

Per risolvere il task sono state utilizzate due architetture diverse: **SqueezeNet** ​ e **VGG16​**. La prima è un&#39;architettura leggera con tempi di training molto ristretti e la seconda ha prestazioni migliori ma più pesante da allenare.

----
## 2. Dataset

Il dataset è stato costruito estraendo tutti i frame di video a 30FPS 1080p, acquisiti tramite fotocamera di uno smartphone XPERIA XZ1.

I video sono stati registrati in modo da facilitare l&#39;etichettatura (eseguita con ELAN): in ognuno di essi viene premuto un solo tasto e alla camera viene applicato del movimento e della rotazione variabili lungo gli assi X, Y e Z. In questo modo, nel software di etichettatura, una volta individuato il momento in cui il dito viene a contatto col tasto, è possibile etichettare manualmente come una sola classe tutti i frame da quell&#39;istante fino alla fine del video. Questo processo genera un file di testo per ogni video in cui viene riportato il range di tempo appena citato. Con lo script &quot;​ **elan\_extraction.py​**&quot; vengono letti i file di testo e vengono estratti i frame da ogni video (che in totale sono circa 11000) rinominandoli con l&#39;espressione regolare &quot;C[0 | 1 | 2 | 3]\_ **i​** ​.jpg&quot;, con *i​​=0, …, n* ​, infine smistandoli in 4 cartelle chiamate &quot;.\\0\\&quot;, &quot;.\\1\\&quot;, &quot;.\\2\\&quot;, &quot;.\\3\\&quot;.

Inoltre è stato creato lo script &quot;​**csv\_creator.py​**&quot; che genera i file **&quot;.csv&quot;** per suddividere a livello logico il dataset in **training set** (60%), **​validation​ set​** ​(10%) ​e **​test set** (30%).

Con lo script &quot;​**preprocessing.py​**&quot; vengono riscalate le immagini, convertendole ad una dimensione di 224x224.

Vengono riportati nella **Tabella** ​ **1** quattro samples casuali, uno per ogni classe.

|   |   |
| --- | --- |
| ![img](doc/img/C0.jpg?raw=true) | ![img](doc/img/C1.jpg?raw=true) |
| Esempio di classe NonPremuto | Esempio di classe A |
| ![img](doc/img/C2.jpg?raw=true) | ![img](doc/img/C3.jpg?raw=true) |
| Esempio di classe B | Esempio di classe C |

**Tabella 1​**

----
## 3. Metodo

Le due architetture utilizzate per risolvere il task sono, come accennato in precedenza, Squeezenet (cit. ​1) e VGG16​ (cit. ​2).​ La scelta è ricaduta su queste per poter eseguire un confronto di prestazioni su due reti con prestazioni e costi computazionali diversi.

**Squeezenet**

Questa architettura nasce dall&#39;esigenza di avere delle CNN più leggere, a cui serve meno banda, meno memoria e le inferenze possono essere eseguite su CPU. Per ottenere questa architettura sono state seguite tre strategie:

1. Rimpiazzare filtri 3x3 con filtri 1x1. Così facendo si ottiene un numero di pesi fino a 9 volte inferiore (​**Squeeze Layer​**);

1. Diminuire il numero di canali di input ai filtri 3x3. Vengono diminuiti i canali in input ai filtri 3x3 usando il cosiddetto  **Fire Module​**;

1. Applicare il downsampling in &quot;fondo&quot; alla rete. È stato dimostrato da &quot;​**K. He e H. Sun​**&quot; che ritardando il downsampling in fondo alla rete è possibile ottenereun&#39;accuracy maggiore.

Il &quot;​**Fire Module​**&quot; è un nuovo building block composto da due parti. La prima parte, denominata &quot;​**Squeeze Layer​**&quot;, risulta essere l&#39;applicazione della strategia 1. Infatti è un layer convolutivo in cui sono presenti solo filtri 1x1 (li chiamiamo s​1x1 ).

La seconda parte è chiamata &quot;​ **Expand Layer​**&quot; e applica un mix di filtri 1x1 (e1X1) e 3x3 (e3X3). In un Fire module #s1X1 è minore di #e1X1 + #e​3x3​.

In questo modo, non solo si riduce il numero di parametri della rete ma si riduce anche il numero di canali in input ai filtri convolutivi 3x3.

![Alt text](doc/img/squeeznet_fire_module.png?raw=true "Fire Module")

La Squeezenet ha un&#39;architettura che si sviluppa come segue: convolution Layer (conv1), seguito da 8 Fire Modules (fire2-9) e un convolution Layer finale (conv10). I layer di pooling (eseguono un max-pool con stride=2) vengono posizionati dopo i layer conv1, fire4, fire8 e conv10. Questo posizionamento del pooling è l&#39;applicazione della strategia 3. La figura sottostante descrive l&#39;architettura di questa NN.

![Alt text](doc/img/squeeznet.png?raw=true "Squeeznet")

**VGG16**

È stata scelta questa CNN per le sue alte prestazioni, famosa per aver risolto la **ImageNet Large-ScaleVisual Recognition Challenge (ILSVRC)​**.

L&#39;input di questa CNN è un set di immagini RGB riscalate a 224 x 224 e l&#39;unica forma di preprocessing è sottrarre ad ogni pixel il valore della media calcolata sul training set. L&#39;immagine passa attraverso uno stack di 13 filtri convolutivi con un receptive field 3 x 3, stride a 1 e padding a 1. I pooling layer eseguono un max-pooling 2 x 2 con stride a 2.

Lo stack di layer convolutivi è seguito da tre FC-layer. I primi due hanno 4096 canali ciascuno e il terzo fa classificazione multiclasse a 1000 classi (di default). L&#39;ultimo layer è il softmax layer.

Di seguito è riportato uno schema che descrive l&#39;architettura appena discussa.

![Alt text](doc/img/vgg16.png?raw=true "VGG16")

**Implementazione**

La strategia attuata è quella di utilizzare le reti pre-allenate su ImageNet, il dataset utilizzato come base per la ImageNet Large-ScaleVisual Recognition Challenge (ILSVRC), anziché allenarle da zero. Questo ha portato ad un significativo risparmio di tempo nella fase di training.

La struttura dei FC-layers della Squeezenet è la seguente:

![Alt text](doc/img/squeeznet_fc.png?raw=true "Squeeznet FC-layers")

Essendo il task una classificazione multiclasse a 4 classi, con la seguente riga di codice è stato modificato l&#39;output dell&#39;ultimo layer convolutivo settandolo a 4.

> model.classifier[1] = nn.Conv2d(512, 4, kernel_size=(1, 1), stride=(1, 1))

Invece, la struttura dei FC-layers della VGG16 è descritta nella figura sottostante:

![Alt text](doc/img/vgg16_fc.png?raw=true "VGG16 FC-layers")

Per poter utilizzare questa rete è stata eseguita una modifica simile a quella fatta alla Squeezenet, è stato modificato l&#39;output dell&#39;ultimo layer FC con la seguente riga:

> model.classifier[6] = nn.Linear(4096, num_class)

La fase di training sia di Squeezenet che di VGG16 è durato circa 50 epoche, suddiviso in più riprese esportando accuratamente il checkpoint ogni volta, per poi riprendere dall&#39;ultimo checkpoint salvato.

Per supportare il nostro lavoro è stata sviluppata una GUI, di cui descriviamo la struttura.

![Alt text](doc/img/gui.png?raw=true "GUI")

1. Il flag &quot;​**Developer mode​**&quot; ha la funzione di far scegliere se entrare in modalità &quot;​**Demo​**&quot; oppure in modalità &quot;​**Developer​**&quot;.

- In **Demo Mode​** vi è:

2. un menu a tendina che permette di scegliere una delle architetture proposte;

3. un tasto &quot;​**Select Image​**&quot; col quale selezionare un&#39;immagine da valutare;

4. l&#39;anteprima dell&#39;immagine selezionata;

5. un tasto &quot;​**Submit​**&quot; che dà in pasto l&#39;immagine alla rete scelta;

6. una sezione in cui viene riportato l&#39;output della rete.

- In **Developer Mode**, oltre alle sezioni presenti in Demo Mode, vengono sbloccate alcune sezioni aggiuntive:

7. un flag per abilitare o meno il Data Augmentation

8. una casella di testo per specificare il numero di epoche di training desiderate;

9. il tasto &quot;​**Train​**&quot; per lanciare l&#39;allenamento e la sezione &quot;​**Accuracy​**&quot; in cui, una volta terminata l&#39;esecuzione, viene riportato la percentuale di accuracy dell&#39;allenamento;

10. un flag &quot;**Pretrained​**&quot; che permette di scegliere la situazione iniziale della rete: trainata su ImageNet oppure usare un checkpoint di training se presente.

----
## 4. Esperimenti e Risultati

Il training è stato eseguito su un PC avente una GPU Nvidia RTX 2060, RAM da 16 GB e CPU Intel i7-8750H.

Sono state fatte varie prove di training e come training definitivo le reti Squeezenet e VGG16 sono state allenate per circa 100 epoche senza data augmentation. Sono stati salvati solo i grafici che mostrano l&#39;andamento di accuracy e loss durante il training della Squeezenet, in quanto il training della VGG16 è stato eseguito in più sessioni. In **Figura 1** viene appunto mostrato l&#39;andamento del training della Squeezenet: a sinistraviene riportata la curva della funzione di loss e a destra la curva dell&#39;accuracy. Da questi due grafici si vede come il modello converge dopo l&#39;ottantesima epoca.

Come ci si aspettava, la Squeezenet ha impiegato un tempo significativamente minore a completare un&#39;epoca. Infatti ha impiegato circa 30&#39;&#39; e la VGG16 circa 3&#39;.

![img](doc/img/loss.png?raw=true) ![img](doc/img/accuracy.png?raw=true)  
**Figura 1**

In fase di test, i due modelli raggiungono un&#39;accuracy del 99% sul test set. Per verificare se effettivamente il modello realizza il task, sono state create delle immagini esterne al test set simulando l&#39;acquisizione da un wearable (&quot;WearableImageSet&quot;) che vengono mostrate in **Tabella 2​**.

Sia la Squeezenet che la VGG raggiungono un&#39;accuracy del 100% su questo insieme di immagini.

|   |   |   |
| --- | --- | --- |
| ![img](doc/WearableImageTest/1.jpg?raw=true) | ![img](doc/WearableImageTest/2.jpg?raw=true) | ![img](doc/WearableImageTest/3.jpg?raw=true) |
| Foto 1 | Foto 2 | Foto 3 |
| ![img](doc/WearableImageTest/4.jpg?raw=true) | ![img](doc/WearableImageTest/5.jpg?raw=true) | ![img](doc/WearableImageTest/6.jpg?raw=true) |
| Foto 4 | Foto 5 | Foto 6 |
| ![img](doc/WearableImageTest/7.jpg?raw=true) | ![img](doc/WearableImageTest/8.jpg?raw=true) | ![img](doc/WearableImageTest/9.jpg?raw=true) |
| Foto 7 | Foto 8 | Foto 9 |
**Tabella 2**

|   | Foto 1 S | Foto 1 V | Foto 2 S | Foto 2 V | Foto 3 S | Foto 3 V | Foto 4 S | Foto 4 V | Foto 5 S | Foto 5 V | Foto 6 S | Foto 6 V | Foto 7 S | Foto 7 V | Foto 8 S | Foto 8 V | Foto 9 S | Foto 9 V |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Dataset 3K| ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ |
| Dataset 2K| ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ |
| Dataset 1K|   | ✔ | ✔ | ✔ |   | ✔ | ✔ |   | ✔ | ✔ |   | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ |
**Tabella 3** - Risultati su Wearable set, per ogni foto il risultato di Squeezenet (S) e VGG16 (V)

È stato condotto un esperimento per verificare dove i modelli falliscono, creando delle immagini ad hoc contenute nella cartella &quot;WeirdImageSet&quot; e mostrate in **Tabella 2​**. Alcune di queste immagini sono state create usando tipi di inquadrature molto differenti rispetto a quelle usate per creare il dataset (inquadrature molto rare in un&#39;acquisizione da un wearable) o situazioni particolari e altre immagini sono state create usando un altro tipo di pulsantiera. La Squeezenet sbaglia a predire la classe per le foto 1, 2, 5, 7, 8, 9. Da questi risultati si nota che essa soffre di più il cambio di pulsantiera e ha qualche problema nel riconoscere spazialmente la posizione del dito. La VGG16, invece, non riesce a classificare in maniera corretta le foto 2, 3, 4, 5, 6, 7. Nonostante la maggior complessità della VGG16, l&#39;accuracy in entrambi i modelli è del 33% ma alcune foto, come quelle della pulsantiera diversa, le ha classificate correttamente. Per cercare di risolvere questi problemi, sono stati condotti alcuni esperimenti variando la grandezza del dataset e provando ad utilizzare data augmentation, ma quest&#39;ultimo ha portato solo a peggiorare i risultati.

Il primo esperimento è stato quello di diminuire il numero di immagini per classe a 1000. Le due reti non ottengono dei buoni risultati sul &quot;WearableImageSet&quot; e non ottengono miglioramenti sul &quot;WeirdImageSet&quot;.

In maniera simile al primo esperimento, sono state abbassate il numero di immagini per classe a 2000. Questo ha portato dei risultati migliori per quanto riguarda le immagini in cui il modello originale aveva delle difficoltà. Con questo nuovo dataset, la Squeezenet sbaglia la predizione sulle foto: 1, 4, 5, 8, portando quindi ad un miglioramento dell&#39;accuracy sul &quot;WeirdImageSet&quot;.

Con la VGG16, l&#39;accuracy sul &quot;WeirdImageSet&quot; viene ancora aumentata in quanto le uniche foto in cui sbaglia sono: 1, 7 e 8. Entrambi i modelli hanno indovinato tutte le foto del&quot;WearableImageSet&quot; ma hanno un&#39;accuracy leggermente minore rispetto al modello originale. Infatti la Squeezenet raggiunge il 98,2% e la VGG16 il 98%.

|   |   |   |
| --- | --- | --- |
| ![img](doc/WeirdImageTest/1.jpg?raw=true) | ![img](doc/WeirdImageTest/2.jpg?raw=true) | ![img](doc/WeirdImageTest/3.jpg?raw=true) |
| Foto 1 | Foto 2 | Foto 3 |
| ![img](doc/WeirdImageTest/4.jpg?raw=true) | ![img](doc/WeirdImageTest/5.jpg?raw=true) | ![img](doc/WeirdImageTest/6.jpg?raw=true) |
| Foto 4 | Foto 5 | Foto 6 |
| ![img](doc/WeirdImageTest/7.jpg?raw=true) | ![img](doc/WeirdImageTest/8.jpg?raw=true) | ![img](doc/WeirdImageTest/9.jpg?raw=true) |
| Foto 7 | Foto 8 | Foto 9 |
**Tabella 5**

|   | Foto 1 S | Foto 1 V | Foto 2 S | Foto 2 V | Foto 3 S | Foto 3 V | Foto 4 S | Foto 4 V | Foto 5 S | Foto 5 V | Foto 6 S | Foto 6 V | Foto 7 S | Foto 7 V | Foto 8 S | Foto 8 V | Foto 9 S | Foto 9 V |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Dataset 3K|   | ✔ |   |   | ✔ |   | ✔ |   |   |   | ✔ |   |   |   |   | ✔ |   | ✔ |
| Dataset 2K|   |   | ✔ | ✔  | ✔ | ✔ |   | ✔ |   | ✔ | ✔ | ✔ | ✔ |   |   |   | ✔ | ✔ |
| Dataset 1K|   |   |   | ✔  | ✔ |   | ✔ |   |   |   | ✔ |   |   |   |   | ✔ |   | ✔ |
**Tabella 6​** - Risultati su Weird set, per ogni foto il risultato di Squeezenet (S) e VGG16 (V)

----
## 5. Conclusioni

Dai risultati ottenuti, è stato verificato che il nostro dataset più grande soffre maggiormente nel cercare di classificare situazioni non presenti nel training set (&quot;WeirdImageSet&quot;). Questo problema viene in parte risolto sfoltendo il dataset così da ridurre l&#39;omogeneità dello stesso. Infatti con un dataset sensibilmente ridotto (2000 immagini per classe nei nostri esperimenti) sul &quot;WeirdImageSet&quot; si ottengono risultati migliori sia con la Squeezenet che con la VGG16. Tuttavia l&#39;accuracy sul test set diminuisce leggermente, ciò significa che bisogna aumentare l&#39;eterogeneità del dataset invece di ridurne la numerosità.

In futuro questo progetto potrebbe essere migliorato per poter usare il nostro modello su una pulsantiera più grande e con più tasti perchè i pesi aggiornati dal training delle due reti possono essere utilizzati come base per il nuovo training, in quanto sono state imparate le feature caratteristiche dei tasti. Ovviamente è necessario cambiare il numero di classi in output ed il dataset essendo, a quel punto, cambiato il task.

----
## 6. Citazioni

1. Iandola, Forrest N., et al. &quot;SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and\&lt; 0.5 MB model size.&quot; arXiv preprint arXiv:1602.07360 (2016).
2. Simonyan, Karen, and Andrew Zisserman. &quot;Very deep convolutional networks for large-scale image recognition.&quot; arXiv preprint arXiv:1409.1556 (2014).
