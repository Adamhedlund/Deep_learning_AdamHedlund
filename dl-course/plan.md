# Kursplan: grunderna i djupinlärning med PyTorch

**Mål:** förstå hela arbetsflödet för att träna ett neuralt nätverk, och veta *när* man använder vilken teknik. Kursen håller sig till grunderna, och fördjupningar läggs till senare (se slutet).

**Förkunskaper:** grundläggande ML med scikit-learn (train/test-split, pipelines, korsvalidering, vanliga mått).

**Omfång:** 15 lektioner på 1,5–2,5 h vardera, totalt cirka 30 h. Med 6–10 h i veckan tar det 4–5 veckor.

**Verktyg:** PyTorch som huvudbibliotek, TensorFlow/Keras som jämförelse i L14. Jupyter i VS Code, GTX 1080 (8 GB). Varje träning tar under cirka 5 minuter.

## Så kommer du igång
1. Öppna `notebooks/L01_arbetsflode_och_miljo.ipynb` i VS Code och välj repots `.venv` som kernel.
2. Kör installationscellen (`%pip install -r ../requirements.txt`) en gång och starta om kerneln.
3. Gå igenom lektionerna i ordning. Lösningarna till övningarna finns i `solutions/LXX_..._losning.ipynb`. Försök själv först!

**Om `dlkit`:** ditt eget hjälppaket skapas när du kör `%%writefile`-cellerna i L05 och byggs ut i L07, L08 och L09. Lektionerna från L06 och framåt förutsätter att de cellerna är körda.

## Så är varje lektion uppbyggd
1. **Mål** och var lektionen hör hemma i arbetsflödet.
2. **Kort teori** i markdown, med intuition före formler.
3. **Code-along:** du skriver koden själv, steg för steg, och varje steg har en förklaring av *varför*.
4. **Vad händer om…:** ändra en sak (learning rate, batchstorlek, aktiveringsfunktion och så vidare). Gissa först och kör sedan.
5. **Övningar med TODO**, med asserts eller förväntade utdata så att du kan kontrollera dig själv. Lösningar finns i `solutions/`.
6. **När använder jag detta?** När tekniken passar, när den inte gör det och var i arbetsflödet den hör hemma.

## Det genomgående arbetsflödet
Arbetsflödet introduceras i L01 och återkommer i varje lektion.

1. Förstå problemet och välj mått.
2. Dela upp data i train/val/test.
3. Ta fram en baslinje (en naiv modell eller sklearn).
4. Bygg en datapipeline med normalisering.
5. Gör en liten modell som kan överanpassa en batch, som ett sanity check.
6. Träna och titta på inlärningskurvorna.
7. Diagnostisera: över- eller underanpassning.
8. Regularisera och justera hyperparametrar mot valideringsmängden.
9. Utvärdera **en gång** på testmängden.
10. Spara modellen.

---

## Del A: Grunder (cirka 7 h)

### L01 – Arbetsflöde och miljö (1,5 h)
- **Mål:** känna till arbetsflödet, ha en fungerande miljö med GPU och förstå varför data delas i tre delar.
- **Innehåll:**
  - Arbetsflödet ovan.
  - Installera torch från `requirements.txt` och välj `device` (CPU/GPU). Verifiera GPU-stödet med `torch.cuda.is_available()` och kontrollera att `sm_60` finns i `torch.cuda.get_arch_list()`. Den koden körs på alla Pascal-kort, även GTX 1080 (sm_61).
  - Seeds för reproducerbarhet, och varför resultaten ändå kan variera lite på GPU.
  - Train/val/test: valideringsmängden används för modellval och testmängden bara en gång. Jfr `train_test_split`, och varför korsvalidering är ovanligt i DL (träningen tar för lång tid).
- **Dataset:** sklearn breast cancer.

### L02 – Tensorer och autograd (2 h)
- **Mål:** kunna skapa och räkna med tensorer och förstå hur PyTorch beräknar gradienter.
- **Innehåll:**
  - Shape, dtype och device, enkel broadcasting och bryggan till NumPy.
  - Vad en gradient är, `requires_grad` och `backward()`.
  - Varför `zero_grad()` behövs (gradienter ackumuleras) och vad `no_grad()` är till för.
- **Dataset:** syntetiskt.

### L03 – Träningsloopen, loss och optimerare (2 h)
- **Mål:** skriva en träningsloop för hand och förstå vad varje rad gör.
- **Innehåll:**
  - Loopen steg för steg: forward, loss, backward och step.
  - Linjär regression för hand jämförs med `LinearRegression`.
  - MSE för regression och CrossEntropy för klassificering. Vad logits är.
  - SGD kontra Adam och varför Adam är ett bra standardval.
  - **Vad händer om:** learning rate för låg, lagom och för hög, samt liten kontra stor batch.
- **Dataset:** California Housing.

### L04 – Data in i modellen (1,5 h)
- **Mål:** ladda data effektivt i batcher och normalisera den på rätt sätt.
- **Innehåll:**
  - `Dataset` och `DataLoader`: batch_size och shuffle, och varför shuffle bara används på träningsdata.
  - Normalisering som anpassas **bara på träningsdata**, som `StandardScaler` i en `Pipeline`.
- **Dataset:** Fashion-MNIST.

## Del B: Träna ett nätverk väl (cirka 10 h)

### L05 – MLP och aktiveringsfunktioner (2 h)
- **Mål:** bygga ett första neuralt nätverk och förstå valen av lager och aktiveringsfunktion.
- **Innehåll:**
  - Varför icke-linjäritet behövs (utan den är nätet bara linjär regression).
  - ReLU som standardval, och varför sigmoid sällan används i dolda lager.
  - Antal lager och neuroner kontra kapacitet. Jämförelse med `MLPClassifier`.
  - Träningsloopen flyttas in i `dlkit.train()`.
- **Dataset:** Fashion-MNIST.

### L06 – Utvärdering (2 h)
- **Mål:** välja rätt mått och kunna läsa av inlärningskurvor.
- **Innehåll:**
  - Mått per problemtyp: MAE och RMSE för regression, accuracy, precision, recall och F1 för klassificering, med `sklearn.metrics`.
  - Inlärningskurvor (train och val loss) och hur över- och underanpassning ser ut.
  - Enkel felanalys: confusion matrix och vilka exempel modellen klassar fel.
- **Dataset:** Fashion-MNIST.

### L07 – Regularisering (2 h)
- **Mål:** kunna motverka överanpassning och veta vilken teknik man tar till när.
- **Innehåll:**
  - Early stopping.
  - Weight decay, jfr Ridge och `C` i sklearn.
  - Dropout.
  - Batch norm i grundform.
  - `model.train()` kontra `model.eval()` och varför det spelar roll.
- **Dataset:** litet urval av Fashion-MNIST, så att överanpassningen syns tydligt.

### L08 – Hyperparametrar och LR schedules (2 h)
- **Mål:** veta vilka hyperparametrar som är viktigast och söka efter dem på ett enkelt sätt.
- **Innehåll:**
  - Prioriteringsordning: learning rate först, sedan batchstorlek, modellstorlek och regularisering.
  - Manuell sökning och enkel random search, jfr `RandomizedSearchCV`.
  - Två LR schedules: StepLR och ReduceLROnPlateau.
- **Dataset:** Fashion-MNIST.

### L09 – Spara, logga och felsöka (2 h) *(tillägg)*
- **Mål:** kunna spara modeller, jämföra experiment och hitta fel när träningen inte fungerar.
- **Innehåll:**
  - `state_dict`: spara och ladda den bästa modellen.
  - Logga körningar i TensorBoard.
  - Felsökningschecklista: överanpassa en batch, rimlig loss vid start, NaN och för hög LR.
  - Kort om varför initieringen spelar roll (PyTorch har bra standardval).
- **Motivering:** utan loggning går det inte att jämföra experiment, och utan en felsökningsrutin är det svårt att hitta tysta buggar. Båda behövs i alla senare lektioner.
- **Dataset:** Fashion-MNIST.

## Del C: Nätverkstyper (cirka 8,5 h)

### L10 – CNN och data augmentation (2,5 h)
- **Mål:** förstå varför CNN passar bilder och kunna träna en enkel CNN.
- **Innehåll:**
  - Faltning, kernel, padding och pooling, förklarat intuitivt.
  - Varför CNN passar bilder bättre än MLP (delade vikter och lokala mönster).
  - Enkel augmentation (flip, crop) bara på träningsdata, som en form av regularisering.
- **Dataset:** CIFAR-10.

### L11 – Transfer learning (2 h)
- **Mål:** återanvända en förtränad modell och veta när det lönar sig.
- **Innehåll:**
  - Förtränad ResNet18.
  - Feature extraction (frysa basen) kontra fine-tuning, och hur valet beror på mängden data.
- **Dataset:** Oxford-IIIT Pets (37 raser, cirka 3 700 träningsbilder, cirka 800 MB nedladdning).

### L12 – RNN/LSTM för sekvenser (2 h)
- **Mål:** förstå idén med rekurrenta nät och hur man arbetar med tidsserier.
- **Innehåll:**
  - Idén med rekurrens och varför LSTM används.
  - Att skapa fönster av en tidsserie.
  - **Tidsbaserad uppdelning** (jfr `TimeSeriesSplit`) och en naiv baslinje (gårdagens värde).
- **Dataset:** CTA ridership (`AI25-Deep_learning/Data/`).

### L13 – Attention och transformers: grundidén (2 h)
- **Mål:** förstå vad attention gör och varför transformers dominerar.
- **Innehåll:**
  - Vad attention gör, med intuition och en liten räkning för hand.
  - Varför transformers har ersatt RNN.
  - En liten textklassificerare med `nn.TransformerEncoder`.
- **Dataset:** IMDB-recensioner via Hugging Face `datasets` (20 000 train, 5 000 val, 25 000 test).

## Del D: Avslutning (cirka 5 h)

### L14 – Samma sak i Keras (1,5 h)
- **Mål:** känna igen samma koncept i TensorFlow/Keras.
- **Innehåll:**
  - Bygg om MLP:n från L05 i Keras.
  - `compile` och `fit` jämfört med en egen loop, callbacks jämfört med early stopping.

### L15 – Miniprojekt (3–4 h)
- **Mål:** genomföra hela arbetsflödet självständigt på ett eget litet problem.
- **Innehåll:** checklistan från arbetsflödet, en experimentlogg och en kort sammanfattning av resultat och val.
- **Placering:** `projects/`.

---

## Framstegslogg
| Lektion | Status | Datum | Anteckningar |
|---|---|---|---|
| L01 | ☐ | | |
| L02 | ☐ | | |
| L03 | ☐ | | |
| L04 | ☐ | | |
| L05 | ☐ | | |
| L06 | ☐ | | |
| L07 | ☐ | | |
| L08 | ☐ | | |
| L09 | ☐ | | |
| L10 | ☐ | | |
| L11 | ☐ | | |
| L12 | ☐ | | |
| L13 | ☐ | | |
| L14 | ☐ | | |
| L15 | ☐ | | |

## Senare fördjupning (inte i kursen, läggs till vid behov)
- Numerisk gradientkontroll och hooks.
- Initiering i detalj (Xavier och He) och gradient clipping.
- LayerNorm i detalj, AdamW kontra L2.
- Cosine och OneCycle, warmup, LR range test, Optuna.
- Residualblock och ResNet från grunden.
- Kalibrering, ROC-AUC och PR-AUC.
- Tokenisering i detalj och packade sekvenser.
- Attention från grunden, en liten GPT.
- Mixed precision (långsam fp16 på GTX 1080).
- Generativa modeller (se AI25 L9–L11).
