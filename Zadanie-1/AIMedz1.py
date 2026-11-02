# %% [markdown]
# # Zadanie 1 — PenDigits
# Rozpoznawanie cyfr pisanych piórem na tablecie: 16 współrzędnych (x, y) -> cyfra 0–9.
# Każdy blok "# %%" to jedna komórka notatnika.

# %% Importy
import pickle
import time

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import binomtest
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, GroupKFold, StratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

RS = 42   # ziarno losowości: za każdym razem te same wyniki
pd.set_option("display.max_columns", 20)   # pokazuj wszystkie kolumny tabel
pd.set_option("display.width", 200)

# %% [markdown]
# ## A — Poznanie i przygotowanie danych

# %% A1. Wczytanie oficjalnych plików train i test
# Pliki nie mają nagłówka, więc nazwy kolumn nadajemy sami:
# x1, y1, x2, y2, ..., x8, y8 (8 punktów trajektorii pióra) + cyfra.
cechy = []
for i in range(1, 9):
    cechy.append(f"x{i}")
    cechy.append(f"y{i}")
TARGET = "cyfra"

FOLDER = "data/zadanie-1/pendigits/"   # ścieżka względem folderu Zadanie-1
train = pd.read_csv(FOLDER + "pendigits.tra", header=None, names=cechy + [TARGET])
test = pd.read_csv(FOLDER + "pendigits.tes", header=None, names=cechy + [TARGET])

print("train:", train.shape, " test:", test.shape)
print(train.head())


# %% A2. Kto napisał każdą cyfrę?
# W archiwum są też pliki "pendigits-orig" z surowymi danymi. Przy każdej cyfrze jest tam linia:
#     .COMMENT 8 12  224
# czyli: cyfra 8, napisana przez osobę nr 12, jej 224. próbka.
# Kolejność cyfr jest taka sama jak w plikach, które wczytaliśmy wyżej.
def wczytaj_piszacych(sciezka):
    piszacy = []
    with open(sciezka) as plik:                 # otwórz plik tekstowy
        for linia in plik:                      # czytaj linijka po linijce
            if linia.startswith(".COMMENT"):
                czesci = linia.split()          # ".COMMENT 8 12 224" -> [".COMMENT", "8", "12", "224"]
                piszacy.append(int(czesci[2]))  # numer osoby to trzecia część
    return piszacy


train["piszacy"] = wczytaj_piszacych(FOLDER + "pendigits-orig.tra")
test["piszacy"] = wczytaj_piszacych(FOLDER + "pendigits-orig.tes")
print("liczba osób w train:", train["piszacy"].nunique(), " w test:", test["piszacy"].nunique())
print("ile próbek napisała każda osoba (train):", train["piszacy"].value_counts().min(),
      "-", train["piszacy"].value_counts().max())

# %% A3. Typy, braki, duplikaty
train.info()
print("braki w train:", train.isna().sum().sum(), " w test:", test.isna().sum().sum())
print("duplikaty w train:", train[cechy].duplicated().sum(), " w test:", test[cechy].duplicated().sum())
print("wiersze wspólne dla train i test:", len(train[cechy].merge(test[cechy])))

# %% A4. Rozkład klas
print(train[TARGET].value_counts(normalize=True).sort_index().round(3))
plt.figure(figsize=(7, 3))                                   # nowy, pusty wykres
train[TARGET].value_counts().sort_index().plot(kind="bar", title="Ile jest każdej cyfry (train)")
plt.xlabel("cyfra")
plt.ylabel("liczba próbek")
plt.show()

# %% A5. Zakresy wartości
print(train[cechy].describe().T[["min", "max", "mean", "std"]].round(1))
plt.figure(figsize=(10, 4))
sns.boxplot(data=train[cechy])
plt.title("Rozkład każdej cechy (train)")
plt.ylabel("współrzędna")
plt.show()

# %% A6. Jak wyglądają dane: jedna przykładowa cyfra każdego rodzaju
# Cechy to kolejne punkty pióra, więc można je narysować jako linię.
fig, axes = plt.subplots(1, 10, figsize=(14, 2))
for cyfra in range(10):
    wiersz = train[train[TARGET] == cyfra].iloc[0]          # pierwsza próbka danej cyfry
    xs = [wiersz[f"x{i}"] for i in range(1, 9)]
    ys = [wiersz[f"y{i}"] for i in range(1, 9)]
    axes[cyfra].plot(xs, ys, "o-")
    axes[cyfra].set_title(str(cyfra))
    axes[cyfra].set_xticks([])
    axes[cyfra].set_yticks([])
plt.show()

# %% [markdown]
# ### Wnioski z eksploracji i preprocessing
# - **Zmienna docelowa:** `cyfra` (0–9), więc to klasyfikacja z 10 klasami.
#   **Cechy:** 16 liczb całkowitych, czyli współrzędne x i y ośmiu punktów śladu pióra.
#   Autorzy zbioru przeskalowali je już do zakresu 0–100.
#   Kolumna `piszacy` (numer osoby) **nie jest cechą**. Służy tylko do podziału danych
#   w walidacji krzyżowej (B2).
# - **Rozkład klas:** prawie równy (9,6–10,4% na cyfrę), taki sam w train i test.
#   Accuracy jest więc sensowną metryką. Dodatkowo podaję macro F1 (średnie F1 wszystkich cyfr).
# - **Braki:** brak. **Duplikaty:** brak, także między train i test.
# - **Zakresy:** wszystkie cechy w 0–100, bez wartości niemożliwych.
# - **Oficjalny podział:** train napisało 30 osób, test 14 innych osób. Nie dzielę danych na nowo.
#   Test sprawdza więc, jak model radzi sobie z pismem osób, których nigdy nie widział.
# - **Preprocessing:** jedynym krokiem jest `StandardScaler` dla modeli wrażliwych na skalę
#   (regresja logistyczna, k-NN, SVM). Cechy mają różne rozrzuty (odchylenie od 17 do 42).
#   Skaler jest w `Pipeline`, więc uczy się tylko na danych treningowych.

# %% [markdown]
# ## B — Modelowanie

# %% B1. Dane do modeli i lista kandydatów
X_train = train[cechy]
y_train = train[TARGET]
osoby_train = train["piszacy"]      # potrzebne tylko do podziału na foldy
X_test = test[cechy]
y_test = test[TARGET]

# nazwa: (pipeline, siatka hiperparametrów do sprawdzenia)
kandydaci = {
    "Baseline": (
        Pipeline([("model", DummyClassifier(strategy="most_frequent"))]),
        {},
    ),
    "Naive Bayes": (
        Pipeline([("model", GaussianNB())]),
        {},
    ),
    "LDA": (
        Pipeline([("model", LinearDiscriminantAnalysis(store_covariance=True))]),
        {},
    ),
    "Regresja logistyczna": (
        Pipeline([("skaler", StandardScaler()), ("model", LogisticRegression(max_iter=3000))]),
        {"model__C": [0.1, 1, 10]},
    ),
    "k-NN": (
        Pipeline([("skaler", StandardScaler()), ("model", KNeighborsClassifier())]),
        {"model__n_neighbors": [1, 3, 5, 7]},
    ),
    "SVM": (
        Pipeline([("skaler", StandardScaler()), ("model", SVC())]),
        {"model__C": [1, 10, 100]},
    ),
    "Las losowy": (
        Pipeline([("model", RandomForestClassifier(random_state=RS))]),
        {"model__n_estimators": [100, 300]},
    ),
}

# %% B2. Walidacja: foldy losowe czy według osób?
# Foldy losowe: próbki tej samej osoby trafiają i do części uczącej, i do sprawdzającej.
#   Model "zna" już jej charakter pisma, więc wynik wychodzi zawyżony.
# Foldy według osób (GroupKFold): każda osoba jest w całości w jednej części foldu.
#   Tak samo jest w oficjalnym teście (inne osoby niż w train), więc to uczciwsze oszacowanie.
cv_losowe = StratifiedKFold(n_splits=5, shuffle=True, random_state=RS)
cv_wg_osob = GroupKFold(n_splits=5)

knn = Pipeline([("skaler", StandardScaler()), ("model", KNeighborsClassifier(n_neighbors=1))])
wynik_losowe = cross_val_score(knn, X_train, y_train, cv=cv_losowe).mean()
wynik_wg_osob = cross_val_score(knn, X_train, y_train, cv=cv_wg_osob, groups=osoby_train).mean()
print(f"k-NN (k=1), foldy losowe:     {wynik_losowe:.4f}")
print(f"k-NN (k=1), foldy według osób: {wynik_wg_osob:.4f}")

# %% B3. Strojenie: walidacja krzyżowa według osób, tylko na zbiorze treningowym
najlepsze = {}     # nazwa -> najlepszy wyuczony pipeline
wynik_cv = {}      # nazwa -> średnia accuracy z walidacji krzyżowej
for nazwa, (pipe, siatka) in kandydaci.items():
    gs = GridSearchCV(pipe, siatka, cv=cv_wg_osob, scoring="accuracy")
    gs.fit(X_train, y_train, groups=osoby_train)   # groups mówi, która próbka jest czyja
    najlepsze[nazwa] = gs.best_estimator_
    wynik_cv[nazwa] = gs.best_score_
    print(f"{nazwa:22s} CV accuracy = {gs.best_score_:.4f}   najlepsze ustawienia: {gs.best_params_}")

# %% [markdown]
# ### Dobór modeli i ochrona przed wyciekiem danych
# - **Baseline:** zawsze odpowiada najczęstszą cyfrą, około 10%. To punkt odniesienia.
# - **Naive Bayes:** prosty klasyfikator statystyczny. Zakłada, że cechy są niezależne.
# - **LDA:** klasyfikator statystyczny. Zakłada, że wszystkie cyfry mają taki sam kształt
#   rozrzutu danych, przez co granice między klasami są liniowe.
# - **Regresja logistyczna:** model liniowy. Stroję siłę regularyzacji `C`.
# - **k-NN:** głosowanie najbliższych sąsiadów. Stroję liczbę sąsiadów.
# - **SVM:** granice nieliniowe (jądro RBF). Stroję `C`.
# - **Las losowy:** wiele drzew decyzyjnych, które głosują. Stroję liczbę drzew.
#
# Ochrona przed wyciekiem:
# 1. skalowanie siedzi w `Pipeline`,
# 2. hiperparametry wybieram walidacją krzyżową na zbiorze treningowym,
# 3. foldy są według osób, bo próbki jednej osoby są do siebie podobne (pokaz w B2),
# 4. zbioru testowego używam tylko raz, w części C,
# 5. train i test nie mają wspólnych wierszy.

# %% [markdown]
# ## C — Porównanie i wnioski (oficjalny zbiór testowy)

# %% C1. Ile liczb przechowuje każdy model? (miara wielkości modelu)
def liczba_parametrow(nazwa, pipe):
    m = pipe.named_steps["model"]               # sam model, bez skalera
    if nazwa == "Regresja logistyczna":
        return m.coef_.size + m.intercept_.size             # wagi + wyrazy wolne
    if nazwa == "LDA":
        return m.means_.size + m.covariance_.size           # średnie cyfr + wspólna kowariancja
    if nazwa == "Naive Bayes":
        return m.theta_.size + m.var_.size                  # średnie i wariancje cech dla każdej cyfry
    if nazwa == "SVM":
        return m.support_vectors_.size + m.dual_coef_.size  # zapamiętane "wektory nośne" + ich wagi
    if nazwa == "k-NN":
        return X_train.size                                 # pamięta cały zbiór treningowy
    if nazwa == "Las losowy":
        return sum(drzewo.tree_.node_count for drzewo in m.estimators_)   # węzły wszystkich drzew
    return 0                                                # baseline niczego nie przechowuje


# %% C2. Metryki, czas i wielkość modeli
wiersze = []
predykcje = {}     # nazwa -> odpowiedzi modelu na teście (przydadzą się w C5)
for nazwa, pipe in najlepsze.items():
    start = time.perf_counter()
    pipe.fit(X_train, y_train)                    # ponowne uczenie tylko po to, żeby zmierzyć czas
    czas_uczenia = time.perf_counter() - start

    start = time.perf_counter()
    y_pred = pipe.predict(X_test)
    czas_predykcji = time.perf_counter() - start
    predykcje[nazwa] = y_pred

    wiersze.append({
        "model": nazwa,
        "CV accuracy": wynik_cv[nazwa],
        "test accuracy": accuracy_score(y_test, y_pred),
        "macro F1": f1_score(y_test, y_pred, average="macro"),
        "liczba błędów": (y_pred != y_test).sum(),
        "uczenie [s]": czas_uczenia,
        "predykcja [s]": czas_predykcji,
        "parametry": liczba_parametrow(nazwa, pipe),
        # pickle zamienia model na bajty (tak jak przy zapisie do pliku), więc to jego rozmiar
        "rozmiar [kB]": len(pickle.dumps(pipe)) / 1024,
    })

wyniki = pd.DataFrame(wiersze).set_index("model").sort_values("test accuracy", ascending=False)
print(wyniki.round(4))

# %% C3. Wykres: dokładność a czas predykcji
bez_baseline = wyniki.drop(index="Baseline")
plt.figure(figsize=(8, 4.5))
plt.scatter(bez_baseline["predykcja [s]"], bez_baseline["test accuracy"])
for nazwa in bez_baseline.index:                       # podpis przy każdym punkcie
    punkt = (bez_baseline.loc[nazwa, "predykcja [s]"], bez_baseline.loc[nazwa, "test accuracy"])
    plt.annotate(nazwa, punkt, textcoords="offset points", xytext=(6, 4))   # napis 6 px w prawo, 4 px w górę
plt.xscale("log")                                      # czasy różnią się kilkadziesiąt razy
plt.margins(x=0.25)                                    # miejsce na podpisy
plt.xlabel("czas predykcji całego zbioru testowego [s, skala log]")
plt.ylabel("accuracy na teście")
plt.title("Dokładność a koszt")
plt.show()

# %% C4. Błędy najlepszego modelu
najlepszy = wyniki.index[0]                       # pierwszy wiersz posortowanej tabeli
y_pred = predykcje[najlepszy]
print("Najlepszy model:", najlepszy)
print(classification_report(y_test, y_pred, digits=3))

macierz = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(macierz, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("przewidziana cyfra")
plt.ylabel("prawdziwa cyfra")
plt.title(f"Macierz pomyłek: {najlepszy}")
plt.show()

# Dokładność najlepszego modelu dla każdej z 14 osób testowych
trafne = y_pred == y_test
na_osobe = trafne.groupby(test["piszacy"]).mean()
print(f"accuracy dla osób testowych: od {na_osobe.min():.3f} do {na_osobe.max():.3f}, "
      f"mediana {na_osobe.median():.3f}")

# %% C5. Czy przewaga najlepszego modelu nie jest przypadkiem? Test McNemara
# Liczymy tylko próbki, na których dwa modele się różnią:
#   a = najlepszy model ma rację, drugi się myli
#   b = najlepszy się myli, drugi ma rację
# Gdyby modele były równie dobre, a i b powinny być podobne (jak orły i reszki przy rzucie monetą).
# binomtest liczy, jak prawdopodobny byłby taki podział przy "rzucie monetą" (p-wartość).
# p < 0,05 -> różnica raczej nie jest przypadkiem.
dobrze_najlepszy = predykcje[najlepszy] == y_test
for nazwa in wyniki.index[1:]:                    # wszystkie modele poza najlepszym
    if nazwa == "Baseline":
        continue                                  # porównanie z baseline nic nie wnosi
    dobrze_inny = predykcje[nazwa] == y_test
    a = (dobrze_najlepszy & ~dobrze_inny).sum()   # ~ to zaprzeczenie: True <-> False
    b = (~dobrze_najlepszy & dobrze_inny).sum()
    p = binomtest(int(a), int(a + b), 0.5).pvalue
    print(f"{najlepszy} vs {nazwa:22s} a = {a:3d}, b = {b:3d}, p = {p:.2g}")

# %% [markdown]
# ### Wnioski
# (Liczby z mojego uruchomienia. Czasy zależą od komputera, reszta wychodzi zawsze tak samo.)
#
# 1. **Wszystkie modele wyraźnie biją baseline** (10,4%). Najlepszy jest SVM, 98,1% accuracy
#    na teście, 65 błędów na 3498 próbek. Drugi jest k-NN (97,5%, 88 błędów), trzeci las losowy
#    (96,5%, 124 błędy).
# 2. **Modele statystyczne i liniowe są wyraźnie słabsze:** Naive Bayes 82,3%, LDA 83,0%,
#    regresja logistyczna 92,8%. Ich założenia (niezależne cechy, liniowe granice) nie pasują
#    do danych, w których kolejne punkty śladu pióra są mocno ze sobą powiązane.
# 3. **Błędy SVM:** najtrudniejsza jest cyfra 7 (czułość 92,3%), mylona z 9, 1 i 3 (po 8–9 razy).
#    Często mylone jest też 0 z 8 (8 razy): obie cyfry zaczynają się u góry i zataczają pętlę.
#    Dokładność zależy od osoby: od 94,8% do 100% (mediana 98,6%) dla 14 osób testowych.
# 4. **Walidacja według osób była potrzebna:** foldy losowe zawyżają wynik (dla k-NN to 99,4% zamiast
#    96,7%), bo model widzi w nauce próbki tej samej osoby. Foldy według osób dały wyniki bliskie
#    testowi (dla SVM to 97,8% w CV, 98,1% na teście), więc wybór modelu był wiarygodny.
# 5. **Czy przewaga SVM jest prawdziwa?** Test McNemara: SVM vs k-NN p ≈ 0,002, vs pozostałe
#    jeszcze mniej, więc różnice raczej nie są przypadkiem. Test zakłada jednak niezależne
#    próbki, a próbki jednej osoby są podobne, więc te p-wartości są raczej zbyt optymistyczne.
# 6. **Koszt:** SVM przechowuje ok. 21 tys. liczb (ok. 170 kB). k-NN pamięta cały zbiór treningowy
#    (ok. 120 tys. liczb, ok. 1 MB). Las losowy jest największy (ok. 181 tys. węzłów, ok. 25 MB),
#    uczy się najdłużej i jest gorszy od SVM. Regresja logistyczna ma tylko 170 parametrów (3 kB)
#    i przewiduje ok. 20 razy szybciej niż SVM, ale robi prawie 4 razy więcej błędów.
#
# **Czy wzrost złożoności jest wart poprawy?** Przejście z modelu liniowego (regresja logistyczna)
# na nieliniowy (SVM) owszem jest warty, liczba błędów spada z 251 do 65, a SVM nadal jest mały i szybki
# (cały zbiór testowy w ułamku sekundy). Dalszy wzrost złożoności jednak nie, las losowy jest
# ok. 150 razy większy od SVM, uczy się dłużej i robi więcej błędów. Rekomenduję SVM jako
# najlepszy kompromis jakości i kosztu.
# **Narzędzia AI:** Claude (Anthropic)
