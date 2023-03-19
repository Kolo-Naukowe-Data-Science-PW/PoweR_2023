##### PoweR: Modele uczenia maszynowego #####
##### Mateusz Krzyziński ######
##### 2023-03-20 ######

# prezentacja (z kodem i obrazkami) dostępna pod linkiem 
# ---> https://rpubs.com/krzyzinskim/1017071 <---

#### Wstęp ####
### Zanim przejdziemy do meritum... ###
## Set up
# Instalacja pakietów, z których będziemy korzystać. Lepiej to zrobić **przed zajęciami** :)
install.packages('DALEX')
install.packages('partykit')
install.packages('randomForest')
install.packages('gbm')
install.packages('e1071')

## Czemu paczek jest tak dużo?
# - w `R` wiele modeli ma swoje oddzielne implementacje w osobnych bibliotekach
# - nawet meta-pakiety wykorzystują de facto pod spodem te mniejsze paczki
# 
# Takie rozdrobnienie ma też swój plus -- pakiety są bardziej dopracowane, mają więcej funkcjonalności i parametrów.
# Ogrom dostępnych bibliotek można zauważyć przeglądając 
# [CRAN Task View: Machine Learning & Statistical Learning](https://cran.r-project.org/web/views/MachineLearning.html).\



### Program ###
## GŁÓWNY CEL: Zapoznanie z ...
# - podziałem (taksonomią) uczenia maszynowego - czyli czego i po co *uczymy maszyny*?
# - podstawowymi modelami uczenia maszynowego - czyli jak je *uczymy*?
  

## Na jakich modelach się skupimy?
# -   regresja liniowa,
# -   regresja logistyczna,
# -   drzewa decyzyjne,
# -   lasy losowe,
# -   modele boostingowe,
# -   metoda k najbliższych sąsiadów.



### Taksonomia uczenia maszynowego ###

## patrz: slajdy


### Uczenie nadzorowane ###

## Co mamy?
# X -- dane (zmienne objaśniające/predyktory/zmienne niezależne)
#      w większości przypadków chcemy, żeby $X \in R_{n x p}$ (n obserwacji p zmiennych) 
#      -- właśnie po to jest nam potrzebny feature engineering (poprzednie zajęcia)
# Y -- odpowiedzi (zmienna objaśniana/target/zmienna zależna)
#      dla zadania regresji $Y \in R$ (odpowiedź to liczba)
#      dla zadania klasyfikacji $Y \in {1, 2, ..., g}$ (odpowiedź to etykieta, kategoria)

## Co chcemy osiągnąć?
# - F -- model, który umie przewidywać Y na podstawie X
# musi się tego nauczyć na podstawie zgromadzonych danych (poprzez trening/fitowanie/dopasowanie)

### Uczenie nadzorowane - przykłady problemów ###
## Regresja
# -   predykcja ceny mieszkania na podstawie jego cech (np. metrażu, położenia, jakości wykończenia, roku wybudowania),
# -   predykcja ilości sprzedaży danego produktu w zależności od jego ceny, promocji i innych czynników marketingowych,
# -   predykcja temperatury w danym miejscu i czasie na podstawie danych meteorologicznych,
# -   predykcja czasu, który użytkownik spędzi na danej stronie internetowej w zależności od liczby kliknięć i czasu spędzonego na innych stronach,

## Klasyfikacja
# -   predykcja, czy dany e-mail jest spamem na podstawie jego treści,
# -   predykcja, czy danemu klientowi banku należy przyznać wnioskowany kredyt na podstawie historii kredytowej i cech klienta,
# -   predykcja, czy na zdjęciu znajduje się pies, czy kot,
# -   predykcja, czy dany pacjent na podstawie wyników choruje na określoną chorobę,



#### Podstawowe modele uczenia maszynowego ####

### Wczytanie zbiorów ###
## dla zadania regresji:
apartments_regr_train <- DALEX::apartments
apartments_regr_test <- DALEX::apartments_test

## dla zadania klasyfikacji:
titanic_dataset <- DALEX::titanic_imputed
titanic_dataset$survived <- as.factor(titanic_dataset$survived )
n <- nrow(titanic_dataset)
set.seed(42)
train_mask <- sample(n, 0.8*n)
titanic_class_train <- titanic_dataset[train_mask,]
titanic_class_test <- titanic_dataset[-train_mask,]

pima_class_train <- MASS::Pima.tr
pima_class_test <- MASS::Pima.te

## Zbiór treningowy a testowy
# -   zbiór treningowy -- trenowanie (uczenie) modelu
# -   zbiór testowy -- testowanie jakości modelu na niewidzianych danych (następne zajęcia)



### Regresja liniowa ###
# -   Działa dla zadania regresji.
# -   Dla $p=1$ (jednej zmiennej objaśniającej) polega na dopasowaniu prostej (funkcji liniowej):
linreg_model_simple <- lm(m2.price ~ surface, data = apartments_regr_train)
plot(apartments_regr_train$surface, apartments_regr_train$m2.price, pch=16,
     xlab="Powierzchnia [m^2]",
     ylab="Cena za m^2")
abline(linreg_model_simple, col="red", lwd=3)
coef(linreg_model_simple)


# -   Ogólniej: nasz model $F$ wyraża się jako: $\hat{Y} = F(X) = X \hat{\beta}$, 
#     czyli dla $i$-tej obserwacji: \hat{y_i} = \hat{\beta_0} + \hat{\beta_1} * x_{i1} +  \hat{\beta_2} * x_{i2} + ... + \hat{\beta_p} * x_{ip}
# -   Na czym polega trening?
#   -   szukamy współczynników (wag modelu) \hat{\beta},
#   -   wybieramy takie \hat{\beta}, które minimalizuje sumę kwadratów błędów predykcji na zbiorze treningowym $L = \sum_{i=1}^{n} (y_i - \hat{y_i})^2$,
# -   przy takiej funkcji straty $L$ mamy wzór jawny na $\hat{\beta}$.


# -   W praktyce używamy wbudowanej w `R` funkcji `lm()` (od linear model):
linreg_model <- lm(m2.price ~ ., data = apartments_regr_train)
summary(linreg_model) # podsumowanie statystyczne


predictions <- predict(linreg_model, apartments_regr_test[1:10, ])
as.numeric(predictions)
apartments_regr_test[1:10, "m2.price"]

## Jak sprawdzić czy model działa dobrze?
# To temat następnych zajęć.
plot(predictions, type="l", col="red", lwd=3,
     xlab="Indeks obserwacji (mieszkania)", ylab="Cena za m^2")
lines(apartments_regr_test[1:10, "m2.price"], col="blue", lwd=3)
legend(x=5, y=4500, lwd=3, 
       col=c("red", "blue"), 
       legend=c("y_hat", "y_true"))

## ZALETY:
# -   Istnieje wiele statystycznych metod testowania i diagnostyki.
# -   Interpretowalność.
# -   Łatwość optymalizacji (jawny wzór) i lekkość.

## WADY:
# -   Podatność na obserwacje odstające.
# -   Założenie o liniowej zależności pomiędzy zmiennymi objaśniającymi a zmienną odpowiedzi.




### Regresja logistyczna ###
# -   "Odpowiednik" regresji liniowej dla zadania klasyfikacji.
# -   Dla dwóch klas {0, 1}: interesuje nas P(y=1/x) = p_1(x) prawdopodobieństwo przynależności danej obserwacji x do klasy 1 (prawdopodobieństwo "sukcesu").
# -   Dla p=1 (jednej zmiennej objaśniającej):
logreg_model_simple <- glm(type~glu, data=pima_class_train, family = "binomial")
plot(pima_class_train$glu, pima_class_train$type == "Yes", pch=16,
     xlab="Stężenie glukozy",
     ylab="Cukrzyca")
glu_sequence <- seq(min(pima_class_train$glu), max(pima_class_train$glu), 1)
predictions <- predict(logreg_model_simple, data.frame(glu=glu_sequence), type = "response")
lines(glu_sequence, predictions, type="l", lwd=3, col="red")
coef(logreg_model_simple)

# -   Zamiast prawdopodobieństwa modelujemy liniowo logarytm szans (p-stwo jest z przedziału [0, 1], a logarytm szans nie).
# -   Szansa to stosunek prawdopodobieństwa sukcesu do prawdopodobieństwa porażki: $odds = p_1 / (1 - p_1).
# -   Nasz model wyraża się jako: log(odds) = \hat{\beta_0} + \hat{\beta_1} * x_{i1} +  \hat{\beta_2} * x_{i2} + ... + \hat{\beta_p} * x_{ip}
# -   Na czym polega trening?
#   -   szukamy współczynników (wag modelu) \hat{\beta},
#   -   wybieramy takie \hat{\beta}, które maksymalizuje funkcję wiarogodności (likelihood), tzn.:
#     -   dla obserwacji z klasy 1 daje wartości p_1(x) możliwie bliskie 1,
#     -   dla obserwacji z klasy 0 daje wartości p_1(x) możliwie bliskie 0.

# W praktyce używamy wbudowanej w `R` funkcji `glm()` (od *generalized linear model*):
logreg_model <- glm(survived~., data=titanic_class_train, family = "binomial")
summary(logreg_model)

# Domyślnie predykowane są log-oddsy. W celu uzyskania prawdopodobieństw należy zmienić parametr `type` na `"response"`.
predictions_logodds <- predict(logreg_model, titanic_class_test[1:10, ]) 
as.numeric(predictions_logodds)

predictions_probs <- predict(logreg_model, titanic_class_test[1:10, ], type="response") 
as.numeric(predictions_probs)

titanic_class_test[1:10, "survived"]

## ZALETY:
# -   Istnieje wiele statystycznych metod testowania i diagnostyki.
# -   Interpretowalność.

## WADY:
# -   Podatność na obserwacje odstające.
# -   Założenie o liniowości.




### Drzewo decyzyjne ###
# -   Może być użyte zarówno do problemu regresji, jak i klasyfikacji -- często nazywane CART (Classification And Regression Trees).
# -   Działanie polega na podziale przestrzeni predyktorów na regiony o różnej wartości zmiennej odpowiedzi (dla klasyfikacji -- prawdopodobieństwo przynależności do klasy).
# -   Algorytm można opisać krokowo:
#   1.  Zaczynamy od korzenia drzewa -- pojedynczego węzła, w którym mamy cały zbiór danych.
#   2.  Dla obecnego węzła szukamy odpowiedniego podziału zbioru danych. W tym celu rozważamy każdą zmienną i dla każdej z nich analizujemy:
#       (a) jeśli jest ciągła: każdy możliwy cutoff (wartość odcięcia),
#       (b) jeśli jest kategoryczna: każdy możliwy podzbiór poziomów.
#       Wybieramy taki podział (zmienną i wartość), która maksymalizuje miarę odseparowania, umożliwia jak najbardziej różnicujący podział danych ze względy na $Y$.
#   3.  Sprawdzamy, czy zostało spełnione kryterium stopu, np.
#       (a) osiągnęliśmy maksymalną głębokość drzewa,
#       (b) osiągnęliśmy minimalny zysk na czystości węzłów.
#       Jeśli warunek stopu jest spełniony -- przerywamy.
#       W przeciwnym wypadku -- dzielimy obecny węzeł na dwa według znalezionego podziału i dla każdego z węzłów-potomków idziemy do kroku 2 (niezależnie).
# Miary jakości podziału:
#   -   dla regresji: minimalizujemy błąd średniokwadratowy patrząc na predykcję dla regionów $R_1$ i $R_2$**:**$MSE = \sum_{i \in R_1} (y_i-\hat{y_{R_1}})^2 + \sum_{i \in R_2} (y_i-\hat{y_{R_2}})^2$
#   -   dla klasyfikacji: Gini index, entropia.

# W praktyce jest wiele różnych pakietów oferujących budowanie drzew decyzyjnych. 
# Tutaj wykorzystamy bibliotekę `partykit`, która umożliwia tworzenie przejrzystych wizualizacji drzew oraz pozwala na specyfikację wielu parametrów (`ctree_control`).
library(partykit)
class_tree_model <- ctree(survived~., data=titanic_class_train)
class_tree_model
plot(class_tree_model)


regr_tree_model <- ctree(m2.price~., 
                         data=apartments_regr_train, 
                         control = ctree_control(maxdepth=4)
                    )
regr_tree_model
plot(regr_tree_model)

# Predykcją jest średnia wartość zmiennej odpowiedzi dla obserwacji ze zbioru treningowego, które trafiły do tego samego liścia.
predict(class_tree_model, titanic_class_test[1:10,])
predict(class_tree_model, titanic_class_test[1:10,], type="prob")
predict(regr_tree_model, apartments_regr_test[1:10,])

## ZALETY:
# -   Interpretowalność.
# -   Intuicyjność.

## WADY:
# -   Nienajlepszy performance.
# -   Nieodporność na zmiany w danych, overfitting.




### Las losowy ###
# -   Podobnie jak drzewo decyzyjne -- może być użyty zarówno do problemu regresji, jak i klasyfikacji.
# -   Idea: "mądrość tłumu" -- połączeniu wielu różnych modeli w jeden poprawi jego własności.
# -   Las tworzy się z wielu **różnych od siebie** drzew decyzyjnych, których predykcje są agregowane (ensembling).
# -   Zróżnicowanie drzew wynika z losowości zastosowanej na dwóch poziomach:
#   1.  trenujemy każde pojedyncze drzewo na nieco innym zbiorze danych 
#       (generujemy B zbiorów treningowych wykorzystując procedurę bootstrapu, 
#       czyli losując ze zwracaniem obserwacje z całego, pierwotnego zbioru treningowego),
#   2.  na każdym poziomie budowanego drzewa rozważamy tylko losowo wybrany podzbiór zmiennych jako kandydatów do podziału węzła.

# W praktyce są dwa bardzo popularne pakiety oferujące budowanie lasów losowych (`ranger` i `randomForest`). Tutaj wykorzystamy bibliotekę `randomForest`.
library(randomForest)
class_rf_model <- randomForest(survived~., data=titanic_class_train)
class_rf_model

regr_rf_model <- randomForest(m2.price~., data=apartments_regr_train)
regr_rf_model

# Predykcja jest dokonywana przez głosowanie pojedynczych drzew.
predict(class_rf_model, titanic_class_test[1:10,]) #klasy
predict(class_rf_model, titanic_class_test[1:10,], type="prob") #prawdopodobieństwa
predict(class_rf_model, titanic_class_test[1:10,], type="vote", norm.votes=FALSE) #liczba głosów

predict(regr_rf_model, apartments_regr_test[1:10,])




### Boosting ###
# -   Może być użyty zarówno do problemu regresji, jak i klasyfikacji.
# -   Idea: "mądrość tłumu" -- połączeniu wielu różnych modeli w jeden poprawi jego własności.
# -   Budowanie modelu polega na sekwencyjnym tworzeniu "słabych modeli" (w przypadku wykorzystania drzew decyzyjnych -- płytkich drzew).
# -   Modele tworzone są po sobie w taki sposób, że każdy kolejny ma na celu redukować błędy poprzedniego. 
#     Pierwsze drzewo trenowane jest z wykorzystaniem oryginalnego $Y$, a każde kolejne trenowane jest już w oparciu o rezydua (błędy) dotychczasowego modelu.

# W praktyce idea boostingu wykorzystywana jest w wielu różnych frameworkach (np. XGBoost, LightGBM, AdaBoost, Catboost, ...), tutaj wykorzystamy natomiast pakiet `gbm`.
library(gbm)
class_gbm_model <- gbm(as.character(survived)~., distribution="bernoulli", data=titanic_class_train)
class_gbm_model

regr_gbm_model <- gbm(m2.price~., distribution="gaussian", data=apartments_regr_train)
regr_gbm_model

# Predykcja jest dokonywana przez agregację predykcji poszczególnych drzew.
# popularny błąd: predykcje nie działają dla klasyfikatora, gdzie y był factorem
predict(class_gbm_model, titanic_class_test[1:10,])
predict(regr_gbm_model, apartments_regr_test[1:10,])




### Metoda k najbliższych sąsiadów ###
# -   Może być użyta zarówno do problemu regresji, jak i klasyfikacji.
# -   Bardzo prosta idea: dla nowej obserwacji wyszukujemy $k$ (predefiniowaną liczbę) najbliższych 
#     pod względem odległości obserwacji ze zbioru treningowego i wyznaczamy predykcję na ich podstawie:
#     -   dla klasyfikacji: prawdopodobieństwa klas to frakcja etykiet danej klasy wśród $k$ znalezionych sąsiadów,
#     -   dla regresji: średnia z $Y$ wśród $k$ znalezionych sąsiadów.
# -   Można wykorzystać właściwie dowolną miarę odległości -- kluczowe jest skalowanie zmiennych.


# W praktyce wykorzystamy pakiet `e1071`.
library(e1071)
class_knn_model <- gknn(survived~., data=titanic_class_train, scale=TRUE, k=5)
class_knn_model$x[1:2,]

regr_knn_model <- gknn(m2.price~., data=apartments_regr_train, scale=TRUE, k=5)
regr_knn_model$x[1:2,]

# Predykcje:
predict(class_knn_model, titanic_class_test[1:10,])
predict(class_knn_model, titanic_class_test[1:10,], type="votes")

predict(regr_knn_model, apartments_regr_test[1:10,])



#### Podsumowanie ####
### Polecane materiały ###
# -   J. Gareth, D. Witten, T. Hastie, R. Tibshirani. [An Introduction to Statistical Learning](https://www.statlearning.com)
# -   T. Hastie, R. Tibshirani, J. Friedman. [The Elements of Statistical Learning](https://hastie.su.domains/Papers/ESLII.pdf)
# -   P. Biecek, A. Kozak, A. Zawada. [The Hitchhiker's Guide to Responsible Machine Learning](https://betaandbit.github.io/RML/)
# -   [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
# -   [pakiet `forester`](https://modeloriented.github.io/forester/)

### Czas na pytania ###
  