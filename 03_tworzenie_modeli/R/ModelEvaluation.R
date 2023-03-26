########################
### PoweR warsztat 3 ###
###   27.03.2023     ###
###  Mikołaj Spytek  ###
########################

## Instalacja potrzebnych pakietów

# install.packages("DALEX")
# install.packages("MASS")
# install.packages("partykit")
# install.packages("randomForest")
# install.packages("gbm")
# install.packages("e1071")
# install.packages("caret")
# install.packages("glmnet")

## Wczytywanie pakietów

library(DALEX)
library(MASS)
library(partykit)
library(randomForest)
library(gbm)
library(e1071)
library(caret)
library(glmnet)

### Wczytywanie potrzebnych danych

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

## Tworzenie modeli i predykcji (modele z poprzednich zajęć)

### ZADANIE REGRESJI - ZBIÓR APARTMENTS
y_true_apartments <- apartments_regr_test$m2.price
## REGRESJA LINIOWA
linreg_model <- lm(m2.price ~ ., data = apartments_regr_train)
y_hat_linreg <- predict(linreg_model, apartments_regr_test)

## DRZEWO DECYZYJNE
regr_tree_model <- ctree(m2.price~.,
                         data = apartments_regr_train,
                         control = ctree_control(maxdepth = 4))
y_hat_tree <- predict(regr_tree_model, apartments_regr_test)

## LAS LOSOWY
regr_rf_model <- randomForest(m2.price~., data = apartments_regr_train)
y_hat_rf <- predict(regr_rf_model, apartments_regr_test)

## BOOSTING
regr_gbm_model <- gbm(m2.price~., distribution = "gaussian", data = apartments_regr_train)
y_hat_gbm <- predict(regr_gbm_model, apartments_regr_test)

## KNN
regr_knn_model <- gknn(m2.price~., data = apartments_regr_train, scale = TRUE, k = 5)
y_hat_knn <- predict(regr_knn_model,apartments_regr_test)

regr_model_names <- c("linreg", "tree", "rf", "gbm", "knn")

#################################
### EWALUACJA MODELI REGRESJI ###
#################################

# CEL: średnio jak daleko, jest predykcja naszego modelu od prawdziwej wartości
#
# Oznaczenia:
# y_hat  - predykcje modeli (y z daszkiem)
# y_true - wartości prawdziwe

## MEAN SQUARED ERROR (MSE) - BŁĄD ŚREDNIOKWADRATOWY
# MSE(y, \hat{y}) = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2

# spróbujmy ręcznie

mean_squared_error <- function(y_true, y_hat){
    mean((y_true - y_hat)^2)
}

mean_squared_error(y_true_apartments, y_hat_linreg)

# w praktyce możemy wykorzystać bibliotekę
library(ModelMetrics)

mse_linreg <- mse(y_true_apartments, y_hat_linreg)
mse_linreg
mse_tree <- mse(y_true_apartments, y_hat_tree)
mse_rf <- mse(y_true_apartments, y_hat_rf)
mse_gbm <- mse(y_true_apartments, y_hat_gbm)
mse_knn <- mse(y_true_apartments, y_hat_knn)

mse_values <- c(mse_linreg, mse_tree, mse_knn, mse_gbm, mse_rf)

# Zwizualizujmy wyniki
barplot(mse_values, names.arg = regr_model_names)

## RMSE - mała modyfikacja
# RMSE(y, \hat{y}) = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2}
# nie jest to żadna nowa informacja, być może łatwiej zintepretować wyniki

rmse_values <- sqrt(mse_values)
barplot(rmse_values, names.arg = regr_model_names)

## MEAN ABSOLUTE ERROR (MAE) - ŚREDNI BŁĄD BEZWZGLĘDNY
# MAE(y, \hat{y}) = \frac{1}{N} \sum_{i=1}^{N} |y_i - \hat{y}_i|

# spróbujmy ręcznie

mean_absolute_error <- function(y_true, y_hat){
    mean(abs(y_true - y_hat))
}

mean_absolute_error(y_true_apartments, y_hat_linreg)

# za pomocą pakietu
mae_linreg <- mae(y_true_apartments, y_hat_linreg)
mae_linreg
mae_tree <- mae(y_true_apartments, y_hat_tree)
mae_rf <- mae(y_true_apartments, y_hat_rf)
mae_gbm <- mae(y_true_apartments, y_hat_gbm)
mae_knn <- mae(y_true_apartments, y_hat_knn)


## WYKRES Y vs. Y_hat
## pozwala oszacować dla jakich wartości y model działa dobrze/najepiej

plot(y = y_hat_rf, x = y_true_apartments, xlab = "True values", ylab = "Predictions")
lines(c(2000,6000), c(2000,6000), col = "red")

## REZYDUA
## R(y, \hat{y}) = y - \hat{y}
## wykres rezyduów

plot(y = y_true_apartments - y_hat_rf, x = y_true_apartments, ylab = "Residuals", xlab = "Predictions")
lines(c(1000, 7000), c(0,0))

### ZADANIE KLASYFIKACJI - ZBIÓR TITANIC
y_true_titanic <- as.numeric(titanic_class_test$survived) - 1
## REGRESJA LOGISTYCZNA
logreg_model <- glm(survived~., data = titanic_class_train, family = "binomial")
probs_logreg <- predict(logreg_model, titanic_class_test, type = "response")
names(probs_logreg) <- NULL

## DRZEWO DECYZYJNE
class_tree_model <- ctree(survived~., data = titanic_class_train)
probs_tree <- predict(class_tree_model, titanic_class_test, type = "prob")[,2]

## LAS LOSOWY
class_rf_model <- randomForest(survived~., data = titanic_class_train)
probs_rf <- predict(class_rf_model, titanic_class_test, type = "prob")[,2]

## BOOSTING
class_gbm_model <- gbm(as.character(survived)~., distribution = "bernoulli", data=titanic_class_train)
probs_gbm <- predict(class_gbm_model, titanic_class_test, type = "response")

## KNN
class_knn_model <- gknn(survived~., data=titanic_class_train, scale=TRUE, k=5)
probs_knn <- predict(class_knn_model, titanic_class_test)

class_model_names <- c("logreg", "tree", "rf", "gbm", "knn")

#####################################
### EWALUACJA MODELI KLASYFIKACJI ###
#####################################

# CEL: średnio jak często popełniamy różnego rodzaju błędy
#
# Oznaczenia:
# probs  - predykcje modeli - prawdopodobieństwo przynależności do klasy 1 (przeżycie)
# y_true - wartości prawdziwe

### Macierz pomyłek
# TP - true positive - model przewiduje 1, prawdziwa wartość 1
# TN - true negative - model przewiduje 0, prawdziwa wartość 0
# FP - false positive - model przewiduje 1, prawdziwa wartość 0
# FN - false negative - model przewiduje 0, prawdziwa wartość 1

model_response <- (probs_rf > 0.5)
TP <- sum(y_true_titanic == 1  & model_response == 1)
TP
TN <- sum(y_true_titanic == 0  & model_response == 0)
TN
FP <- sum(y_true_titanic == 0  & model_response == 1)
FP
FN <- sum(y_true_titanic == 1  & model_response == 0)
FN

ModelMetrics::confusionMatrix(y_true_titanic, probs_rf)

### Dla tego przypadku nasza macierz ma postać,
### wiele zależy od kodowania klas i przyjętej konwencji
###  TN | FN
### ----+----
###  FP | TP

## Na podstawie macierzy możemy wyznaczyć następujące metryki


# accuracy - jaką część obserwacji sklasyfikowaliśmy poprawnie
rf_accuracy <- (TN + TP) / (TN + TP + FP + FN)
rf_accuracy

# precision - jaką część spośród obserwacji sklasyfikowanych przez nas jako pozytywne jest prawdziwa
rf_precision <- (TP) / (TP + FP)
rf_precision

ModelMetrics::precision(y_true_titanic, probs_rf)

# sensitivity/recall - jaką część spośród obserwacji pozytywnych znaleźliśmy
rf_sensitivity <- (TP) / (TP + FN)
rf_sensitivity

ModelMetrics::sensitivity(y_true_titanic, probs_rf)

# f1 - średnia ważona precision i recall

rf_f1 <- 2*rf_precision*rf_sensitivity/(rf_precision + rf_sensitivity)
rf_f1

f1Score(y_true_titanic, probs_rf)

## AUC
auc(y_true_titanic, probs_rf)


###############################################
### SELEKCJA ZMIENNYCH DLA MODELI LINIOWYCH ###
###############################################


## METODY KROKOWE DLA MODELI ZAGNIEŹDŹONYCH
baseline_lm <- lm(m2.price ~ ., data = apartments_regr_train)
step(baseline_lm, direction = "backward", k = 2) -> selected_lm

y_hat_selected <- predict(selected_lm, apartments_regr_test)

rmse(y_hat_selected, y_true_apartments)
rmse(y_hat_linreg, y_true_apartments)

## REGRESJA GRZBIETOWA

library(caret)

apartments_encoder <- dummyVars(~., apartments_regr_train)
numerical_apartments_train <- predict(apartments_encoder, apartments_regr_train)
numerical_apartments_test <- predict(apartments_encoder, apartments_regr_test)


cv.glmnet(numerical_apartments_train[,-1],
          numerical_apartments_train[,1]) -> lasso_lm

coef(lasso_lm)
y_hat_lasso <- predict(lasso_lm, numerical_apartments_test[,-1])

rmse(y_hat_lasso, y_true_apartments)
rmse(y_hat_linreg, y_true_apartments)

mae(y_hat_lasso, y_true_apartments)
mae(y_hat_linreg, y_true_apartments)

## DOBÓR HIPERPARAMETRÓW I KROSWALIDACJA

set.seed(42)

# RANDOM SEARCH

fitControl <- trainControl(method = "repeatedcv",
                           number = 3,
                           repeats = 5,
                           search = "random")


gbm_model_random_search <- train(m2.price~ .,
                          data = numerical_apartments_train,
                          distribution = "gaussian",
                          method = "gbm",
                          trControl = fitControl,
                          verbose = TRUE,
                          tuneLength = 6)


predict(gbm_model_random_search, numerical_apartments_test) -> rs_predictions

rmse(numerical_apartments_test[,1], rs_predictions)
rmse(y_true_apartments, y_hat_gbm)

## GRID SEARCH

fitControl <- trainControl(method = "repeatedcv",
                           number = 3,
                           repeats = 5,
                           search = "grid")

gbm_grid <- expand.grid(n.trees = c(100, 200, 250),
                        interaction.depth = c(1, 4, 6),
                        shrinkage = 0.1,
                        n.minobsinnode = 10)

gbm_model_grid_search <- train(m2.price~ .,
                                 data = numerical_apartments_train,
                                 distribution = "gaussian",
                                 method = "gbm",
                                 trControl = fitControl,
                                 verbose = TRUE,
                                 tuneGrid = gbm_grid)


predict(gbm_model_grid_search, numerical_apartments_test) -> gs_predictions

rmse(numerical_apartments_test[,1], gs_predictions)
rmse(y_true_apartments, y_hat_gbm)

