library(tidyverse)
library(caret)
library(isotree)
library(pROC)

# Import des données et prétraitement
url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
columns <- c('ID', 'Diagnosis', paste0('feature_', 1:30))
data <- read.csv(url, header = FALSE, col.names = columns)
data <- data %>% select(-ID)
data$Diagnosis <- ifelse(data$Diagnosis == "M", 1, 0)
data$Diagnosis <- as.factor(data$Diagnosis)

# Sélection des données intéressantes, création des anomalies
set.seed(123)
modalite_1 <- data %>% filter(Diagnosis == 1)
n_to_keep <- round(nrow(modalite_1) * 0.1)
modalite_1_sample <- modalite_1 %>% sample_n(n_to_keep)
data_reduit <- bind_rows(modalite_1_sample, data %>% filter(Diagnosis != 1))
summary(data_reduit$Diagnosis)

# Normalisation des données
preProc <- preProcess(data_reduit %>% select(-Diagnosis), method = c("center", "scale"))
data_normalized <- predict(preProc, data_reduit %>% select(-Diagnosis))
data_normalized$Diagnosis <- data_reduit$Diagnosis
dta <- data_normalized[,-1]

# Entraîner le modèle Isolation Forest
model <- isolation.forest(dta)

# Prédire les anomalies
predictions <- predict(model, newdata = dta)

# Évaluer la performance avec ROC
roc_result <- roc(as.numeric(data_normalized$Diagnosis) - 1, predictions)

# Afficher la courbe ROC
plot(roc_result, main = paste("Courbe ROC - Pli", k), col = "blue")
auc_value <- auc(roc_result)

final_anomaly_predictions <- ifelse(predictions > adjusted_threshold_value, 1, 0)
data_normalized$Predictions <- factor(final_anomaly_predictions)

confusion_mat <- confusionMatrix(data_normalized$Predictions, data_normalized$Diagnosis)
print(confusion_mat)

# Visualisation des scores d'Anomalie
data_normalized$Anomaly_Score <- final_predictions

# Histogramme
ggplot(data_normalized, aes(x = Anomaly_Score, fill = Diagnosis)) +
  geom_histogram(binwidth = 0.05, alpha = 0.7, position = "identity") +
  labs(title = "Distribution des Scores d'Anomalie",
       x = "Score d'Anomalie",
       y = "Nombre d'Instances") +
  theme_minimal()

# Densité
ggplot(data_normalized, aes(x = Anomaly_Score, color = Diagnosis)) +
  geom_density(alpha = 0.7) +
  labs(title = "Densité des Scores d'Anomalie",
       x = "Score d'Anomalie",
       y = "Densité") +
  theme_minimal()


library(FactoMineR)
library(Factoshiny)
res.PCA<-PCA(data_normalized,quali.sup=c(31:33),graph=FALSE)
plot.PCA(res.PCA,invisible=c('quali','ind.sup'),habillage=31,title="Véritables Anomalies",label ='none')


res.PCA<-PCA(data_normalized,quali.sup=c(31:33),graph=FALSE)
plot.PCA(res.PCA,invisible=c('quali','ind.sup'),habillage=32,title="Anomalies Prédites",label ='none')
