library(tidyverse)
library(caret)
library(isotree)
library(pROC)

### Import des données et prétraitement ----
data_reduit <- read.csv("data_cancer.csv", header=T)
data_reduit$Diagnosis <- as.factor(data_reduit$Diagnosis)
summary(data_reduit$Diagnosis)

# Normalisation des données
preProc <- preProcess(data_reduit %>% select(-Diagnosis), method = c("center", "scale"))
data_normalized <- predict(preProc, data_reduit %>% select(-Diagnosis))
data_normalized$Diagnosis <- data_reduit$Diagnosis
dta <- data_normalized[,-1]

### Recherche des hyperparamètres ----
# Hyperparamètres de la fonction isolation.forest : 
#   - sample_size
#   - ntrees
set.seed(123)
sampsiz <- seq(300,400,length = 10)
ntrees <- seq(400,600, length = 10)
res <- matrix(0, nrow = length(sampsiz), ncol = length(ntrees))
for (ss in 1:length(sampsiz)){
  for (nt in 1:length(ntrees)){
    model <- isolation.forest(dta, sample_size = sampsiz[ss], ntrees = ntrees[nt])
    predictions <- predict(model, newdata = dta)
    thresholds <- seq(0, 1, by = 0.01)
    BA <- sapply(thresholds, function(threshold) {
      final_anomaly_predictions <- ifelse(predictions > threshold, 1, 0)
      confusion_mat <- confusionMatrix(factor(final_anomaly_predictions), data_normalized$Diagnosis)
      Balanced_Accuracy = confusion_mat$byClass['Balanced Accuracy']
    })
    res[ss, nt] <- max(BA)
    print(ss,nt)
  }
}

library(ggplot2)
library(reshape2)  # pour la fonction melt

# Convertir la matrice res en data frame pour ggplot
df_res <- data.frame(sampsiz = rep(sampsiz, each = length(ntrees)),
                     ntrees = rep(ntrees, length(sampsiz)),
                     BA = as.vector(res))

# Tracer un heatmap avec ggplot
ggplot(df_res, aes(x = sampsiz, y = ntrees, fill = BA)) +
  geom_tile() +
  scale_fill_gradient2(low = "blue", high = "red", mid = 'yellow',midpoint = 0.932) +
  labs(title = "Balanced Accuracy selon sample_size et ntrees",
       x = "Sample Size",
       y = "Number of Trees",
       fill = "Balanced Accuracy") +
  theme_minimal()

max_value <- max(res)
indices <- which(res == max_value, arr.ind = TRUE)
best_sampsiz <- sampsiz[indices[1, 1]]
best_ntrees <- ntrees[indices[1, 2]]

cat("La plus grande Balanced Accuracy est:", max_value, "\n")
cat("Elle est obtenue pour sample_size =", best_sampsiz, "et ntrees =", best_ntrees, "\n")


### Création de modèle et prédiction ----
# Création du modèle 
model <- isolation.forest(dta, ntrees = best_ntrees, sample_size = best_sampsiz)

# Prédire les anomalies
predictions <- predict(model, newdata = dta)


# Recherche d'un seuil idéal
thresholds <- seq(0, 1, by = 0.01)
BA <- sapply(thresholds, function(threshold) {
  final_anomaly_predictions <- ifelse(predictions > threshold, 1, 0)
  confusion_mat <- confusionMatrix(factor(final_anomaly_predictions), data_normalized$Diagnosis)
  Balanced_Accuracy = confusion_mat$byClass['Balanced Accuracy']
})

metrics_df <- data.frame(Threshold = thresholds, Balanced_Accuracy = BA)
summary(metrics_df)

ggplot(metrics_df, aes(x = Threshold)) +
  geom_line(aes(y = Balanced_Accuracy, color = "Balanced_Accuracy"), size = 1) +
  labs(title = "Balanced Accuracy en fonction du seuil", x = "Seuil de décision", y = "Valeur") +
  scale_color_manual(values =c("Balanced_Accuracy" = "black")) +
  theme_minimal()

# Choix du seuil en fonction de la meilleure Balanced Accuracy
seuil <- metrics_df$Threshold[which.max(metrics_df$Balanced_Accuracy)]

# Stockage des données prédites
data_normalized$Predictions <- factor(ifelse(predictions > seuil, 1, 0))

### Visualisation des scores d'Anomalie ----
data_normalized$Anomaly_Score <- predictions

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

# Matrice de confusion
confusion_mat <- confusionMatrix(data_normalized$Predictions, data_normalized$Diagnosis)
print(confusion_mat)
confusion_mat$overall





### Graphes des individus issus d'une ACP ----
library(FactoMineR)

res.PCA<-PCA(data_normalized,quali.sup=c(31:33),graph=FALSE)
plot.PCA(res.PCA,invisible=c('quali'),habillage=31,title="Anomalies Observées",label ='none', col.hab = c("grey","black"))


res.PCA<-PCA(data_normalized,quali.sup=c(31:33),graph=FALSE)
plot.PCA(res.PCA,invisible=c('quali'),habillage=32,title="Anomalies Prédites",label ='none',col.hab = c("grey","black"))

