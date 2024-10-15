library(e1071)    # Pour le SVM
library(pROC)     # Pour la courbe ROC
library(caret)    # Pour la matrice de confusion
library(dplyr)

# Charger et prétraiter les données
url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
columns <- c('ID', 'Diagnosis', paste0('feature_', 1:30))
data <- read.csv(url, header = FALSE, col.names = columns)
data <- data %>% select(-ID)
data$Diagnosis <- ifelse(data$Diagnosis == "M", 1, 0)
summary(data)
data$Diagnosis <- as.factor(data$Diagnosis)


set.seed(123)
modalite_1 <- data %>% filter(Diagnosis == 1)
n_to_keep <- round(nrow(modalite_1) * 0.1)
modalite_1_sample <- modalite_1 %>% sample_n(n_to_keep)
data_reduit <- bind_rows(modalite_1_sample, data %>% filter(Diagnosis != 1))
data_reduit$Diagnosis <- as.factor(data_reduit$Diagnosis)
summary(data_reduit)
head(data_reduit)


# Définir la grille d'hyperparamètres pour nu et gamma
nu_values <- seq(0.01, 0.1, by = 0.01)    # Exemple de valeurs pour nu
gamma_values <- c(0.001,0.005,0.01,0.03,0.05,0.1,1)  # Exemple de valeurs pour gamma

# Stocker les résultats finaux
results <- data.frame(nu = numeric(), gamma = numeric(), AUC = numeric(), Specificity = numeric())

# Boucle sur les valeurs de gamma
for (gamma in gamma_values) {
  
  # Boucle sur les valeurs de nu pour chaque gamma
  for (nu in nu_values) {
    
    # Variable pour stocker les vraies valeurs et les prédictions pour tout le dataset
    all_true_labels <- data_reduit$Diagnosis
    all_predicted_scores <- c()
    
    # Créer les folds pour Diagnosis == 0
    folds_0 <- createFolds(data_reduit$Diagnosis[data_reduit$Diagnosis == 0], k = 10)
    
    # Créer les folds pour Diagnosis == 1
    folds_1 <- createFolds(data_reduit$Diagnosis[data_reduit$Diagnosis == 1], k = 10)
    
    # Boucle pour chaque fold (de 1 à 10)
    for (fold in 1:10) {
      # Indices d'entraînement et de test pour Diagnosis == 0
      train_indices_0 <- unlist(folds_0[-fold])
      test_indices_0 <- unlist(folds_0[fold])
      
      # Indices d'entraînement et de test pour Diagnosis == 1
      train_indices_1 <- unlist(folds_1[-fold])
      test_indices_1 <- unlist(folds_1[fold])
      
      # Créer les jeux d'entraînement et de test
      train_fold_0 <- data_reduit[train_indices_0, ]
      test_fold_0 <- data_reduit[test_indices_0, ]
      
      train_fold_1 <- data_reduit[train_indices_1, ]
      test_fold_1 <- data_reduit[test_indices_1, ]
      
      # Combine les données d'entraînement et de test pour les deux groupes
      test_fold <- rbind(test_fold_0, test_fold_1)
      
      # Entraînez le modèle sur les données Diagnosis == 0 uniquement (comme avant)
      X_train <- train_fold_0[, -which(names(train_fold_0) == "Diagnosis")]
      model <- svm(X_train, type = "one-classification", nu = nu, gamma = gamma)
      
      # Prédire sur le jeu de test
      X_test <- test_fold[, -which(names(test_fold) == "Diagnosis")]
      decision_scores <- attributes(predict(model, X_test, decision.values = TRUE))$decision.values
      
      # Collecter les vraies valeurs et les scores de prédiction (comme avant)
      all_predicted_scores <- c(all_predicted_scores, decision_scores)
    }
    
    # Calculer la courbe ROC et l'AUC globalement
    roc_curve <- roc(all_true_labels, all_predicted_scores)
    auc_value <- auc(roc_curve)
    
    # Prédictions binaires basées sur les scores pour calculer la spécificité
    predicted_labels <- ifelse(all_predicted_scores < 0, 0, 1)
    
    # Créer des facteurs avec les mêmes niveaux pour éviter l'erreur
    predicted_labels <- factor(predicted_labels, levels = c(0, 1))
    all_true_labels <- factor(all_true_labels, levels = c(0, 1))
    
    # Calculer la matrice de confusion
    conf_matrix <- confusionMatrix(predicted_labels, all_true_labels)
    
    print(conf_matrix)
    
    # Extraire la spécificité globale
    specificity_value <- conf_matrix$byClass['Specificity']
    
    # Ajouter les résultats dans le dataframe
    results <- rbind(results, data.frame(nu = nu, gamma = gamma, AUC = auc_value, Specificity = specificity_value))
  }
}

# Affichage des résultats finaux
print(results)

# Heatmap de l'AUC
library(ggplot2)
ggplot(results, aes(x = factor(nu), y = factor(gamma), fill = AUC)) +
  geom_tile() +
  scale_fill_gradient2(high = "red", mid = "yellow", low = "green", midpoint = max(results$AUC)/2) +
  labs(title = "Heatmap de l'AUC",
       x = "Valeurs de nu",
       y = "Valeurs de gamma",
       fill = "AUC") +
  theme_minimal()

# Heatmap de la Spécificité
ggplot(results, aes(x = factor(nu), y = factor(gamma), fill = Specificity)) +
  geom_tile() +
  scale_fill_gradient2(high = "red", mid = "yellow", low = "green", midpoint = max(results$Specificity)/2) +
  labs(title = "Heatmap de la Spécificité",
       x = "Valeurs de nu",
       y = "Valeurs de gamma",
       fill = "Spécificité") +
  theme_minimal()



all_true_labels <- data_reduit$Diagnosis
all_predicted_scores <- c()
# Créer les folds pour Diagnosis == 0
folds_0 <- createFolds(data_reduit$Diagnosis[data_reduit$Diagnosis == 0], k = 10)

# Créer les folds pour Diagnosis == 1
folds_1 <- createFolds(data_reduit$Diagnosis[data_reduit$Diagnosis == 1], k = 10)

# Boucle pour chaque fold (de 1 à 10)
for (fold in 1:10) {
  # Indices d'entraînement et de test pour Diagnosis == 0
  train_indices_0 <- unlist(folds_0[-fold])
  test_indices_0 <- unlist(folds_0[fold])
  
  # Indices d'entraînement et de test pour Diagnosis == 1
  train_indices_1 <- unlist(folds_1[-fold])
  test_indices_1 <- unlist(folds_1[fold])
  
  # Créer les jeux d'entraînement et de test
  train_fold_0 <- data_reduit[train_indices_0, ]
  test_fold_0 <- data_reduit[test_indices_0, ]
  
  train_fold_1 <- data_reduit[train_indices_1, ]
  test_fold_1 <- data_reduit[test_indices_1, ]
  
  # Combine les données d'entraînement et de test pour les deux groupes
  test_fold <- rbind(test_fold_0, test_fold_1)
  
  # Entraînez le modèle sur les données Diagnosis == 0 uniquement (comme avant)
  X_train <- train_fold_0[, -which(names(train_fold_0) == "Diagnosis")]
  model <- svm(X_train, type = "one-classification", nu=0.02, gamma=0.01)
  
  # Prédire sur le jeu de test
  X_test <- test_fold[, -which(names(test_fold) == "Diagnosis")]
  decision_scores <- attributes(predict(model, X_test, decision.values = TRUE))$decision.values
  
  # Collecter les vraies valeurs et les scores de prédiction (comme avant)
  all_predicted_scores <- c(all_predicted_scores, decision_scores)
}

# Calculer la courbe ROC et l'AUC globalement
roc_curve <- roc(all_true_labels, all_predicted_scores)
auc_value <- auc(roc_curve)

# Prédictions binaires basées sur les scores pour calculer la spécificité
predicted_labels <- ifelse(all_predicted_scores < 0, 0, 1)

# Créer des facteurs avec les mêmes niveaux pour éviter l'erreur
predicted_labels <- factor(predicted_labels, levels = c(0, 1))
all_true_labels <- factor(all_true_labels, levels = c(0, 1))

# Calculer la matrice de confusion
conf_matrix <- confusionMatrix(predicted_labels, all_true_labels)

print(conf_matrix)

# Extraire la spécificité globale
specificity_value <- conf_matrix$byClass['Specificity']

print(paste("AUC: ", auc_value))
print(paste("Spécificité: ", specificity_value))


library(FactoMineR)

data_reduit$Predicted <- as.factor(predicted_labels)


res.PCA<-PCA(data_reduit,quali.sup=c(1,32),graph=FALSE)
plot.PCA(res.PCA,invisible=c('ind.sup','quali'),habillage=32,title="Graphe des individus de l'ACP",label ='none')
plot.PCA(res.PCA,invisible=c('ind.sup','quali'),habillage=1,title="Graphe des individus de l'ACP",label ='none')
