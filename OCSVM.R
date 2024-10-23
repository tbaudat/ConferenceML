library(e1071)    # Pour le SVM
library(caret)    # Pour la matrice de confusion
library(dplyr)
library(FactoMineR)

# Charger et prétraiter les données
data_reduit <- read.csv("data_cancer.csv", header=T)
data_reduit$Diagnosis <- as.factor(data_reduit$Diagnosis)

set.seed(123)
# Générer un vecteur d'index aléatoires pour l'ensemble de test (30%)
test_index <- sample(1:378, size = round(0.3 * 378))

# Créer les ensembles d'entraînement (70%) et de test (30%)
train_data <- data_reduit[-test_index, ]
summary(train_data)
#write.csv(x = train_data, "train.csv")
test_data <- data_reduit[test_index, ]
summary(test_data)
#write.csv(x = test_data, "test.csv")

data_reduit <- train_data

# Modèle de classification

mod <- glm(data=data_reduit, Diagnosis~., family='binomial')
#erreur de convergence

##########
# Optimisation du modèle à l'aide des hyperparamètres
##########

# Définir la grille d'hyperparamètres pour nu et gamma
nu_values <- seq(0.02, 0.11, by = 0.01)    # Exemple de valeurs pour nu
gamma_values <- c(0.001,0.005,0.01,0.03,0.05,0.1,1)  # Exemple de valeurs pour gamma

# Stocker les résultats finaux
results <- data.frame(nu = numeric(), gamma = numeric(), balanced_accuracy = numeric(), Kappa = numeric())

# Boucle sur les valeurs de gamma
for (gamma in gamma_values) {
  
  # Boucle sur les valeurs de nu pour chaque gamma
  for (nu in nu_values) {
    
    # Variable pour stocker les vraies valeurs et les prédictions pour tout le dataset
    all_true_labels <- data_reduit$Diagnosis
    all_predicted_scores <- c()
    
    # Créer les folds pour Diagnosis == 0
    set.seed(123)
    folds_0 <- createFolds(data_reduit$Diagnosis[data_reduit$Diagnosis == 0], k = 5)
    
    # Créer les folds pour Diagnosis == 1
    set.seed(123)
    folds_1 <- createFolds(data_reduit$Diagnosis[data_reduit$Diagnosis == 1], k = 5)
    
    # Boucle pour chaque fold (de 1 à 10)
    for (fold in 1:5) {
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
      
      # Entraînez le modèle sur les données Diagnosis == 0 uniquement
      set.seed(123)
      X_train <- train_fold_0[, -which(names(train_fold_0) == "Diagnosis")]
      model <- svm(X_train, type = "one-classification", nu = nu, gamma = gamma)
      
      # Prédire sur le jeu de test
      X_test <- test_fold[, -which(names(test_fold) == "Diagnosis")]
      decision_scores <- predict(model,X_test)
      
      predicted_labels <- ifelse(decision_scores == TRUE, 0, 1)
      # Collecter les vraies valeurs et les scores de prédiction
      all_predicted_scores <- c(all_predicted_scores, predicted_labels)
    }

    # Définir les indices associés aux labels prédits
    indices <- as.numeric(names(all_predicted_scores))  # Récupérer les indices (de 1 à 400)
    
    # Trier les labels prédits selon les indices
    sorted_indices <- order(indices)  # Obtenir les indices triés
    sorted_predicted_scores <- all_predicted_scores[sorted_indices]  # Réorganiser les labels prédits
    
    # Afficher les indices et les labels triés
    sorted_data <- data.frame(Indices = indices[sorted_indices], Labels = sorted_predicted_scores)
    # Créer des facteurs avec les mêmes niveaux pour éviter l'erreur
    all_true_labels <- factor(all_true_labels, levels = c(0, 1))
    sorted_data$Labels <- factor(sorted_data$Labels, levels = c(0, 1))
    
    #Stockage de la matrice de confusion
    conf_matrix <- confusionMatrix(sorted_data$Labels, all_true_labels)
    print(conf_matrix)
    
    # Extraire la balanced accuracy et le kappa
    balanced_accuracy_value <- conf_matrix$byClass['Balanced Accuracy']
    kappa_value <- conf_matrix$overall['Kappa']
    
    # Ajouter les résultats dans le dataframe
    results <- rbind(results, data.frame(nu = nu, gamma = gamma, balanced_accuracy = balanced_accuracy_value, Kappa = kappa_value))
  }
}

# Affichage des résultats finaux
print(results)

#Affichage de la plus haute valeur 
max(results$balanced_accuracy)

# Heatmap de balanced_accuracy
library(ggplot2)
ggplot(results, aes(x = factor(nu), y = factor(gamma), fill = balanced_accuracy)) +
  geom_tile() +
  scale_fill_gradient2(high = "red", mid = "yellow", low = "darkblue", midpoint = 0.55) +
  labs(title = "Heatmap du balanced accuracy",
       x = "Valeurs de nu",
       y = "Valeurs de gamma",
       fill = "balanced accuracy") +
  theme_minimal()


##########
# Evaluation du modèle avec les meilleurs hyperparamètres
##########

all_true_labels <- data_reduit$Diagnosis
all_predicted_labels <- c()

set.seed(123)
# Créer les folds pour Diagnosis == 0
folds_0 <- createFolds(data_reduit$Diagnosis[data_reduit$Diagnosis == 0], k = 5)
set.seed(123)
# Créer les folds pour Diagnosis == 1
folds_1 <- createFolds(data_reduit$Diagnosis[data_reduit$Diagnosis == 1], k = 5)

# Boucle pour chaque fold (de 1 à 5)
for (fold in 1:5) {
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
  set.seed(123)
  # Combine les données d'entraînement et de test pour les deux groupes
  test_fold <- rbind(test_fold_0, test_fold_1)
  # Entraînez le modèle sur les données Diagnosis == 0 uniquement
  X_train <- train_fold_0[, -which(names(train_fold_0) == "Diagnosis")]
  model <- svm(X_train, type = "one-classification", nu=0.09, gamma=0.03)
  
  # Prédire sur le jeu de test
  X_test <- test_fold[, -which(names(test_fold) == "Diagnosis")]
  decision_scores <- predict(model,X_test)
  
  predicted_labels <- ifelse(decision_scores == TRUE, 0, 1)
  # Collecter les vraies valeurs et les scores de prédiction
  all_predicted_labels <- c(all_predicted_labels, predicted_labels)
}

# Définir les indices associés aux labels prédits
indices <- as.numeric(names(all_predicted_labels))  # Récupérer les indices (de 1 à 400)

# Trier les labels prédits selon les indices
sorted_indices <- order(indices)  # Obtenir les indices triés
sorted_predicted_labels <- all_predicted_labels[sorted_indices]  # Réorganiser les labels prédits

# Afficher les indices et les labels triés
sorted_data <- data.frame(Indices = indices[sorted_indices], Labels = sorted_predicted_labels)
# Créer des facteurs avec les mêmes niveaux pour éviter l'erreur
all_true_labels <- factor(all_true_labels, levels = c(0, 1))
sorted_data$Labels <- factor(sorted_data$Labels, levels = c(0, 1))

# Calculer la matrice de confusion
conf_matrix <- confusionMatrix(sorted_data$Labels, all_true_labels)
print(conf_matrix)

#############
#ACP
#############

data_reduit$Predicted <- sorted_data$Labels

res.PCA<-PCA(data_reduit,quali.sup=c(1,32),graph=FALSE)
plot.PCA(res.PCA,invisible=c('quali'),habillage=32,title="Anomalies prédites",label ='none', col.hab=c("grey","black"))
plot.PCA(res.PCA,invisible=c('quali'),habillage=1,title="Anomalies observées",label ='none', col.hab=c("grey","black"))
