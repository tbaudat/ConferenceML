#LOF

# import donnees ----

library(tidyr)

url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
columns <- c('ID', 'Diagnosis', paste0('feature_', 1:30))
data <- read.csv(url, header = FALSE, col.names = columns)
data <- data %>% select(-ID)
data$Diagnosis <- ifelse(data$Diagnosis == "M", 1, 0)
summary(data)
data$Diagnosis <- as.factor(data$Diagnosis)

library(dplyr)
set.seed(123)
modalite_1 <- data %>% filter(Diagnosis == 1)
n_to_keep <- round(nrow(modalite_1) * 0.1)
modalite_1_sample <- modalite_1 %>% sample_n(n_to_keep)
data_reduit <- bind_rows(modalite_1_sample, data %>% filter(Diagnosis != 1))
data_reduit$Diagnosis <- as.factor(data_reduit$Diagnosis)
summary(data_reduit)

# LOF ----

library(DescTools)

Prediction<- function(seuil, k){
  
  #Ici la fonction Prediction assigne a chaque point un score LOF
  
  Local_density <- LOF(data = data_reduit[,3:31], k = k)
  
  #On ne garde alors que les points ayant un score supérieur à un certain seuil
  outliers <- which(Local_density>seuil)
  
  
  #on stocke alors les resultats dans une variable du jeu de donnees
  data_reduit$prediction <- NA
  data_reduit$prediction[outliers] <- 1
  data_reduit$prediction[-outliers] <- 0
  data_reduit$prediction <- as.factor(data_reduit$prediction)
  return(data_reduit$prediction)
}

Specificite<- function(seuil, k){
  #Cette fonction retourne la specificite
  pred <- Prediction(seuil,k)
  tableau <- table(data_reduit$Diagnosis, pred)
  Spec <- tableau[2,2]/(tableau[2,1]+tableau[2,2])
  return(Spec)
}

#recherche des hyperparametres optimaux ----

#en observant les jeux de donnees, on etablit des valeurs autour desquelles
#les hyperparametres devraient se trouver
#on va tester ici des k entre 1 et 10
#et des seuils entre 1 et 2

parametres <- matrix(rep(NA, 15*15), nrow = 15, ncol = 15)
seuils <- seq(from = 1.1, to = 2.5, length = 15)

#On choisit ici la specificite comme metrique de comparaison

for (k in 1:15){
  for (s in 1:15){
    seuil <- seuils[s]
    parametres[k,s] <- Specificite(k=k,seuil=seuil)
  }
}

colnames(parametres) <- seuils
rownames(parametres) <- c(1:15)
heatmap(parametres, Rowv = NA, Colv = NA, 
        main = "Specificite en fonction des hyperparametres", 
        xlab = "seuil", ylab = "k")

#on va chercher une valeur plus precise pour le seuil

library(pROC)

AUC <- matrix(rep(NA, times = 100), nrow = 10, ncol = 10)
Spe <- matrix(rep(NA, times = 100), nrow = 10, ncol = 10)
Sen <- matrix(rep(NA, times = 100), nrow = 10, ncol = 10)

seuils <- seq(from = 1.11, to = 1.2, length = 10)

for (k in 1:10){
  for (s in 1:10){
    
    Local_density <- LOF(data = data_reduit[,3:31], k = k+4)
    outliers <- which(Local_density>seuils[s])
    
    data_reduit$prediction <- NA
    data_reduit$prediction[outliers] <- 1
    data_reduit$prediction[-outliers] <- 0
    
    ROC <- roc(as.numeric(data_reduit$Diagnosis)-1, data_reduit$prediction)
    AUC[k,s] <- auc(ROC)
    Spe[k,s] <- ROC$specificities[2]
    Sen[k,s] <- ROC$sensitivities[2]
  }
}

colnames(AUC) <- seuils
colnames(Spe) <- seuils
colnames(Sen) <- seuils

rownames(AUC) <- c(5:14)
rownames(Spe) <- c(5:14)
rownames(Sen) <- c(5:14)

heatmap(AUC, Rowv = NULL, Colv = NULL, main = "AUC")
heatmap(Spe, Rowv = NULL, Colv = NULL, main = "Specificite")
heatmap(Sen, Rowv = NULL, Colv = NULL, main = "Sensitivite")

Specificite(seuil = 1.13, k = 14)


# Cross-Validation ----

segments <- cvsegments(N=378, k=3)
cvpred <- rep(0, times=378)
for(k in 1:3){
  petit <- data_reduit[segments[[k]],]
  
  cvpred[segments[[k]]] <- 
  Local_density <- LOF(data = petit[,2:31], k = 2)
  outliers <- which(Local_density>1.9)
  pred_petit <- data_reduit[outliers, 1]
  
  prediction <- rep(NA, times = length(segments[[k]]))
  prediction[outliers] <- 1
  prediction[-outliers] <- 0
  cvpred[segments[[k]]] <- prediction
}

table(cvpred, data_reduit$Diagnosis)
rmsep

#Resultats avec hyperprametres trouves

Local_density <- LOF(data = data_reduit[,3:31], k = 14)
outliers <- which(Local_density>1.13)

data_reduit$prediction <- NA
data_reduit$prediction[outliers] <- 1
data_reduit$prediction[-outliers] <- 0

ROC <- roc(as.numeric(data_reduit$Diagnosis)-1, data_reduit$prediction)
AUC <- auc(ROC)
Spe <- ROC$specificities[2]
Sen <- ROC$sensitivities[2]


#Presentation des resultats sous forme graphique ----

library(FactoMineR)

data_reduit$prediction <- as.factor(data_reduit$prediction)
ACP <- PCA(data_reduit[1:33], quali.sup = c(1, 33), quanti.sup = 32, 
           graph = FALSE)
plot(ACP, choix = "ind", habillage = 1, label = "ind.sup",
     title = "Veritables Anomalies")
plot(ACP, choix = "ind", habillage = 33, label = "ind.sup",
     title = "Anomalies Predites")

plot(ACP, choix = "ind", habillage = 32, label = "ind.sup",
     title = "Anomalies Predites")

