#LOF

# import donnees ----

library(tidyr)
library(dplyr)


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

# LOF ----

library(DescTools)

#Principe de la fonction LOF

Prediction<- function(seuil, k){
  
  #Ici la fonction Prediction assigne a chaque point un score LOF
  #correspondant à la densité locale des points
  # a savoir : ont-ils plus ou moins de voisins proches que leurs voisins
  #proches? 
  
  Local_density <- LOF(data = data_reduit[,2:31], k = k)
  
  #On ne garde alors que les points ayant un score supérieur à un certain seuil
  outliers <- which(Local_density>seuil)
  
  #on stocke alors les resultats dans une variable du jeu de donnees
  prediction <- rep(NA, times = length(data_reduit$Diagnosis))
  prediction[outliers] <- 1
  prediction[-outliers] <- 0
  prediction <- as.factor(prediction)
  return(prediction)
}


#recherche des hyperparametres optimaux ----

#en observant les jeux de donnees, on etablit des valeurs autour desquelles
#les hyperparametres devraient se trouver
#on va tester ici des k entre 1 et 10
#et des seuils entre 1 et 2
library(caret)
library(pls)
library(fields)

BalAcc <- function(k, seuil){
  
  #Cette fonction mesure la balanced Accuracy
  #BalAcc  = (Specificite + Sensitivite)/2
  
  pred <- Prediction(seuil = seuil, k= k)
  Res <- confusionMatrix(data_reduit$Diagnosis, pred)
  BalAcc <- Res$byClass[11]
  return(BalAcc)
}

Bal_Accuracy <- matrix(rep(NA, times = 15*10), nrow = 10, ncol = 15)

# On cree une liste de seuils a tester. 
seuils <- seq(from = 1.1, to = 2.5, length = 15)

for (k in 1:10){
  for (s in 1:15){
    BA <- BalAcc(k=k, seuil=seuils[s])
    Bal_Accuracy[k,s] <- BA
  }
}

#on represente graphiquement les valeurs de balanced accuracy et de kappa
#en fonction des parametres

colnames(Bal_Accuracy) <- seuils
rownames(Bal_Accuracy) <- c(1:10)


image.plot(1:10,seuils,Bal_Accuracy,xlab="k",
           ylab="seuil",
           main="Balanced Accuracy en fonction des paramètres",
           cex.lab=1.25,cex.axis=1.25,cex.main=1.25)

BalAcc(k=10, seuil = 2.2)

table(data_reduit$Diagnosis, Prediction(seuil = 1.96, k= 9))

#Resultats avec hyperprametres trouves ----
library(pROC)

data_reduit$pred <- as.numeric(Prediction(seuil = 2.2, k=10))-1

ROC <- roc(as.numeric(data_reduit$Diagnosis)-1, data_reduit$pred)
AUC <- auc(ROC)
Spe <- ROC$specificities[2]
Sen <- ROC$sensitivities[2]

#Presentation des resultats sous forme graphique ----

library(FactoMineR)

data_reduit$pred <- as.factor(data_reduit$pred)

data_reduit[,33] <- LOF(data = data_reduit[,2:31], k = 10)
colnames(data_reduit)[33] <- "LOF"

ACP <- PCA(data_reduit[1:33], quali.sup = c(1, 32), quanti.sup = 33,
           graph = FALSE)

#les veritables "anomalies"
plot(ACP, choix = "ind", habillage = 1, label = "ind.sup",
     title = "Anomalies Observees", col.hab = c("grey", "black"), 
     invisible = c("ind.sup", "quali"))
#les anomalies predites
plot(ACP, choix = "ind", habillage = 32, label = "ind.sup",
     title = "Anomalies Predites", col.hab = c("grey", "black"), 
     invisible = c("ind.sup", "quali"))
#Le score LOF de chaque point
plot(ACP, choix = "ind", habillage = 33, label = "ind.sup",
     title = "Anomalies Predites", 
     invisible = c("ind.sup", "quali"))

confusionMatrix(table(data_reduit$Diagnosis, Prediction(seuil = 2.2, k= 10)))

