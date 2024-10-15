<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a id="readme-top"></a>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

La détection d'anomalies est un enjeu crucial dans des domaines variés, notamment en médecine, où elle permet de repérer des cas rares qui pourraient échapper aux méthodes classiques de classification. Ce travail qui a été réalisé pour la conférence s’inscrit dans le contexte de la détection d'anomalies des données relatives au cancer du sein. Le jeu de données utilisé présente une forte asymétrie entre les cellules bénignes (présente en grande quantité) et les cellules malignes (en faible quantité). Ce déséquilibre rend compliqué l’application d'algorithmes de classification. Pour surmonter cette difficulté, nous avons eu recours à des techniques de détection d'anomalies, qui sont adaptées à des jeux de données déséquilibrés.

Le jeu de données utilisé est composé d’informations liées à des images numérisées d'une aspiration à l'aiguille fine (FNA) d'une masse mammaire de différents patients. Les variables du jeu de données sont des caractéristiques géométriques des différents noyaux cellulaires présents sur les images. Les cellules malignes représentent 6% du jeu de données. Chaque individu ayant cette modalité est considéré comme une anomalie au sens statistique. L’objectif est de comparer les performances de plusieurs méthodes de détection d’anomalies en termes de spécificité et d’aire sous la courbe ROC (AUC) afin de déterminer la méthode qui est la plus efficace pour ce type de données fortement déséquilibrées. 

Nous avons envisagé quatre approches de machine learning pour cette tâche : le Local Outlier Factor (LOF), le One-Class Support Vector Machine (OCSVM), DBScan et l'Isolation Forest. La méthode DBScan qui utilise la proximité à d’autres points et crée des clusters n’a finalement pas été utilisée car elle est moins performante sur des jeux de données à nombreuses dimensions Ces algorithmes ont été sélectionnés pour leur capacité à identifier des individus présents en quantité minoritaire dans un jeu de données complexes. Le LOF utilise la densité locale pour détecter les anomalies en fonction du voisinage des données, tandis que l’OCSVM projette les données dans un espace de grande dimension pour isoler les points anormaux. Enfin, l’Isolation Forest est un algorithme basé sur des arbres de décision qui identifie les anomalies en fractionnant les données de manière aléatoire et donc en les isolant. 

Les résultats obtenus pourront permettre une détection plus précoce des cancers. Cela pourrait ouvrir la voie à des systèmes de diagnostic automatisés plus fiables et robustes. Cependant, toutes les méthodes envisagées nécessitent des hyperparamètres qui forcent donc une intervention humaine. 
<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTACT -->
## Contact

Paola Andrieu  - paola.andrieu@agrocampus-ouest.fr

Augustin Robert  - augustin.robert@agrocampus-ouest.fr

Timéo Baudat  - timeo.baudat@agrocampus-ouest.fr

<p align="right">(<a href="#readme-top">back to top</a>)</p>

