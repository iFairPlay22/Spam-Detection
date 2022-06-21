# Spam detection 🙄

## Le concept 😊

Mise au point d'un réseau de Deep Learning afin de catégoriser les spams.

## Hiérarchie du projet 🤔

- `./main.py` > Le programme python à executer 

- `./data.csv` > Le fichier contenant les données d'entrainement et de tests ;
	
- `./plots/*` > Les graphes générés ;
	
- `./models/*` > Les fichiers de modèle, permettant de stocker le résultat de l'entrainement ;
	
- `./vectorizer/*` > L'objet permettant de transformer la liste de mots en une liste d'entiers ;

## Faire tourner le projet 😁

Il y a **4 modes de lancement** : `clear`, `load`, `train` et `predict`. Afin de choisir le mode, il suffit de mettre à jour la liste `todo` du fichier `main.py`. On pourra ensuite lancer la programme via la commande `python main.py`. NB : Certaines variables situées dans le main permettent de **modifier facilement et rapidement les paramètres du réseau de neurones**. 

> `clear` : 
> - Prérequis :  Aucun. 
> - Fonction : Supprime les répertoires où l'on stocke les données des derniers lancements du programme : `./models`, `./plots` et `./vectorizer`. Cela permet notamment de pouvoir recommencer l'entraînement de 0, et ne pas repartir du dernier modèle entraîné.

> `load` : 
> - Prérequis :  Le fichier `./data.csv`. 
> - NB : Attention a bien installer la bonne version de `pytorch`. Privilégiez l'installation du mode `CUDA`, si jamais votre poste a une carte graphique `NVIDIA`, ce qui vous permettra d'augmenter grandement les temps de calculs, en utilisant le **GPU** de votre ordinateur.
> - Fonction : Charge un **vocabulaire** des différents mots prédictibles à partir des différents labels du dataset et le sauvegarde dans le répertoire `./vectorizer`.

> `train`
> - Prérequis : `load`
> - Fonction : **Entraine** le réseau de neurones à partir des datasets en fonction des paramètres spécifiés dans le main (totalEpochs, step, etc.). Une fois le réseau entraîné, **exporte les paramètres du réseau** dans le répertoire `./models` et les graphiques dans le répertoire `./plots`. 
> - NB : Si les fichiers du répertoire `./models` ne sont pas effacés, on récupère l'état du modèle sauvegardé dans le fichier.

> `predict`
> - Prérequis : `train`
> - Fonction : Effectue la **prédiction** du texte par la variable `PREDICTION_SENTANCE`, et l'affiche.

## Auteur

Ewen BOUQUET, le 21/06/2022