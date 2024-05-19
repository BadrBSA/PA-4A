# Intellisum

## Contexte général

Ce projet a pour but d'implémenter et déployer sur le cloud une solution permettant de généré du résumé de très long texte.
L'application acceptera en entrée du texte saisi par l'utilisateur, un fichier au format txt ou pdf.
L'utilisateur pourra choisir d'enregistrer le résultat du résumé au format txt et pourra lire le résumé avec un bouton permettant de déclencher un mp3 du texte.

Actuellement, ce projet en est encore à l'étape de prototype. 
Il a d'abord été lancé de nombreuses fois en local afin d'utiliser différents models disponibles sur huggingface, de manière à auditer leur performances.
Nous avons fait le choix de partir sur le modèle mistral-7b-instruct-v0.2 car ce dernier permettait d'obtenir des résumés plus cohérents dans un laps de temps convenable.

Pour le moment, nous pouvons résumer des chapitres entiers de livres en environ une minute, et nous avons pu résumé en une exécution l'entièreté d'Harry Potter 1 donné en input au format txt. 
Pour un texte de la taille d'Harry Potter 1, soit 76 944 mots, 309 pages ou encore 127 505 tokens, notre algorithme fournit un résumé satisfaisant en 26min environ.
Voici le résumé obtenu :

[[' In the opening chapter of "Harry Potter and the Sorcerer\'s Stone," Harry Potter, a muggle child living with his muggle relatives, the Dursleys, experiences strange occurrences and dreams of being a wizard. Mr. Dursley dismisses these events as tricks of the mind. Meanwhile, Albus Dumbledore and Professor McGonagall discuss Voldemort\'s return and the rumors about the Potter family. Hagrid, the Keeper of Keys and Grounds at Hogwarts, arrives to take Harry to the magical school, revealing his true identity as a wizard.\n\nHarry, Hagrid, and Dumbledore go to Diagon Alley to buy school supplies, and Harry purchases a wand, which chooses him. They meet Hermione Granger and set off for Hogwarts on the Hogwarts Express. Harry is excited to start his new life at Hogwarts but is met with challenges as they suspect Professor Snape of attempting to steal the Sorcerer\'s Stone, Harry\'s first Quidditch match approaches, and the Christmas holidays arrive.\n\nWhile searching for information on the Sorcerer\'s Stone and Nicolas Flamel, Harry and his friends, Ron Weasley and Hermione Granger, encounter Snape jinxing Harry\'s broom during a Quidditch match, Harry\'s birthday presents include an invisibility cloak from an unknown sender, and they overhear Malfoy insulting their family, resulting in a fight. Harry also discovers Dumbledore\'s mirror, the Mirror of Erised, and becomes enamored with it but is warned of its dangers. They continue to search for the wounded unicorn, with Harry, Ron, and Malfoy going separate ways to find it and eventually healing it.\n\nThroughout the text, Harry, Ron, and Hermione face various challenges as they uncover the truth about the Sorcerer\'s Stone and Snape\'s involvement. They struggle with fear, frustration, and isolation as they piece together the puzzle of the mystery. Despite the obstacles, they remain determined and are supported by their friendship and guidance from Dumbledore.\n\nKey events and themes:\n\n* Harry\'s identity as a wizard is revealed.\n* Suspected theft of the Sorcerer\'s Stone by Professor Snape.\n* Strained relationships with the Dursleys.\n* Harry\'s excitement for Quidditch and Christmas.\n\nKey characters:\n\n* Harry Potter\n* Mr. Dursley\n* Mrs. Dursley\n* Petunia Dursley\n* Percy Dursley\n* Albus Dumbledore\n* Professor McGonagall\n* Hagrid\n* Ron Weasley\n* Hermione Granger\n* Professor Snape\n* Nicolas Flamel\n* Malfoy.\n\nKey words and phrases:\n\n* Muggle\n* Magic\n* Wands\n* Wizarding World\n* Hogwarts\n* Transfiguration\n* Quidditch\n* Sorting Hat\n* Platform 9 and 3/4\n* Hogwarts Express\n* Diagon Alley\n* Ollivanders Wand Shop\n* Unicorn.</s>']]

# Déploiement du modèle sur GCP Vertex AI

## Introduction

Cette section décrit les étapes que nous avons suivies pour déployer un modèle pré-entraîné sur Google Cloud Platform (GCP) en utilisant Vertex AI.

## Étapes de déploiement

### 1. Initialisation de Vertex AI

Nous avons commencé par initialiser Vertex AI avec les détails du projet et de la localisation. Cela permet de configurer l'environnement de travail pour interagir avec les services GCP.

### 2. Configuration des identifiants Google Cloud

Nous avons configuré les identifiants Google Cloud pour nous assurer que l'authentification avec GCP se fait correctement. Cela impliquait de définir les variables d'environnement nécessaires.

### 3. Création d'un endpoint

Nous avons créé un endpoint dans Vertex AI. Un endpoint est une URL unique où le modèle sera déployé, ce qui est essentiel pour pouvoir utiliser le modèle via des requêtes HTTP.

### 4. Téléchargement du modèle

Nous avons téléchargé le modèle pré-entraîné, déjà stocké dans un bucket Google Cloud Storage. Nous avons spécifié l'image Docker appropriée pour le serveur de prédiction afin de l'utiliser lors du déploiement.

### 5. Tentative de déploiement du modèle

Enfin, nous avons tenté de déployer le modèle sur le endpoint créé. Cela comprenait la spécification du type de machine et des accélérateurs nécessaires (par exemple, GPU), ainsi que d'autres configurations comme le compte de service utilisé pour le déploiement.

## État actuel

Le projet est en cours de déploiement sur le cloud GCP. Nous rencontrons actuellement des difficultés qui ralentissent le processus de déploiement. Cependant, le côté applicatif est déjà en place, tout comme les algorithmes nécessaires au bon fonctionnement du projet une fois que le déploiement sur le cloud sera opérationnel.

## Participants

Ce projet est développé et maintenu par :
- Yahia Ferchouli
- Badr Bouaissa
- Clément Devarieux

## Prochaines étapes

La suite logique du projet est le déploiement sur le cloud, la création des rôles users de l'application et les tests post déploiement.

## Explication du repo git

Pour le moment, ce repository git n'est pas propre.
Nous avons créé et travaillé sur plusieurs branches :

- HEAD
- badr-branch
- main
- models-exp

Nous n'avons pas été propres et avons codé un peu en désordre sur les différentes branches, mais nous nous y repérons.
Tout cela sera bien évidemment mis au propre prochainement.
Le fichier qui représente au mieux pour le moment la démarche algorithmique qui sera utilisée pour le modèle est disponible dans la branche badr-branch et se nomme 'test_summmarize_tokenizer.ipynb'.
Dans ce fichier, nous faisons les imports nécessaires au notebook, configurons le modèle issue de MistralAI avec de la quantization et de l'accélération GPU avec cuda.
Nous implémentons ensuite des fonctions qui nous permettent de tokenizer un texte en entrée, le découper en segments de 5000 tokens, de faire des résumés de chaque segments de tokens, puis de regrouper ces résumer et les redécouper, les rerésumer etc... jusqu'à ce que nous obtenions un texte avec suffisamment peu de tokens pour être résumé d'un coup. Ce résumé final sera notre output du modèle.

