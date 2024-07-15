# Intellisum

## Contexte général

Ce projet a pour but d'implémenter et déployer sur le cloud une solution permettant de généré du résumé de très long texte.
L'application accepte en entrée du texte saisi par l'utilisateur, un fichier au format txt ou pdf.
L'utilisateur pourra choisir d'enregistrer le résultat du résumé au format txt et pourra lire le résumé avec un bouton permettant de déclencher un mp3 du texte.

Nous avons fait le choix de partir sur le modèle mistral-7b-instruct-v0.2 car ce dernier permettait d'obtenir des résumés plus cohérents dans un laps de temps convenable.

Nous pouvons résumer des chapitres entiers de livres en environ une minute, et nous avons pu résumé en une exécution l'entièreté d'Harry Potter 1 donné en input au format txt. 
Pour un texte de la taille d'Harry Potter 1, soit 76 944 mots, 309 pages ou encore 127 505 tokens, notre algorithme fournit un résumé satisfaisant en 29min environ.
Voici le résumé obtenu :

The last month of Harry's summer with the Dursleys is filled with tension and anxiety, especially after he turns eleven and receives a letter from Hagrid, asking him to meet at Diagon Alley. Harry feels relieved after Hogwarts officials arrive to take him to school, and he meets Ollivander at Wand Shop, who reveals that his parents had left him a chest with instructions for him to open it when he was ready. Harry is surprised when he finds the chest responds only to his presence and when he opens it, he finds a map leading him to a location and a letter from his father, explaining that they couldn't keep Harry with them due to Voldemort's searches for him. Harry finally starts to enjoy his journey to Hogwarts, after some last-minute fights with Quirrell and Moody, he finds the Sorcerer's Stone hidden in Quirrell's turban, and destroys it to keep Voldemort from returning. Dumbledore explains how Quirrell was under Voldemort's control, and they all celebrate the victory over the dark lord. The chapter ends with Dumbledore giving all of Slytherin's house points to Gryffindor, allowing the latter to win the house cup, and Dumbledore thanking Harry for saving England's magic.

## Étapes de déploiement

### AVANT TOUT: vous devez installer les librairies nécessaires à travers le fichier "requirements.txt" avec la commande "pip install -r requirements.txt" et il faut avoir à disposition "chocolatey" permettant l'installation de NGROK

### Explication scripts
fine_tuning.py est le script utilisé afin de fine tune notre modèle

generate_summary.py permet le lancement de résumé le texte. Pour lancer un test, rajouter votre fichier texte dans le dossier "data/books" et modifier le with "open("data/books/first_chapter", "r", encoding="utf-8") as f:" avec le chemin adéquat

model_local_deployement.py déploie une app flask en local sur le port 5000, qui à travers un endpoint permet de lancer le résumé d'un texte choisi

test_endpoint.py permet de tester l'app flask

### Etapes

1 - Lancer model_local_deployement.py 
2 - Lancer powershell en mode admin
3 - utiliser la commande : ngrok http http://localhost:5000/
4 - l'app est déployée est accessible partout à travers le lien généré par ngrok

## Participants

Ce projet est développé et maintenu par :
- Yahia Ferchouli
- Badr Bouaissa
- Clément Devarieux

