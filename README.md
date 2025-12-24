# STATS_APP_EPL

Voici le lien pour la connexion.

https://statsappepl-aregba.streamlit.app/



Documentation de l’application EPL_STATS_APP

1️ Présentation général
EPL_STATS_APP est une application interactive développée avec Streamlit permettant l’analyse académique des étudiants à partir d’un fichier CSV.
Elle permet :
•	D’explorer les résultats par parcours, année d’étude, matière, enseignant et sexe
•	De calculer automatiquement les effectifs, statistiques, classements
•	D’identifier les étudiants en difficulté
•	D’exporter les données analysées et les graphes
2️- Objectifs de l’application
•	Analyser les effectifs étudiants et enseignants
•	 Détecter les étudiants en difficulté
•	 Croiser les données (parcours - année - matière - professeur - sexe)
•	 Exporter les résultats pour exploitation externe
3️-Outils

Python
Streamlit
Pandas
NumPy
Matplotlib
Seaborn
CSV

4-Structure du fichier CSV attendu
Colonnes obligatoires
Colonne	Description
Matricule :	Identifiant unique de l’étudiant
Nom	:Nom de l’étudiant
Prénom:	Prénom de l’étudiant
sexe:	M ou F
Niveau	:Niveau académique
Département	:Département
Parcours:	Parcours académique
Année_étude:	Année d’étude (1, 2, 3…)
Année_académique:	Année académique
UE	: d’enseignement
:	Matière
Professeur:	Enseignant
Note:	Note obtenue
Décision:	Validé / NV
 L’application vérifie automatiquement la présence des colonnes essentielles

