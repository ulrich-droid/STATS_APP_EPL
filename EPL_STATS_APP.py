import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
# ---------------- CONFIG ----------------
st.set_page_config(page_title="EPL_STATS_APP", layout="wide")
st.title("📊 EPL_STATS_APP")

# ---------------- UPLOAD ----------------
uploaded_file = st.file_uploader("📂 Importer un fichier CSV", type=["csv"])

if uploaded_file is None:
    st.info("Veuillez charger un fichier CSV pour continuer.")
    st.stop()

try:
    df = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"Erreur lors de la lecture du fichier CSV : {e}")
    st.stop()

# ---------------- COLONNES OBLIGATOIRES ----------------
required_cols = ["Nom", "Note", "Parcours", "Matière", "Année_étude", "Professeur", "sexe"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"Colonnes manquantes : {missing}")
    st.stop()

# ---------------- SIDEBAR ----------------
st.sidebar.title("🎛️ Paramètres")

# --- Parcours ---
parcours = st.sidebar.multiselect(
    "Parcours",
    options=sorted(df["Parcours"].dropna().unique()),
    default=sorted(df["Parcours"].dropna().unique())
)

# --- Année d'étude ---
annee = st.sidebar.selectbox(
    "Année d'étude",
    ["Toutes"] + sorted(df["Année_étude"].dropna().unique().tolist())
)

# --- Filtrer la base selon Parcours et Année ---
df_base = df[df["Parcours"].isin(parcours)]
if annee != "Toutes":
    df_base = df_base[df_base["Année_étude"] == annee]

# --- Sexe ---
sexes_disponibles = sorted(df_base["sexe"].dropna().unique())
sexe = st.sidebar.multiselect(
    "Sexe",
    options=sexes_disponibles,
    default=sexes_disponibles
)
if sexe:
    df_base = df_base[df_base["sexe"].isin(sexe)]

# --- Matière ---
matieres_disponibles = sorted(df_base["Matière"].dropna().unique())
matiere = st.sidebar.multiselect(
    "Matières",
    options=matieres_disponibles,
    default=matieres_disponibles,
    key="matiere_sidebar"
)
if matiere:
    df_base = df_base[df_base["Matière"].isin(matiere)]

# --- Professeur ---
professeurs_disponibles = sorted(df_base["Professeur"].dropna().unique())
professeur = st.sidebar.multiselect(
    "Professeurs",
    options=professeurs_disponibles,
    default=professeurs_disponibles,
    key="professeur_sidebar"
)
if professeur:
    df_base = df_base[df_base["Professeur"].isin(professeur)]

# --- Filtrage final ---
df_filtre = df_base.copy()

if df_filtre.empty:
    st.warning("⚠️ Aucun résultat pour ces filtres.")
    st.stop()

# ---------------- AFFICHAGE TABLE ----------------
st.subheader("📋 Données filtrées")
st.dataframe(df_filtre, use_container_width=True)
# ---------------- EXPORT CSV ----------------
st.subheader("📥 Exporter les données")

csv = df_filtre.to_csv(index=False).encode("utf-8")

st.download_button(
    label="📄 Télécharger les données filtrées (CSV)",
    data=csv,
    file_name="donnees_filtrees.csv",
    mime="text/csv"
)
#################recherche étudiant
st.sidebar.markdown("---")
st.sidebar.subheader("🔍 Recherche étudiant")

recherche = st.sidebar.text_input(
    "Nom ou identifiant",
    placeholder="Ex: KOFFI ou LPGL001"
)

try:
    if recherche:  # on ne filtre que si l'utilisateur a saisi quelque chose
        df_filtre = df_base.copy()
        recherche_lower = recherche.lower()

        colonnes_recherche = ["Nom"]
        if "Matricule" in df_filtre.columns:
            colonnes_recherche.append("Matricule")

        masque = False
        for col in colonnes_recherche:
            masque = masque | df_filtre[col].astype(str).str.lower().str.contains(recherche_lower)

        df_filtre = df_filtre[masque]

        st.subheader("📋 Résultat de la recherche")
        st.dataframe(df_filtre, use_container_width=True)

        # ------------------- Export CSV -------------------
        if not df_filtre.empty:
            try:
                csv_data = df_filtre.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="⬇️ Exporter les données filtrées en CSV",
                    data=csv_data,
                    file_name="etudiants_filtrés.csv",
                    mime="text/csv"
                )
            except Exception as e_csv:
                st.error(f"Erreur lors de l'export CSV : {e_csv}")

    else:
        st.info("Veuillez saisir un nom ou un identifiant pour effectuer la recherche.")

except KeyError as e:
    st.error(f"Colonne manquante : {e}")
except Exception as e:
    st.error(f"Une erreur est survenue : {e}")


    
##################

# ---------------- TRANSFORMATIONS ----------------
df_modif = df_filtre.copy()
df_modif["Note"] = pd.to_numeric(df_modif["Note"], errors="coerce")
df_modif = df_modif.dropna(subset=["Note"])
seuil_validation = st.sidebar.slider("Seuil de validation", 0, 20, 10)
df_modif["Validee"] = np.where(df_modif["Note"] >= seuil_validation, "V", "NV")
type_classement = st.sidebar.radio(
    "Classer les étudiants par :",
    ["Moyenne générale", "Nombre de matières validées"]
)

# ---------------- EFFECTIFS ----------------
st.subheader("👥 Effectifs")
etudiants_uniques = df_filtre.drop_duplicates(subset=["Matricule"] if "Matricule" in df_filtre.columns else ["Nom"])
nb_etudiants = etudiants_uniques.shape[0]
nb_filles = etudiants_uniques[etudiants_uniques["sexe"] == "F"].shape[0]
nb_garcons = etudiants_uniques[etudiants_uniques["sexe"] == "M"].shape[0]
profs_uniques = df_filtre["Professeur"].dropna().unique()
nb_profs = len(profs_uniques)

col1, col2, col3, col4 = st.columns(4)
col1.metric("👨‍🎓 Étudiants", nb_etudiants)
col2.metric("👩 Filles", nb_filles)
col3.metric("👨 Garçons", nb_garcons)
col4.metric("👨‍🏫 Professeurs", nb_profs)


# ---------------- STATISTIQUES ----------------
st.subheader("📈 Statistiques globales")
notes = df_modif["Note"]
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Moyenne", round(notes.mean(), 2))
col2.metric("Médiane", round(notes.median(), 2))
col3.metric("Écart-type", round(notes.std(), 2))
col4.metric("Q1 / Q3", f"{round(notes.quantile(0.25),2)} / {round(notes.quantile(0.75),2)}")
col5.metric("Taux de réussite (%)", round((df_modif["Validee"] == "V").mean() * 100, 2))

# ---------------- TAUX DE VALIDATION PAR MATIÈRE ----------------
st.subheader("✅ Taux de validation par matière")
taux_par_matiere = (
    df_modif.groupby("Matière")["Validee"]
    .apply(lambda x: (x == "V").mean() * 100)
    .round(2)
    .reset_index(name="Taux_validation (%)")
)
st.dataframe(taux_par_matiere, use_container_width=True)
####################### EXPORTATION ###################
csv_taux = taux_par_matiere.to_csv(index=False).encode("utf-8")
st.download_button(
    label="📄 Télécharger le taux de validation (CSV)",
    data=csv_taux,
    file_name="taux_validation_par_matiere.csv",
    mime="text/csv"
)

# ---------------- BARPLOT ----------------
st.subheader("📊 Taux de validation par matière")
fig, ax = plt.subplots(figsize=(8,5))
sns.barplot(
    data=taux_par_matiere,
    x="Taux_validation (%)",
    y="Matière",
    hue="Matière",
    legend=False,
    ax=ax
)
st.pyplot(fig)
########################EXPORTATION
fig.savefig("taux_validation_par_matiere.png", bbox_inches="tight")
with open("taux_validation_par_matiere.png", "rb") as f:
    st.download_button(
        label="📥 Télécharger le graphe (PNG)",
        data=f,
        file_name="taux_validation_par_matiere.png",
        mime="image/png"
    )

# ---------------- HISTOGRAMME ----------------
st.subheader("📊 Histogramme des notes")
bins = st.number_input("Nombre de bins", 5, 50, 10)
fig, ax = plt.subplots()
ax.hist(df_modif["Note"], bins=bins, edgecolor="black")
st.pyplot(fig)

fig.savefig("histogramme_notes.png", bbox_inches="tight")
with open("histogramme_notes.png", "rb") as f:
    st.download_button(
        label="📥 Télécharger l’histogramme (PNG)",
        data=f,
        file_name="histogramme_notes.png",
        mime="image/png"
    )

# ---------------- BOX PLOT PAR SEXE ----------------
st.subheader("📊 Répartition des notes par sexe (Box plot)")

fig, ax = plt.subplots(figsize=(6,5))
sns.boxplot(
    data=df_modif,
    x="sexe",
    y="Note",
    palette={"F": "pink", "M": "blue"}
)
ax.set_xlabel("Sexe")
ax.set_ylabel("Note")
ax.set_title("Répartition des notes par sexe")
st.pyplot(fig)
fig.savefig("boxplot_notes_par_sexe.png", bbox_inches="tight")
with open("boxplot_notes_par_sexe.png", "rb") as f:
    st.download_button(
        label="📥 Télécharger le box plot (PNG)",
        data=f,
        file_name="boxplot_notes_par_sexe.png",
        mime="image/png"
    )

# ---------------- BOXPLOT PAR PARCOURS DYNAMIQUE ----------------
st.subheader("📦 Distribution des notes par parcours")

# Nettoyer les parcours pour éviter les espaces invisibles
df_filtre["Parcours"] = df_filtre["Parcours"].astype(str).str.strip()

# Vérifier qu'il y a des parcours sélectionnés
if df_filtre["Parcours"].nunique() > 0:
    fig, ax = plt.subplots(figsize=(8,5))
    sns.boxplot(
        data=df_filtre,
        x="Parcours",
        y="Note",
        palette="Set2"
    )
    ax.set_title("Répartition des notes par parcours")
    ax.set_ylabel("Note")
    ax.set_xlabel("Parcours")
    plt.xticks(rotation=45)  # rotation si les noms sont longs
    st.pyplot(fig)
else:
    st.warning("⚠️ Aucune donnée pour le parcours sélectionné.")

fig.savefig("boxplot_notes_par_parcours.png", bbox_inches="tight")
with open("boxplot_notes_par_parcours.png", "rb") as f:
    st.download_button(
        label="📥 Télécharger le box plot (PNG)",
        data=f,
        file_name="boxplot_notes_par_parcours.png",
        mime="image/png"
    )
######################## Fonction d'exportation
def export_plot(fig, filename, title=None):
    if title:
        st.subheader(title)
    st.pyplot(fig)
    fig.savefig(filename, bbox_inches="tight")
    with open(filename, "rb") as f:
        st.download_button(
            label=f"📥 Télécharger {filename}",
            data=f,
            file_name=filename,
            mime="image/png"
        )


# ---------------- HISTOGRAMME MOYENNE PAR SEXE ----------------

# Calculer la moyenne des notes par sexe
moyenne_par_sexe = df_modif.groupby("sexe")["Note"].mean().reset_index()

# Créer le graphique
fig, ax = plt.subplots(figsize=(5,5))
sns.barplot(
    data=moyenne_par_sexe,
    x="sexe",
    y="Note",
    palette={"F": "pink", "M": "blue"},
    edgecolor="black"
)

ax.set_xlabel("Sexe")
ax.set_ylabel("Moyenne des notes")
ax.set_title("Moyenne des notes par sexe")

export_plot(fig, "moyenne_notes_par_sexe.png", "📊 Moyenne des notes par sexe")

# ---------------- CLASSEMENT ----------------
st.subheader("🏆 Classement des étudiants ")

try:
    df_classement = df_modif.copy()
    df_classement["Validee_int"] = (df_classement["Note"] >= seuil_validation).astype(int)

    # ---------------- Identifier les colonnes pour l'étudiant ----------------
    student_key = []
    if "Matricule" in df_classement.columns:
        student_key.append("Matricule")
    if "Nom" in df_classement.columns:
        student_key.append("Nom")
    if "Prénom" in df_classement.columns:
        student_key.append("Prénom")

    if not student_key:
        st.error("Aucune colonne pour identifier l’étudiant (Matricule/Nom/Prénom) !")
        st.stop()

    # ---------------- Calcul du classement ----------------
    if type_classement == "Moyenne générale":
        classement = df_classement.groupby(student_key, as_index=False).agg(
            Moyenne=("Note", "mean")
        ).sort_values(by="Moyenne", ascending=False)
    else:
        classement = df_classement.groupby(student_key, as_index=False).agg(
            Moyenne=("Note", "mean"),
            Matieres_validees=("Validee_int", "sum"),
            Nb_matieres=("Note", "count")
        ).sort_values(by=["Matieres_validees", "Moyenne"], ascending=[False, False])

    # ---------------- Ajouter infos supplémentaires ----------------
    infos_cols = ["Parcours", "Année_étude", "sexe"]
    infos_cols = [c for c in infos_cols if c in df_classement.columns]

    agg_dict = {}
    if "Parcours" in infos_cols: agg_dict["Parcours"] = "first"
    if "Année_étude" in infos_cols: agg_dict["Année_étude"] = "max"  # la plus élevée
    if "sexe" in infos_cols: agg_dict["sexe"] = "first"

    infos = df_classement.groupby(student_key, as_index=False).agg(agg_dict)

    # ---------------- Merge final ----------------
    classement = classement.merge(infos, on=student_key, how="left")

    # ---------------- Affichage ----------------
    if not classement.empty:
        # Réorganiser les colonnes pour que Nom/Prénom/Matricule soient visibles
        final_cols = student_key + [c for c in classement.columns if c not in student_key]
        classement = classement[final_cols]

        st.dataframe(classement, width='stretch')

        # ---------------- Export CSV ----------------
        csv_classement = classement.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📄 Télécharger le classement (CSV)",
            data=csv_classement,
            file_name="classement_etudiants.csv",
            mime="text/csv",
            key="download_classement"
        )
    else:
        st.info("Aucun étudiant à afficher pour ce classement.")

except Exception as e:
    st.warning(f"Impossible de générer le classement : {e}")
    st.exception(e)



# ---------------- ÉTUDIANTS EN DIFFICULTÉ ----------------
st.subheader("⚠️ Étudiants en difficulté")

try:
    # ---------------- Identifier les colonnes pour chaque étudiant ----------------
    student_key = []
    if "Matricule" in df_modif.columns:
        student_key.append("Matricule")
    if "Nom" in df_modif.columns:
        student_key.append("Nom")
    if "Prénom" in df_modif.columns:
        student_key.append("Prénom")

    if not student_key:
        st.error("Aucune colonne pour identifier l’étudiant (Matricule/Nom/Prénom) !")
        st.stop()

    # ---------------- Calcul de la moyenne par étudiant ----------------
    df_moyenne = df_modif.groupby(student_key, as_index=False).agg(Moyenne=("Note", "mean"))

    # ---------------- Identifier les étudiants en difficulté ----------------
    df_difficulte = df_moyenne[df_moyenne["Moyenne"] < seuil_validation]

    # ---------------- Affichage du nombre ----------------
    st.metric("Nombre d'étudiants en difficulté", df_difficulte.shape[0])

    # ---------------- Affichage du tableau ----------------
    if not df_difficulte.empty:
        # Réorganiser les colonnes pour que Nom/Prénom/Matricule soient visibles en premier
        final_cols = student_key + [c for c in df_difficulte.columns if c not in student_key]
        df_difficulte = df_difficulte[final_cols]

        st.dataframe(df_difficulte, width='stretch')

        # ---------------- Export CSV ----------------
        csv_difficulte = df_difficulte.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📄 Télécharger les étudiants en difficulté (CSV)",
            data=csv_difficulte,
            file_name="etudiants_en_difficulte.csv",
            mime="text/csv",
            key="download_difficulte"
        )
    else:
        st.info("Aucun étudiant en difficulté pour ces filtres.")

except Exception as e:
    st.error(f"❌ Une erreur est survenue : {e}")
    st.exception(e)

# ---------------- HEATMAP ----------------
st.subheader("🔗 Corrélation entre matières")

try:
    # Pivot table : 1 ligne par étudiant, 1 colonne par matière
    df_pivot = df_modif.pivot_table(
        index="Matricule" if "Matricule" in df_modif.columns else "Nom",
        columns="Matière",
        values="Note",
        aggfunc="mean"
    )

    # Vérifier qu’il y a au moins 2 matières pour calculer la corrélation
    if df_pivot.shape[1] >= 2 and not df_pivot.dropna(how="all").empty:
        fig, ax = plt.subplots(figsize=(10,6))
        sns.heatmap(df_pivot.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

        # Sauvegarde et export PNG
        fig.savefig("heatmap_correlation_matieres.png", bbox_inches="tight")
        with open("heatmap_correlation_matieres.png", "rb") as f:
            st.download_button(
                label="📥 Télécharger la heatmap (PNG)",
                data=f,
                file_name="heatmap_correlation_matieres.png",
                mime="image/png",
                 key="download_heatmap"
            )
    else:
        st.warning("Pas assez de matières valides pour calculer la corrélation.")

except Exception as e:
    st.error("❌ Une erreur est survenue lors de la génération de la heatmap.")
    st.exception(e)
