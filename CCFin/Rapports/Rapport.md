Rapport d'Introduction : Base de Données Wholesale Customers
1. Contexte Général
La base de données Wholesale Customers est un jeu de données issu du référentiel UCI Machine Learning Repository, une ressource académique reconnue mondialement dans le domaine de l'apprentissage automatique et de l'analyse statistique. Cette base constitue un support précieux pour l'étude du comportement des clients dans le secteur de la distribution en gros.
2. Origine et Domaine d'Application
Secteur d'Activité
Cette base de données provient du secteur de la distribution en gros (wholesale), un maillon essentiel de la chaîne d'approvisionnement qui relie les producteurs aux détaillants et aux établissements de restauration. Le secteur wholesale se caractérise par des volumes importants et des relations commerciales B2B (Business-to-Business).
Contexte Commercial
Les données reflètent la réalité des transactions entre un distributeur grossiste et ses clients professionnels. Ces clients peuvent être classés en deux catégories principales :

HoReCa (Hôtels, Restaurants, Cafés) : établissements de restauration et d'hébergement
Retail (Commerce de détail) : magasins, supermarchés et autres points de vente au détail

3. Nature des Données Collectées
Informations Géographiques et Structurelles
Les données incluent des informations sur la région géographique des clients et leur canal de distribution, permettant d'analyser les variations comportementales selon la localisation et le type d'établissement.
Données de Consommation
Le cœur de la base de données repose sur les dépenses annuelles des clients réparties en plusieurs catégories de produits :

Fresh (Produits frais) : fruits, légumes, viandes fraîches
Milk (Produits laitiers) : lait, fromages, yaourts
Grocery (Épicerie) : produits d'épicerie sèche, conserves
Frozen (Produits surgelés) : aliments congelés
Detergents_Paper (Détergents et papier) : produits d'entretien et papeterie
Delicatessen (Traiteur) : produits de charcuterie fine et spécialités

Ces montants sont exprimés en unités monétaires et représentent les achats totaux effectués sur une année.
4. Objectifs et Applications Pratiques
Segmentation de Clientèle
L'un des principaux intérêts de cette base est la segmentation de marché. En analysant les patterns de dépenses, les entreprises peuvent identifier différents profils de clients ayant des comportements d'achat similaires. Par exemple, un restaurant peut privilégier les produits frais et traiteur, tandis qu'un supermarché aura des dépenses plus équilibrées.
Optimisation Stratégique
Les insights tirés de ces données permettent aux distributeurs de :

Personnaliser les offres commerciales selon les profils identifiés
Optimiser la gestion des stocks en anticipant les besoins spécifiques de chaque segment
Adapter les stratégies de pricing selon les catégories et les types de clients
Améliorer la logistique en comprenant les volumes et fréquences d'achat

Analyse Prédictive
La base permet également de développer des modèles prédictifs pour :

Classifier automatiquement de nouveaux clients
Anticiper les tendances de consommation
Identifier les clients à fort potentiel
Détecter des comportements atypiques ou des opportunités commerciales

5. Importance dans le Contexte Académique et Professionnel
Recherche et Enseignement
Cette base de données est largement utilisée dans les universités et centres de recherche pour enseigner et expérimenter diverses techniques d'analyse :

Analyse en composantes principales (ACP)
Clustering (K-means, classification hiérarchique)
Algorithmes de classification supervisée
Techniques de réduction de dimensionnalité

Applications Industrielles
Dans le monde professionnel, les méthodes appliquées à ce type de données sont essentielles pour :

Le Customer Relationship Management (CRM)
L'intelligence marketing
La business intelligence
L'optimisation de la supply chain

6. Pertinence et Actualité
Malgré sa création il y a plusieurs années, cette base demeure pertinente car les principes de segmentation et d'analyse comportementale restent fondamentaux dans le commerce moderne. Les techniques développées sur ces données sont directement transposables aux défis actuels du e-commerce, de la personnalisation client et de l'analyse big data.
La compréhension des patterns de consommation dans le secteur wholesale offre des insights précieux non seulement pour les distributeurs, mais aussi pour toute entreprise cherchant à mieux comprendre et anticiper les besoins de sa clientèle professionnelle.
from ucimlrepo import fetch_ucirepo 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

'''# ========== CHARGEMENT DES DONNÉES ==========
print("=" * 60)
print("CHARGEMENT DU DATASET WHOLESALE CUSTOMERS")
print("=" * 60)

# Fetch dataset
wholesale_customers = fetch_ucirepo(id=292) 

# Data (as pandas dataframes)
X = wholesale_customers.data.features 
y = wholesale_customers.data.targets 

# Combiner X et y pour l'analyse complète
df = pd.concat([X, y], axis=1)

# Afficher les métadonnées
print("\n--- MÉTADONNÉES ---")
print(wholesale_customers.metadata)

print("\n--- INFORMATIONS SUR LES VARIABLES ---")
print(wholesale_customers.variables)

# ========== EXPLORATION DES DONNÉES ==========
print("\n" + "=" * 60)
print("EXPLORATION DES DONNÉES")
print("=" * 60)

print("\n--- Premières lignes du dataset ---")
print(df.head())

print("\n--- Informations sur le dataset ---")
print(df.info())

print("\n--- Statistiques descriptives globales ---")
print(df.describe())

# ========== STATISTIQUES DESCRIPTIVES DÉTAILLÉES ==========
print("\n" + "=" * 60)
print("STATISTIQUES DESCRIPTIVES DÉTAILLÉES PAR VARIABLE")
print("=" * 60)

# Sélectionner les colonnes numériques
numeric_cols = df.select_dtypes(include=[np.number]).columns

for col in numeric_cols:
    print(f"\n{'=' * 60}")
    print(f"VARIABLE: {col}")
    print(f"{'=' * 60}")
    
    colonne = df[col].dropna()
    
    print(f"Nombre d'observations: {len(colonne)}")
    print(f"Valeurs manquantes: {df[col].isna().sum()}")
    print(f"Moyenne: {colonne.mean():.2f}")
    print(f"Médiane: {colonne.median():.2f}")
    print(f"Mode: {colonne.mode()[0]:.2f}" if len(colonne.mode()) > 0 else "Mode: N/A")
    print(f"Écart-type: {colonne.std():.2f}")
    print(f"Variance: {colonne.var():.2f}")
    print(f"Min: {colonne.min():.2f}")
    print(f"Max: {colonne.max():.2f}")
    print(f"Q1 (25%): {colonne.quantile(0.25):.2f}")
    print(f"Q2 (50% - Médiane): {colonne.quantile(0.50):.2f}")
    print(f"Q3 (75%): {colonne.quantile(0.75):.2f}")
    print(f"IQR (Écart interquartile): {colonne.quantile(0.75) - colonne.quantile(0.25):.2f}")
    print(f"Asymétrie (Skewness): {colonne.skew():.2f}")
    print(f"Kurtosis: {colonne.kurtosis():.2f}")

# ========== VISUALISATIONS ==========
print("\n" + "=" * 60)
print("GÉNÉRATION DES VISUALISATIONS")
print("=" * 60)

# Configuration du style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)

# Créer des visualisations pour chaque variable numérique
for col in numeric_cols:
    colonne = df[col].dropna()
    
    # Créer une figure avec plusieurs sous-graphiques
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Analyse Statistique - {col}', fontsize=16, fontweight='bold')
    
    # 1. Histogramme avec moyenne et médiane
    axes[0, 0].hist(colonne, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0, 0].set_title('Distribution des valeurs', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Valeur')
    axes[0, 0].set_ylabel('Fréquence')
    axes[0, 0].axvline(colonne.mean(), color='red', linestyle='--', linewidth=2,
                       label=f'Moyenne: {colonne.mean():.2f}')
    axes[0, 0].axvline(colonne.median(), color='green', linestyle='--', linewidth=2,
                       label=f'Médiane: {colonne.median():.2f}')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Boxplot pour détecter les outliers
    bp = axes[0, 1].boxplot(colonne, vert=True, patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    axes[0, 1].set_title('Boxplot - Détection des outliers', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Valeur')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Ajouter les statistiques sur le boxplot
    q1, median, q3 = colonne.quantile([0.25, 0.5, 0.75])
    axes[0, 1].text(1.15, q1, f'Q1: {q1:.2f}', fontsize=9)
    axes[0, 1].text(1.15, median, f'Médiane: {median:.2f}', fontsize=9)
    axes[0, 1].text(1.15, q3, f'Q3: {q3:.2f}', fontsize=9)
    
    # 3. Courbe de densité
    colonne.plot(kind='density', ax=axes[1, 0], color='purple', linewidth=2)
    axes[1, 0].set_title('Courbe de densité', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Valeur')
    axes[1, 0].set_ylabel('Densité')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].fill_between(axes[1, 0].get_lines()[0].get_xdata(), 
                            axes[1, 0].get_lines()[0].get_ydata(), 
                            alpha=0.3, color='purple')
    
    # 4. Q-Q Plot pour tester la normalité
    stats.probplot(colonne, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Q-Q Plot (Test de normalité)', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Ajouter le test de normalité (Shapiro-Wilk)
    if len(colonne) < 5000:  # Shapiro-Wilk fonctionne mieux pour n < 5000
        stat, p_value = stats.shapiro(colonne)
        axes[1, 1].text(0.05, 0.95, f'Shapiro-Wilk p-value: {p_value:.4f}', 
                       transform=axes[1, 1].transAxes, fontsize=9,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.show()
    print(f"✓ Graphiques générés pour {col}")

# ========== MATRICE DE CORRÉLATION ==========
print("\n" + "=" * 60)
print("MATRICE DE CORRÉLATION")
print("=" * 60)

plt.figure(figsize=(12, 10))
correlation_matrix = df[numeric_cols].corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Matrice de Corrélation entre les Variables', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.show()
print("✓ Matrice de corrélation générée")

# ========== DISTRIBUTION PAR CANAL (si la variable Channel existe) ==========
if 'Channel' in df.columns:
    print("\n" + "=" * 60)
    print("ANALYSE PAR CANAL")
    print("=" * 60)
    
    print("\nRépartition par Canal:")
    print(df['Channel'].value_counts())
    
    # Boxplot comparatif par canal
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Distribution des Variables par Canal', fontsize=16, fontweight='bold')
    
    product_cols = [col for col in numeric_cols if col not in ['Channel', 'Region']]
    
    for idx, col in enumerate(product_cols[:6]):
        row = idx // 3
        col_idx = idx % 3
        df.boxplot(column=col, by='Channel', ax=axes[row, col_idx])
        axes[row, col_idx].set_title(col)
        axes[row, col_idx].set_xlabel('Canal')
    
    plt.tight_layout()
    plt.show()
    print("✓ Analyse par canal générée")

print("\n" + "=" * 60)
print("ANALYSE TERMINÉE AVEC SUCCÈS!")
print("=" * 60)'''
