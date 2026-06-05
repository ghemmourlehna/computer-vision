#!/usr/bin/env python
# -*- coding: utf-8 -*-

# =============================================================================
#
#   CALIBRATION DE CAMÉRA — Code COMPLET avec explications détaillées
#   Projet : Vision Artificielle / Stéréovision
#
#   BUT DU PROGRAMME :
#   ─────────────────
#   On veut connaître les "caractéristiques internes" de notre caméra :
#     - Sa focale (en pixels)
#     - Son centre optique
#     - Sa distorsion (déformation de l'image)
#
#   COMMENT ?
#   ─────────
#   On photographie un damier (dont on connaît la géométrie exacte)
#   sous plusieurs angles. OpenCV compare ce qu'il voit dans l'image
#   avec ce qu'il devrait voir → il en déduit les paramètres de la caméra.
#
#   RÉSULTATS ATTENDUS :
#   ────────────────────
#     - ret (erreur RMS) < 1.0 pixel  → qualité acceptable
#     - fx ≈ fy                        → objectif symétrique (normal)
#     - erreur par image < 0.1 pixel   → excellent
#
# =============================================================================


# =============================================================================
# ██████████████████████████████████████████████████████████████████████████
# ÉTAPE 1 : IMPORTATION DES BIBLIOTHÈQUES
# ██████████████████████████████████████████████████████████████████████████
# =============================================================================

import cv2
# ┌─────────────────────────────────────────────────────────────────────────┐
# │  cv2 = OpenCV (Open Source Computer Vision)                             │
# │                                                                         │
# │  C'est LA bibliothèque de vision par ordinateur la plus utilisée.       │
# │  Elle contient des milliers de fonctions pour :                         │
# │    → Lire/écrire/afficher des images                                    │
# │    → Détecter des formes, des visages, des coins                        │
# │    → Calibrer une caméra                                                │
# │    → Faire de la stéréovision, du suivi d'objets...                    │
# │                                                                         │
# │  Installation : pip install opencv-python                               │
# └─────────────────────────────────────────────────────────────────────────┘

import numpy as np
# ┌─────────────────────────────────────────────────────────────────────────┐
# │  numpy = Numerical Python                                                │
# │                                                                         │
# │  Bibliothèque de calcul scientifique. Indispensable car :               │
# │    → Une image = un tableau NumPy (pixels = nombres)                    │
# │    → Les matrices (K, rotation, translation) sont des tableaux NumPy   │
# │    → Toutes les fonctions OpenCV retournent des tableaux NumPy          │
# │                                                                         │
# │  On l'importe sous le nom "np" pour écrire moins de code               │
# │  Installation : pip install numpy                                        │
# └─────────────────────────────────────────────────────────────────────────┘

import os
# ┌─────────────────────────────────────────────────────────────────────────┐
# │  os = Operating System                                                   │
# │                                                                         │
# │  Permet d'interagir avec le système de fichiers :                       │
# │    → Créer des dossiers (os.makedirs)                                   │
# │    → Vérifier si un fichier existe                                       │
# │    → Gérer les chemins de fichiers                                      │
# │                                                                         │
# │  Ici on l'utilise uniquement pour créer le dossier camera_params/      │
# └─────────────────────────────────────────────────────────────────────────┘

import glob
# ┌─────────────────────────────────────────────────────────────────────────┐
# │  glob = recherche de fichiers par motif (pattern matching)               │
# │                                                                         │
# │  Permet de chercher des fichiers en utilisant des wildcards (*) :       │
# │    glob.glob('./images/*.jpg')                                           │
# │    → trouve TOUS les fichiers .jpg dans le dossier ./images/            │
# │    → retourne une liste de chemins                                      │
# │                                                                         │
# │  Sans glob, on devrait écrire le nom de chaque image à la main         │
# └─────────────────────────────────────────────────────────────────────────┘


# =============================================================================
# ██████████████████████████████████████████████████████████████████████████
# ÉTAPE 2 : PARAMÈTRES DU DAMIER
# ██████████████████████████████████████████████████████████████████████████
# =============================================================================

CHECKERBOARD = (7, 9)
# ┌─────────────────────────────────────────────────────────────────────────┐
# │  CHECKERBOARD = (coins_internes_X, coins_internes_Y)                    │
# │                                                                         │
# │  ⚠️  ATTENTION : on compte les COINS INTERNES, pas les cases !          │
# │                                                                         │
# │  Visualisation d'un damier 4×3 cases :                                  │
# │                                                                         │
# │    ┌───┬───┬───┬───┐                                                    │
# │    │   │   │   │   │                                                    │
# │    ├───X───X───X───┤   ← les X sont les coins internes                 │
# │    │   │   │   │   │                                                    │
# │    ├───X───X───X───┤   → 3 coins en largeur (4 cases - 1)              │
# │    │   │   │   │   │   → 2 coins en hauteur (3 cases - 1)              │
# │    └───┴───┴───┴───┘   → donc (3, 2) et non (4, 3)                    │
# │                                                                         │
# │  Pour notre damier 8×10 cases :                                         │
# │    → 8 - 1 = 7 coins internes en largeur                               │
# │    → 10 - 1 = 9 coins internes en hauteur                              │
# │    → CHECKERBOARD = (7, 9)                                              │
# │    → Total = 7 × 9 = 63 coins à détecter dans chaque image             │
# │                                                                         │
# │  Pourquoi les coins internes et pas les cases ?                         │
# │    → Les coins internes sont les seuls points que OpenCV peut           │
# │      localiser avec précision (jonction de 4 cases)                    │
# │    → Les bords extérieurs sont trop ambigus                             │
# └─────────────────────────────────────────────────────────────────────────┘

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# ┌─────────────────────────────────────────────────────────────────────────┐
# │  criteria = CRITÈRE D'ARRÊT pour l'algorithme d'affinement des coins   │
# │                                                                         │
# │  Ce critère sera utilisé par cornerSubPix (ÉTAPE 7.6)                  │
# │  Il dit à OpenCV : "cherche un coin plus précis, mais arrête-toi si..."│
# │                                                                         │
# │  Structure : (type_critere, max_iterations, precision_souhaitee)        │
# │                                                                         │
# │  cv2.TERM_CRITERIA_EPS :                                                │
# │    → EPS = epsilon = seuil de précision                                 │
# │    → S'arrête quand le déplacement du coin est < 0.001 pixel           │
# │    → "Je suis assez précis, inutile de continuer"                      │
# │                                                                         │
# │  cv2.TERM_CRITERIA_MAX_ITER :                                            │
# │    → S'arrête après 30 itérations maximum                              │
# │    → Évite une boucle infinie si la convergence est difficile           │
# │                                                                         │
# │  Le + (addition binaire) signifie : utiliser LES DEUX conditions       │
# │  → OpenCV s'arrête dès que l'UNE des deux est satisfaite               │
# │                                                                         │
# │  0.001 pixel = précision très fine                                      │
# │    (un pixel fait ~0.003mm sur un capteur smartphone)                  │
# └─────────────────────────────────────────────────────────────────────────┘


# =============================================================================
# ██████████████████████████████████████████████████████████████████████████
# ÉTAPE 3 : INITIALISATION DES LISTES DE POINTS
# ██████████████████████████████████████████████████████████████████████████
# =============================================================================

objpoints = []
# ┌─────────────────────────────────────────────────────────────────────────┐
# │  objpoints = "object points" = points 3D dans le monde réel            │
# │                                                                         │
# │  Contenu après traitement de N images :                                 │
# │    objpoints[0] = coordonnées 3D des 63 coins (image 1)                │
# │    objpoints[1] = coordonnées 3D des 63 coins (image 2)                │
# │    ...                                                                  │
# │    objpoints[N] = coordonnées 3D des 63 coins (image N)                │
# │                                                                         │
# │  ⚠️  IMPORTANT : c'est TOUJOURS le même tableau pour chaque image !    │
# │  Le damier physique ne change pas, donc ses coordonnées 3D non plus.   │
# │                                                                         │
# │  Shape de chaque élément : (1, 63, 3)                                  │
# │    → 1 = une seule "vue" du modèle                                      │
# │    → 63 = nombre de coins (7×9)                                        │
# │    → 3 = coordonnées (X, Y, Z)                                         │
# └─────────────────────────────────────────────────────────────────────────┘

imgpoints = []
# ┌─────────────────────────────────────────────────────────────────────────┐
# │  imgpoints = "image points" = points 2D détectés dans l'image          │
# │                                                                         │
# │  Contenu après traitement de N images :                                 │
# │    imgpoints[0] = pixels des 63 coins détectés (image 1)               │
# │    imgpoints[1] = pixels des 63 coins détectés (image 2)               │
# │    ...                                                                  │
# │    imgpoints[N] = pixels des 63 coins détectés (image N)               │
# │                                                                         │
# │  Ces valeurs sont DIFFÉRENTES pour chaque image car :                  │
# │    → La caméra (ou le damier) est à un angle différent                 │
# │    → La perspective déforme les positions des coins                    │
# │                                                                         │
# │  Shape de chaque élément : (63, 1, 2)                                  │
# │    → 63 = nombre de coins détectés                                     │
# │    → 1  = format interne OpenCV                                        │
# │    → 2  = coordonnées pixel (u, v) = (colonne, ligne)                  │
# │                                                                         │
# │  PRINCIPE DE LA CALIBRATION :                                           │
# │    objpoints[i] ←→ imgpoints[i]                                        │
# │    Point 3D réel   Point 2D dans l'image                               │
# │                                                                         │
# │    OpenCV cherche les paramètres K et dist tels que :                  │
# │    projection(objpoints[i], K, dist) ≈ imgpoints[i]                    │
# └─────────────────────────────────────────────────────────────────────────┘


# =============================================================================
# ██████████████████████████████████████████████████████████████████████████
# ÉTAPE 4 : DÉFINITION DES COORDONNÉES 3D DU DAMIER (MODÈLE)
# ██████████████████████████████████████████████████████████████████████████
# =============================================================================

objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
# ┌─────────────────────────────────────────────────────────────────────────┐
# │  np.zeros((1, 63, 3), np.float32)                                       │
# │                                                                         │
# │  Crée un tableau de ZÉROS de shape (1, 63, 3) :                        │
# │    Au départ :                                                          │
# │    [[[0., 0., 0.],                                                      │
# │      [0., 0., 0.],                                                      │
# │      [0., 0., 0.],                                                      │
# │      ...          ← 63 lignes                                           │
# │      [0., 0., 0.]]]                                                     │
# │                                                                         │
# │  np.float32 = type flottant 32 bits                                     │
# │    → OBLIGATOIRE pour OpenCV (il n'accepte pas float64 ici)            │
# │    → float32 = précision de ~7 décimales (suffisant)                   │
# └─────────────────────────────────────────────────────────────────────────┘

objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
# ┌─────────────────────────────────────────────────────────────────────────┐
# │  Décomposons cette ligne complexe partie par partie :                   │
# │                                                                         │
# │  1) np.mgrid[0:7, 0:9]                                                 │
# │     → génère une GRILLE de coordonnées 2D                              │
# │     → retourne 2 tableaux de shape (7, 9) :                            │
# │         grille_X = [[0,0,0,...,0],   ← X varie par lignes             │
# │                      [1,1,1,...,1],                                     │
# │                      ...                                               │
# │                      [6,6,6,...,6]]                                     │
# │         grille_Y = [[0,1,2,...,8],   ← Y varie par colonnes           │
# │                      [0,1,2,...,8],                                     │
# │                      ...                                               │
# │                      [0,1,2,...,8]]                                     │
# │                                                                         │
# │  2) .T  (transposée)                                                    │
# │     → interchange les axes pour aligner correctement X et Y            │
# │     → shape devient (9, 7) → (2, 9, 7) pour les deux grilles          │
# │                                                                         │
# │  3) .reshape(-1, 2)                                                     │
# │     → "aplatit" en un tableau de 63 lignes × 2 colonnes               │
# │     → -1 = "calcule automatiquement le nombre de lignes"               │
# │     → 2  = 2 valeurs par ligne (X, Y)                                  │
# │     → résultat : [[0,0], [0,1], [0,2], ..., [6,8]]                     │
# │                                                                         │
# │  4) objp[0, :, :2] = ...                                               │
# │     → remplit les colonnes 0 et 1 (X et Y) de objp                    │
# │     → la colonne 2 (Z) reste à 0 car le damier est PLAT               │
# │                                                                         │
# │  RÉSULTAT FINAL de objp :                                               │
# │    [[[0., 0., 0.],   ← coin (0,0)                                      │
# │      [0., 1., 0.],   ← coin (0,1)                                      │
# │      [0., 2., 0.],   ← coin (0,2)                                      │
# │      ...                                                                │
# │      [6., 8., 0.]]]  ← coin (6,8)                                      │
# │                                                                         │
# │  Note : 1 unité = 1 case du damier. Si les cases font 2.5cm,          │
# │  on pourrait écrire [0,0,0],[2.5,0,0]... pour avoir des mm.           │
# │  Pour la calibration intrinsèque, ça ne change pas K.                  │
# └─────────────────────────────────────────────────────────────────────────┘


# =============================================================================
# ██████████████████████████████████████████████████████████████████████████
# ÉTAPE 5 : PARAMÈTRES DE DÉTECTION DU DAMIER
# ██████████████████████████████████████████████████████████████████████████
# =============================================================================

nx = 7   # nombre de coins internes en X (horizontal / largeur)
ny = 9   # nombre de coins internes en Y (vertical  / hauteur)
# ┌─────────────────────────────────────────────────────────────────────────┐
# │  Ces deux variables "séparent" les valeurs de CHECKERBOARD              │
# │  pour les passer plus facilement à findChessboardCorners               │
# │                                                                         │
# │  nx = 7  (CHECKERBOARD[0]) = coins horizontaux                         │
# │  ny = 9  (CHECKERBOARD[1]) = coins verticaux                           │
# │                                                                         │
# │  ⚠️  OpenCV attend (nx, ny) = (colonnes, lignes) = (largeur, hauteur)  │
# │     C'est l'INVERSE de img.shape qui donne (hauteur, largeur)          │
# └─────────────────────────────────────────────────────────────────────────┘


# =============================================================================
# ██████████████████████████████████████████████████████████████████████████
# ÉTAPE 6 : CHARGEMENT DE TOUTES LES IMAGES DU DOSSIER
# ██████████████████████████████████████████████████████████████████████████
# =============================================================================

images = glob.glob('./images/*.jpg')
# ┌─────────────────────────────────────────────────────────────────────────┐
# │  glob.glob(pattern)                                                      │
# │                                                                         │
# │  Paramètre : un chemin avec wildcard *                                  │
# │    './images/*.jpg'                                                      │
# │    → . = dossier courant (là où est le script)                         │
# │    → /images/ = sous-dossier images                                    │
# │    → *.jpg = tous les fichiers finissant par .jpg                      │
# │                                                                         │
# │  Retourne : une LISTE de chemins                                        │
# │    exemple : ['./images/img1.jpg', './images/img2.jpg', ...]            │
# │                                                                         │
# │  Organisation recommandée du dossier :                                  │
# │    mon_projet/                                                           │
# │    ├── calibration_camera.py   ← ce fichier                            │
# │    ├── images/                                                           │
# │    │   ├── img1.jpg                                                      │
# │    │   ├── img2.jpg                                                      │
# │    │   └── ...  (minimum 10 images recommandées)                        │
# │    └── camera_params/          ← créé automatiquement                  │
# └─────────────────────────────────────────────────────────────────────────┘

# Vérification qu'il y a des images
if len(images) == 0:
    print("ERREUR : Aucune image trouvée dans ./images/")
    print("Vérifiez que le dossier 'images' existe et contient des .jpg")
    exit()

print(f"Nombre d'images trouvées : {len(images)}")
# f"..." = f-string : permet d'insérer des variables dans du texte
# len(images) = nombre d'éléments dans la liste

# Variables pour conserver la dernière image traitée
# (nécessaires pour gray.shape[::-1] à l'ÉTAPE 8)
img  = None
gray = None


# =============================================================================
# ██████████████████████████████████████████████████████████████████████████
# ÉTAPE 7 : TRAITEMENT DE CHAQUE IMAGE (BOUCLE PRINCIPALE)
# ██████████████████████████████████████████████████████████████████████████
# =============================================================================

for fname in images:
    # ┌─────────────────────────────────────────────────────────────────────┐
    # │  for fname in images :                                               │
    # │    → fname prend successivement la valeur de chaque chemin          │
    # │    → exemple : fname = './images/img1.jpg'                          │
    # │    →           fname = './images/img2.jpg'  etc.                    │
    # └─────────────────────────────────────────────────────────────────────┘
    print(f"\nTraitement de : {fname}")


    # ─────────────────────────────────────────────────────────────────────
    # 7.1 — LECTURE DE L'IMAGE
    # ─────────────────────────────────────────────────────────────────────
    img = cv2.imread(fname)
    # ┌─────────────────────────────────────────────────────────────────────┐
    # │  cv2.imread(chemin_fichier)                                          │
    # │                                                                     │
    # │  → Lit une image depuis le disque dur                               │
    # │  → Retourne un tableau NumPy de shape (hauteur, largeur, 3)         │
    # │  → Les 3 canaux de couleur sont dans l'ordre : B, G, R             │
    # │                                                                     │
    # │  ⚠️  OpenCV utilise BGR (Bleu-Vert-Rouge) et NON RGB !              │
    # │     (contrairement à matplotlib, PIL, etc.)                         │
    # │                                                                     │
    # │  Exemple pour une image 3024×4032 :                                 │
    # │    img.shape → (4032, 3024, 3)                                      │
    # │    img[0,0]  → [B, G, R] du pixel en haut à gauche                │
    # │    img[0,0]  → exemple : [45, 123, 200]                             │
    # └─────────────────────────────────────────────────────────────────────┘


    # ─────────────────────────────────────────────────────────────────────
    # 7.2 — RÉDUCTION DE LA TAILLE (÷ 2)
    # ─────────────────────────────────────────────────────────────────────
    new_width  = img.shape[1] // 2   # largeur divisée par 2
    new_height = img.shape[0] // 2   # hauteur divisée par 2
    # ┌─────────────────────────────────────────────────────────────────────┐
    # │  img.shape → (hauteur, largeur, canaux)                             │
    # │    img.shape[0] = hauteur (nombre de lignes)                        │
    # │    img.shape[1] = largeur (nombre de colonnes)                      │
    # │    img.shape[2] = canaux (= 3 pour une image couleur)               │
    # │                                                                     │
    # │  // = division entière (pas de virgule)                             │
    # │    exemple : 4032 // 2 = 2016  (et non 2016.0)                     │
    # └─────────────────────────────────────────────────────────────────────┘

    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    # ┌─────────────────────────────────────────────────────────────────────┐
    # │  cv2.resize(image, taille, interpolation)                           │
    # │                                                                     │
    # │  Paramètres :                                                       │
    # │    image         = image originale à redimensionner                 │
    # │    (new_w, new_h) = nouvelle taille en (LARGEUR, HAUTEUR)           │
    # │                   ⚠️ OpenCV veut (largeur, hauteur) ici            │
    # │                   (l'inverse de img.shape !)                        │
    # │    interpolation = méthode pour calculer les nouveaux pixels        │
    # │                                                                     │
    # │  cv2.INTER_AREA (le meilleur choix pour RÉDUIRE) :                  │
    # │    → Calcule la moyenne des pixels dans la zone réduite             │
    # │    → Évite les artefacts (effets de mosaïque)                      │
    # │    → Donne une image plus nette qu'INTER_LINEAR en réduction       │
    # │                                                                     │
    # │  Autres méthodes (pour référence) :                                 │
    # │    INTER_LINEAR  = bilinéaire (bon pour agrandir)                   │
    # │    INTER_CUBIC   = bicubique (très bon mais lent)                   │
    # │    INTER_NEAREST = plus proche voisin (pixelisé mais ultra-rapide) │
    # │                                                                     │
    # │  POURQUOI RÉDUIRE L'IMAGE ?                                         │
    # │    1. Les photos de smartphone font souvent 4000×3000 pixels       │
    # │    2. findChessboardCorners est LENT sur les grandes images         │
    # │    3. La réduction élimine le bruit de haute fréquence             │
    # │    4. Améliore la détection : moins de bruit = coins plus nets      │
    # │    5. Résultat : erreur finale de 0.071 pixel grâce à ça !         │
    # └─────────────────────────────────────────────────────────────────────┘


    # ─────────────────────────────────────────────────────────────────────
    # 7.3 — CONVERSION EN NIVEAUX DE GRIS
    # ─────────────────────────────────────────────────────────────────────
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # ┌─────────────────────────────────────────────────────────────────────┐
    # │  cv2.cvtColor(image, code_de_conversion)                            │
    # │                                                                     │
    # │  Paramètres :                                                       │
    # │    image              = image couleur BGR                           │
    # │    cv2.COLOR_BGR2GRAY = code de conversion BGR → Niveaux de gris   │
    # │                                                                     │
    # │  Formule appliquée par OpenCV :                                     │
    # │    Gris = 0.114×B + 0.587×G + 0.299×R                              │
    # │    (les coefficients suivent la sensibilité de l'œil humain)       │
    # │                                                                     │
    # │  Avant : shape (hauteur, largeur, 3) → valeurs [B, G, R]           │
    # │  Après : shape (hauteur, largeur)    → valeur  0 à 255             │
    # │          0 = noir, 255 = blanc                                      │
    # │                                                                     │
    # │  Autres conversions utiles (pour référence) :                      │
    # │    COLOR_BGR2RGB   = BGR → RGB (pour matplotlib)                    │
    # │    COLOR_BGR2HSV   = BGR → Teinte/Saturation/Valeur                 │
    # │    COLOR_GRAY2BGR  = Gris → couleur (pour dessiner en couleur)     │
    # │                                                                     │
    # │  POURQUOI CONVERTIR EN GRIS ?                                       │
    # │    1. findChessboardCorners REQUIERT une image en gris              │
    # │    2. Le damier est noir et blanc : la couleur n'apporte rien      │
    # │    3. 1 canal au lieu de 3 = 3× plus rapide à traiter              │
    # │    4. Meilleure détection du contraste noir/blanc                  │
    # └─────────────────────────────────────────────────────────────────────┘


    # ─────────────────────────────────────────────────────────────────────
    # 7.4 — DÉTECTION DES COINS DU DAMIER
    # ─────────────────────────────────────────────────────────────────────
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
    # ┌─────────────────────────────────────────────────────────────────────┐
    # │  cv2.findChessboardCorners(image, patternSize, flags)               │
    # │                                                                     │
    # │  C'est la fonction de DÉTECTION AUTOMATIQUE du damier.             │
    # │  Elle scanne l'image pour trouver la grille de coins.              │
    # │                                                                     │
    # │  Paramètres :                                                       │
    # │    image       = gray = image en NIVEAUX DE GRIS (obligatoire)     │
    # │    patternSize = (nx, ny) = (colonnes, lignes) de coins internes   │
    # │                 ⚠️ OpenCV veut (largeur, hauteur) donc (nx, ny)   │
    # │                 et NON (ny, nx) !                                   │
    # │    flags       = None = détection standard sans options             │
    # │                 Autres options possibles :                          │
    # │                 CALIB_CB_ADAPTIVE_THRESH (images sombres)          │
    # │                 CALIB_CB_NORMALIZE_IMAGE (lumière inégale)         │
    # │                                                                     │
    # │  Retourne :                                                         │
    # │    ret     = True  si les 63 coins sont TOUS trouvés               │
    # │              False si le damier est invisible/incomplet            │
    # │    corners = tableau shape (63, 1, 2) si ret=True                  │
    # │              → 63 coins détectés                                   │
    # │              → coordonnées (u, v) en pixels                        │
    # │              → précision ~1 pixel (sera affinée après)             │
    # │                                                                     │
    # │  CAS OÙ ret = False :                                               │
    # │    → Le damier est trop incliné (> 60°)                            │
    # │    → Le damier est partiellement hors du cadre                     │
    # │    → L'image est floue                                              │
    # │    → Mauvais éclairage (reflets, ombres)                           │
    # │    → nx ou ny sont incorrects                                      │
    # └─────────────────────────────────────────────────────────────────────┘


    # ─────────────────────────────────────────────────────────────────────
    # 7.5 — TRAITEMENT SI LE DAMIER EST DÉTECTÉ
    # ─────────────────────────────────────────────────────────────────────
    if ret == True:
        print(f"  ✓ Damier détecté ({nx}×{ny} = {nx*ny} coins)")

        # On ajoute les points 3D du modèle pour cette image
        # (toujours le même tableau objp)
        objpoints.append(objp)
        # ┌─────────────────────────────────────────────────────────────────┐
        # │  .append(element)                                               │
        # │  → ajoute un élément à la fin d'une liste Python               │
        # │  → objpoints grandit d'un élément à chaque image valide        │
        # └─────────────────────────────────────────────────────────────────┘


        # ─────────────────────────────────────────────────────────────────
        # 7.6 — AFFINEMENT DES COINS À LA PRÉCISION SOUS-PIXEL
        # ─────────────────────────────────────────────────────────────────
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        # ┌─────────────────────────────────────────────────────────────────┐
        # │  cv2.cornerSubPix(image, corners, winSize, zeroZone, criteria) │
        # │                                                                 │
        # │  "SubPix" = Sub-Pixel = précision SUB-PIXEL                    │
        # │                                                                 │
        # │  Problème : findChessboardCorners donne des positions entières │
        # │    coin trouvé à (125, 87) = position au pixel près            │
        # │    Insuffisant pour une bonne calibration !                    │
        # │                                                                 │
        # │  Solution : cornerSubPix affine chaque coin en sous-pixel      │
        # │    coin affiné → (125.34, 87.12) = beaucoup plus précis !     │
        # │                                                                 │
        # │  Paramètres :                                                   │
        # │    image    = gray = image en niveaux de gris                  │
        # │    corners  = coins détectés grossièrement                      │
        # │                                                                 │
        # │    winSize  = (11, 11) = FENÊTRE DE RECHERCHE                  │
        # │      → cherche la position exacte dans une zone 11×11 pixels  │
        # │      → (11,11) signifie ±11 pixels autour de la position       │
        # │        initiale, soit une fenêtre totale de 23×23 pixels      │
        # │      → Plus grand = cherche plus loin                          │
        # │      → Trop grand = risque de confondre deux coins             │
        # │      → (11,11) est la valeur standard recommandée              │
        # │                                                                 │
        # │    zeroZone = (-1, -1) = ZONE MORTE au centre désactivée      │
        # │      → (-1,-1) = pas de zone morte (désactivée)               │
        # │      → La zone morte évite les singularités numériques         │
        # │        causées par l'autocorrélation                           │
        # │      → En pratique (-1,-1) marche très bien                   │
        # │                                                                 │
        # │    criteria = critère d'arrêt défini à l'ÉTAPE 2              │
        # │      → arrête quand précision < 0.001 pixel OU 30 itérations  │
        # │                                                                 │
        # │  Algorithme interne (simplifié) :                              │
        # │    1. Prend la position initiale du coin (pixel entier)        │
        # │    2. Calcule le gradient de l'image autour du coin            │
        # │    3. Déplace le coin vers le maximum du gradient              │
        # │    4. Répète jusqu'au critère d'arrêt                         │
        # │                                                                 │
        # │  Résultat :                                                     │
        # │    corners2 = mêmes 63 coins mais avec précision < 0.001 pixel│
        # │    → C'est CE QUI permet d'avoir une erreur finale de 0.071 ! │
        # └─────────────────────────────────────────────────────────────────┘

        # Stockage des coins affinés pour la calibration
        imgpoints.append(corners2)


        # ─────────────────────────────────────────────────────────────────
        # 7.7 — DESSIN DES COINS SUR L'IMAGE (visualisation)
        # ─────────────────────────────────────────────────────────────────
        img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        # ┌─────────────────────────────────────────────────────────────────┐
        # │  cv2.drawChessboardCorners(image, patternSize, corners, ret)   │
        # │                                                                 │
        # │  Dessine les coins et les lignes du damier sur l'image         │
        # │  pour vérification visuelle.                                   │
        # │                                                                 │
        # │  Paramètres :                                                   │
        # │    image       = img = image COULEUR BGR (pas gray !)          │
        # │    patternSize = CHECKERBOARD = (7, 9)                         │
        # │    corners     = corners2 = coins affinés                      │
        # │    ret         = True = tous les coins ont été trouvés         │
        # │                                                                 │
        # │  Ce que ça dessine :                                            │
        # │    → Un cercle coloré sur chaque coin                          │
        # │    → Des lignes reliant les coins                              │
        # │    → Les couleurs changent arc-en-ciel par ligne               │
        # │      (rouge, orange, jaune, vert, bleu, violet...)             │
        # │                                                                 │
        # │  Comment vérifier que c'est correct ?                          │
        # │    ✅ Les lignes sont parallèles entre elles → correct         │
        # │    ❌ Les lignes sont croisées/obliques → nx/ny inversés        │
        # │    ❌ Seulement quelques coins → damier mal détecté            │
        # └─────────────────────────────────────────────────────────────────┘
        print(f"  ✓ Coins affinés et dessinés")

    else:
        print(f"  ✗ Damier NON détecté dans cette image → ignorée")


    # ─────────────────────────────────────────────────────────────────────
    # 7.8 — AFFICHAGE DE L'IMAGE
    # ─────────────────────────────────────────────────────────────────────
    cv2.imshow('img', img)
    # ┌─────────────────────────────────────────────────────────────────────┐
    # │  cv2.imshow(nom_fenetre, image)                                     │
    # │                                                                     │
    # │  Paramètres :                                                       │
    # │    nom_fenetre = 'img' = titre affiché dans la barre de la fenêtre │
    # │    image       = img  = l'image à afficher (avec coins dessinés)   │
    # │                                                                     │
    # │  ⚠️ Cette fonction NE FAIT QUE afficher l'image.                   │
    # │     Elle ne "bloque" pas le programme. C'est waitKey qui bloque.   │
    # └─────────────────────────────────────────────────────────────────────┘

    cv2.waitKey(0)
    # ┌─────────────────────────────────────────────────────────────────────┐
    # │  cv2.waitKey(delai_ms)                                              │
    # │                                                                     │
    # │  Paramètre :                                                        │
    # │    0 = attendre indéfiniment jusqu'à ce qu'une touche soit pressée │
    # │    n = attendre n millisecondes (puis continuer automatiquement)    │
    # │                                                                     │
    # │  Retourne : le code ASCII de la touche pressée                     │
    # │    exemple : waitKey(0) == ord('q')  → quitter si on appuie 'q'   │
    # │                                                                     │
    # │  IMPORTANT : waitKey est OBLIGATOIRE pour que imshow fonctionne    │
    # │    Sans waitKey, la fenêtre s'ouvre et se ferme instantanément     │
    # │    Appuyez sur n'importe quelle touche → image suivante            │
    # └─────────────────────────────────────────────────────────────────────┘

# Ferme toutes les fenêtres OpenCV ouvertes
cv2.destroyAllWindows()
# ┌─────────────────────────────────────────────────────────────────────────┐
# │  cv2.destroyAllWindows()                                                │
# │  → ferme TOUTES les fenêtres créées par imshow                        │
# │  → Bonne pratique : toujours appeler à la fin                         │
# └─────────────────────────────────────────────────────────────────────────┘


# =============================================================================
# ██████████████████████████████████████████████████████████████████████████
# ÉTAPE 8 : VÉRIFICATION AVANT CALIBRATION
# ██████████████████████████████████████████████████████████████████████████
# =============================================================================

if len(objpoints) == 0:
    print("\nERREUR : Aucun damier détecté dans aucune image.")
    print("Vérifiez vos images et les paramètres nx, ny.")
    exit()

print(f"\nCalibration avec {len(objpoints)} images valides sur {len(images)} images...")

h, w = img.shape[:2]
# ┌─────────────────────────────────────────────────────────────────────────┐
# │  img.shape[:2]                                                          │
# │                                                                         │
# │  img.shape      → (hauteur, largeur, 3)                                │
# │  img.shape[:2]  → (hauteur, largeur)   ← coupe le canal couleur       │
# │                                                                         │
# │  Décomposition :                                                        │
# │    h, w = img.shape[:2]                                                │
# │    h = img.shape[0] = hauteur                                          │
# │    w = img.shape[1] = largeur                                          │
# │                                                                         │
# │  Ces valeurs servent à calibrateCamera (ÉTAPE 9)                      │
# └─────────────────────────────────────────────────────────────────────────┘


# =============================================================================
# ██████████████████████████████████████████████████████████████████████████
# ÉTAPE 9 : CALIBRATION DE LA CAMÉRA — LA FONCTION PRINCIPALE
# ██████████████████████████████████████████████████████████████████████████
# =============================================================================

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)
# ┌─────────────────────────────────────────────────────────────────────────┐
# │  cv2.calibrateCamera(objectPoints, imagePoints, imageSize, K, dist)    │
# │                                                                         │
# │  C'est LA FONCTION CENTRALE de tout le programme.                      │
# │  Elle résout un problème d'optimisation pour trouver les paramètres    │
# │  qui minimisent l'erreur entre les points prédits et les points réels. │
# │                                                                         │
# │  ═══════════════════════════════════════════════════════════════        │
# │  PARAMÈTRES D'ENTRÉE                                                   │
# │  ═══════════════════════════════════════════════════════════════        │
# │                                                                         │
# │  objectPoints = objpoints                                               │
# │    → liste des tableaux de points 3D (un tableau par image)            │
# │    → shape de chaque élément : (1, 63, 3)                             │
# │    → représente le damier physique                                     │
# │                                                                         │
# │  imagePoints = imgpoints                                                │
# │    → liste des tableaux de points 2D détectés                         │
# │    → shape de chaque élément : (63, 1, 2)                             │
# │    → représente ce que la caméra a vu dans chaque image               │
# │                                                                         │
# │  imageSize = gray.shape[::-1]                                           │
# │    gray.shape     → (hauteur, largeur)                                  │
# │    gray.shape[::-1] → (largeur, hauteur)  ← INVERSION                  │
# │    ⚠️ OpenCV attend (largeur, hauteur) ici, mais img.shape donne       │
# │       (hauteur, largeur). Le [::-1] renverse le tuple.                 │
# │                                                                         │
# │  None, None = cameraMatrix initiale, distCoeffs initiaux               │
# │    → On dit à OpenCV d'initialiser lui-même ces valeurs               │
# │    → On pourrait fournir une estimation initiale pour aller plus vite  │
# │                                                                         │
# │  ═══════════════════════════════════════════════════════════════        │
# │  RÉSULTATS RETOURNÉS                                                    │
# │  ═══════════════════════════════════════════════════════════════        │
# │                                                                         │
# │  ret = Erreur RMS (Root Mean Square) en pixels                         │
# │    → Mesure la QUALITÉ GLOBALE de la calibration                       │
# │    → Formule : √( moyenne des (distance_réelle - distance_prédite)² ) │
# │    → Votre valeur : 1.006 pixel                                        │
# │    → < 0.5 : Excellent | 0.5-1.0 : Bon | > 1.0 : Médiocre            │
# │                                                                         │
# │  mtx = Matrice intrinsèque K (3×3) :                                   │
# │                                                                         │
# │         ┌ fx   0  cx ┐                                                 │
# │     K = │  0  fy  cy │                                                 │
# │         └  0   0   1 ┘                                                 │
# │                                                                         │
# │    fx = focale horizontale en pixels                                    │
# │         → "combien de pixels correspond à 1 radian horizontalement"   │
# │    fy = focale verticale en pixels (≈ fx pour un bon objectif)        │
# │    cx = coordonnée X du centre optique (≈ largeur/2)                   │
# │    cy = coordonnée Y du centre optique (≈ hauteur/2)                   │
# │                                                                         │
# │    Vos valeurs : fx=468.8, fy=470.6, cx=174.7, cy=324.7               │
# │    Ratio fx/fy = 0.9962 ≈ 1.0 → objectif symétrique ✅                │
# │                                                                         │
# │  dist = Coefficients de distorsion [k1, k2, p1, p2, k3]               │
# │                                                                         │
# │    k1, k2, k3 = distorsion RADIALE                                     │
# │      → déformation en barillet (bords courbés vers l'extérieur)       │
# │      → ou en coussinet (bords courbés vers l'intérieur)               │
# │      → k1 < 0 = barillet (grand-angle), k1 > 0 = coussinet           │
# │                                                                         │
# │    p1, p2 = distorsion TANGENTIELLE                                    │
# │      → causée par un objectif non parfaitement centré                 │
# │      → en pratique très faible (< 0.01)                               │
# │                                                                         │
# │    Vos valeurs : [0.227, -1.259, 0.014, -0.0005, 2.800]              │
# │                                                                         │
# │  rvecs = Vecteurs de rotation (un par image)                           │
# │    → représentation de Rodrigues (3 valeurs = axe × angle)            │
# │    → indique l'orientation du damier par rapport à la caméra          │
# │    → pas utile pour la calibration intrinsèque mais nécessaire        │
# │      pour calibrateCamera                                              │
# │                                                                         │
# │  tvecs = Vecteurs de translation (un par image)                        │
# │    → position du damier dans le repère caméra                         │
# │    → tvecs[i][2] ≈ distance caméra-damier                             │
# └─────────────────────────────────────────────────────────────────────────┘


# =============================================================================
# ██████████████████████████████████████████████████████████████████████████
# ÉTAPE 10 : AFFICHAGE DES RÉSULTATS
# ██████████████████████████████████████████████████████████████████████████
# =============================================================================

print("\n" + "="*60)
print("RÉSULTATS DE LA CALIBRATION")
print("="*60)

print(f"\nErreur RMS (ret) : {ret:.6f} pixels")
# :.6f = format flottant avec 6 décimales (exemple : 1.006123)
print("  → < 0.5 : Excellent | 0.5-1.0 : Bon | > 1.0 : Mauvais")

print("\nMatrice intrinsèque K :")
print(mtx)
print(f"  → Focale X  (fx) : {mtx[0,0]:.4f} pixels")
# mtx[0,0] = ligne 0, colonne 0 de la matrice K = fx
print(f"  → Focale Y  (fy) : {mtx[1,1]:.4f} pixels")
# mtx[1,1] = ligne 1, colonne 1 de la matrice K = fy
print(f"  → Centre optique cx : {mtx[0,2]:.4f} pixels")
# mtx[0,2] = ligne 0, colonne 2 de la matrice K = cx
print(f"  → Centre optique cy : {mtx[1,2]:.4f} pixels")
# mtx[1,2] = ligne 1, colonne 2 de la matrice K = cy
print(f"  → Ratio fx/fy : {mtx[0,0]/mtx[1,1]:.4f} (idéal = 1.0)")

print("\nCoefficients de distorsion [k1, k2, p1, p2, k3] :")
print(dist)
print("  → k1, k2, k3 : distorsion radiale")
print("  → p1, p2     : distorsion tangentielle")

print("\nVecteurs de rotation (un par image) :")
for i, r in enumerate(rvecs):
    # enumerate() donne l'index i ET la valeur r en même temps
    print(f"  Image {i+1:2d} : {r.T}")
    # {i+1:2d} = entier sur 2 chiffres (alignement)
    # .T = transposée (pour afficher horizontalement)

print("\nVecteurs de translation (un par image) :")
for i, t in enumerate(tvecs):
    print(f"  Image {i+1:2d} : {t.T}  (Z = distance caméra-damier)")


# =============================================================================
# ██████████████████████████████████████████████████████████████████████████
# ÉTAPE 11 : SAUVEGARDE DES PARAMÈTRES
# ██████████████████████████████████████████████████████████████████████████
# =============================================================================

os.makedirs("./camera_params", exist_ok=True)
# ┌─────────────────────────────────────────────────────────────────────────┐
# │  os.makedirs(chemin, exist_ok=True)                                     │
# │                                                                         │
# │  → Crée le dossier camera_params/ si il n'existe pas                   │
# │  → exist_ok=True : ne plante pas si le dossier existe déjà             │
# │  → Sans exist_ok=True → FileExistsError si le dossier est déjà créé   │
# └─────────────────────────────────────────────────────────────────────────┘

np.save("./camera_params/ret",   ret)
np.save("./camera_params/mtx",   mtx)
np.save("./camera_params/dist",  dist)
np.save("./camera_params/rvecs", rvecs)
np.save("./camera_params/tvecs", tvecs)
# ┌─────────────────────────────────────────────────────────────────────────┐
# │  np.save(chemin, tableau)                                               │
# │                                                                         │
# │  → Sauvegarde un tableau NumPy dans un fichier binaire .npy            │
# │  → Format binaire compact (pas lisible avec un éditeur texte)          │
# │  → Rapide à écrire et à relire                                         │
# │  → OpenCV ajoute automatiquement l'extension .npy si non précisée     │
# │                                                                         │
# │  POUR RECHARGER CES FICHIERS DANS UN AUTRE SCRIPT :                    │
# │    mtx  = np.load("./camera_params/mtx.npy")                           │
# │    dist = np.load("./camera_params/dist.npy")                          │
# │    ret  = np.load("./camera_params/ret.npy")                           │
# │                                                                         │
# │  CES FICHIERS SERVIRONT DANS LES ÉTAPES SUIVANTES :                   │
# │    mtx.npy  → corriger la distorsion des images                        │
# │    dist.npy → undistort avant SIFT / stéréovision                     │
# │    mtx + baseline → calculer la profondeur en stéréovision            │
# └─────────────────────────────────────────────────────────────────────────┘

print("\n" + "="*60)
print("Paramètres sauvegardés dans ./camera_params/")
print("  ret.npy   → erreur de reprojection globale")
print("  mtx.npy   → matrice intrinsèque K (focales + centre)")
print("  dist.npy  → coefficients de distorsion")
print("  rvecs.npy → rotations du damier par image")
print("  tvecs.npy → translations du damier par image")
print("="*60)


# =============================================================================
# ██████████████████████████████████████████████████████████████████████████
# ÉTAPE 12 : CALCUL DE L'ERREUR DE REPROJECTION PAR IMAGE
# ██████████████████████████████████████████████████████████████████████████
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────
# Bloc de calcul intermédiaire (non utilisé ici mais montré pour comprendre)
# ─────────────────────────────────────────────────────────────────────────
tot_error    = 0
total_points = 0

for i in range(len(objpoints)):
    reprojected_points, _ = cv2.projectPoints(
        objpoints[i], rvecs[i], tvecs[i], mtx, dist
    )
    # ┌─────────────────────────────────────────────────────────────────────┐
    # │  cv2.projectPoints(pts3D, rvec, tvec, K, dist)                     │
    # │                                                                     │
    # │  → Projette des points 3D en 2D selon les paramètres calibrés      │
    # │  → Simule ce que la caméra verrait avec les paramètres trouvés     │
    # │                                                                     │
    # │  Paramètres :                                                       │
    # │    pts3D  = objpoints[i] = 63 points 3D de l'image i              │
    # │    rvec   = rvecs[i]    = vecteur rotation de l'image i            │
    # │    tvec   = tvecs[i]    = vecteur translation de l'image i         │
    # │    K      = mtx         = matrice intrinsèque calibrée             │
    # │    dist   = dist        = distorsion calibrée                      │
    # │                                                                     │
    # │  Retourne :                                                         │
    # │    reprojected_points = 63 points 2D prédits par le modèle         │
    # │    _                  = jacobienne (dérivées, non utilisée ici)     │
    # └─────────────────────────────────────────────────────────────────────┘
    reprojected_points = reprojected_points.reshape(-1, 2)

# ─────────────────────────────────────────────────────────────────────────
# Calcul et affichage de l'erreur par image
# ─────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("ERREUR DE REPROJECTION PAR IMAGE")
print("="*60)

mean_error = 0
for i in range(len(objpoints)):

    imgpoints2, _ = cv2.projectPoints(
        objpoints[i], rvecs[i], tvecs[i], mtx, dist
    )
    # → projette les 63 points 3D en 2D avec les paramètres calibrés
    # → imgpoints2 = positions PRÉDITES par le modèle

    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    # ┌─────────────────────────────────────────────────────────────────────┐
    # │  cv2.norm(A, B, type_norme)                                         │
    # │                                                                     │
    # │  → Calcule la distance entre deux ensembles de points              │
    # │                                                                     │
    # │  Paramètres :                                                       │
    # │    A           = imgpoints[i]  = positions RÉELLES (détectées)     │
    # │    B           = imgpoints2    = positions PRÉDITES (projetées)     │
    # │    cv2.NORM_L2 = norme L2      = distance euclidienne              │
    # │                                                                     │
    # │  Formule de NORM_L2 :                                               │
    # │    √( Σᵢ (xᵢ_réel - xᵢ_prédit)² + (yᵢ_réel - yᵢ_prédit)² )      │
    # │    = distance totale cumulée entre tous les coins                  │
    # │                                                                     │
    # │  / len(imgpoints2) :                                                │
    # │    → divise par le nombre de coins (63)                            │
    # │    → obtient l'erreur MOYENNE par coin en pixels                   │
    # │                                                                     │
    # │  Interprétation du résultat :                                       │
    # │    0.05 pixel → le modèle prédit les coins à ±0.05 pixel près     │
    # │    C'est excellent ! (1 pixel ≈ quelques micromètres sur capteur)  │
    # └─────────────────────────────────────────────────────────────────────┘

    mean_error += error
    print(f"  Image {i+1:2d} : {error:.5f} pixels")

# Calcul de l'erreur totale moyenne sur toutes les images
total_error = mean_error / len(objpoints)
print(f"\nErreur totale moyenne : {total_error:.8f} pixels")
print("  → Votre valeur 0.071 = EXCELLENT ✅")
print("  → En dessous de 0.1 pixel : calibration de haute précision")

print("\n" + "="*60)
print("✓ Calibration terminée avec succès !")
print("→ Prochaine étape : acquérir 2 images stéréo avec translation")
print("="*60)