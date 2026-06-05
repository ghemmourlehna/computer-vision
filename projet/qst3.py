#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
=============================================================================
QST3 - DÉTECTION SIFT + MISE EN CORRESPONDANCE ROBUSTE POUR STÉRÉOVISION
=============================================================================

OBJECTIF :
À partir de deux images d'une même scène prises avec un léger déplacement
horizontal, on veut :

1) détecter des points d'intérêt dans chaque image,
2) associer les points correspondants entre image gauche et image droite,
3) supprimer les faux appariements,
4) sauvegarder les vraies paires de points pour la reconstruction 3D.

PIPELINE UTILISÉ :
- cv2.imread                -> lecture des images
- cv2.cvtColor             -> conversion en niveaux de gris
- cv2.SIFT_create          -> création du détecteur/descripteur SIFT
- sift.detectAndCompute    -> détection des keypoints + descripteurs
- cv2.FlannBasedMatcher    -> matching rapide entre descripteurs
- flann.knnMatch           -> 2 plus proches voisins pour chaque descripteur
- Lowe ratio test          -> suppression des matches ambigus
- cv2.findFundamentalMat   -> calcul de la géométrie épipolaire + RANSAC
- filtrage stéréo final    -> cohérence disparité + différence verticale
- cv2.drawKeypoints        -> dessin des points détectés
- cv2.drawMatches          -> dessin des correspondances
- np.save                  -> sauvegarde des points pour qst4.py

SORTIES :
- resultats_sift/keypoints_gauche.png
- resultats_sift/keypoints_droite.png
- resultats_sift/matches_apres_lowe.png
- resultats_sift/matches_finaux.png
- resultats_sift/pts_gauche.npy
- resultats_sift/pts_droite.npy
=============================================================================
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# ██████████████████████████████████████████████████████████████████████████
# ÉTAPE 1 : CHEMINS DU PROJET
# ██████████████████████████████████████████████████████████████████████████
# =============================================================================

BASE_DIR = r"C:\Users\SALOUA\Desktop\TPS\Tp Vision\projet\TP3CV_2026\TP3CV_2026\projet"
# ┌─────────────────────────────────────────────────────────────────────────┐
# │ BASE_DIR = dossier racine où se trouvent :                              │
# │   - les images source (gauche.jpg, droite.jpg)                          │
# │   - le dossier resultats_sift/ qui sera créé pour sauvegarder           │
# │     les résultats du matching                                           │
# └─────────────────────────────────────────────────────────────────────────┘

OUT_DIR = os.path.join(BASE_DIR, "resultats_sift")
# ┌─────────────────────────────────────────────────────────────────────────┐
# │ os.path.join(a, b)                                                      │
# │   → construit un chemin correct selon Windows / Linux                   │
# │   → ici : OUT_DIR = BASE_DIR + "\\resultats_sift"                        │
# └─────────────────────────────────────────────────────────────────────────┘

os.makedirs(OUT_DIR, exist_ok=True)
# ┌─────────────────────────────────────────────────────────────────────────┐
# │ os.makedirs(dossier, exist_ok=True)                                     │
# │   → crée le dossier s'il n'existe pas                                   │
# │   → exist_ok=True évite une erreur si le dossier existe déjà            │
# └─────────────────────────────────────────────────────────────────────────┘


# IMPORTANT :
# Ici on garde TON ordre actuel car c'est lui qui donne une disparité correcte
LEFT_NAME = "droite.jpg"
RIGHT_NAME = "gauche.jpg"

LEFT_PATH = os.path.join(BASE_DIR, LEFT_NAME)
RIGHT_PATH = os.path.join(BASE_DIR, RIGHT_NAME)
# ┌─────────────────────────────────────────────────────────────────────────┐
# │ Convention utilisée dans le script :                                    │
# │   LEFT_PATH  = image considérée comme vue gauche                        │
# │   RIGHT_PATH = image considérée comme vue droite                        │
# │                                                                         │
# │ Même si les noms des fichiers sont "droite.jpg" et "gauche.jpg",        │
# │ on garde cet ordre car c'est celui qui t'a donné de bonnes disparités.  │
# └─────────────────────────────────────────────────────────────────────────┘


# =============================================================================
# ██████████████████████████████████████████████████████████████████████████
# ÉTAPE 2 : CHARGEMENT DES IMAGES
# ██████████████████████████████████████████████████████████████████████████
# =============================================================================

img_g = cv2.imread(LEFT_PATH)
img_d = cv2.imread(RIGHT_PATH)
# ┌─────────────────────────────────────────────────────────────────────────┐
# │ cv2.imread(chemin_fichier)                                              │
# │                                                                         │
# │ BUT : lire une image depuis le disque.                                  │
# │                                                                         │
# │ ENTRÉE :                                                                │
# │   - chemin complet vers une image (.jpg, .png, ...)                     │
# │                                                                         │
# │ SORTIE :                                                                │
# │   - un tableau NumPy de shape (hauteur, largeur, 3)                     │
# │   - les 3 canaux sont stockés en ordre BGR et NON RGB                   │
# │                                                                         │
# │ EXEMPLE :                                                               │
# │   image 1280x721 couleur → shape = (721, 1280, 3)                       │
# │                                                                         │
# │ SI ÉCHEC :                                                              │
# │   retourne None                                                         │
# │   (mauvais chemin, fichier absent, nom erroné, extension erronée)       │
# └─────────────────────────────────────────────────────────────────────────┘

if img_g is None:
    raise FileNotFoundError(f"Image gauche introuvable : {LEFT_PATH}")
if img_d is None:
    raise FileNotFoundError(f"Image droite introuvable : {RIGHT_PATH}")

if img_g.shape[:2] != img_d.shape[:2]:
    raise ValueError("Les deux images n'ont pas la même taille. Il faut des images de même dimension.")
# ┌─────────────────────────────────────────────────────────────────────────┐
# │ img.shape[:2] donne uniquement (hauteur, largeur)                       │
# │                                                                         │
# │ Pourquoi vérifier ?                                                     │
# │   En stéréovision, les deux images doivent avoir la même résolution     │
# │   pour que les coordonnées de points soient comparables.                │
# └─────────────────────────────────────────────────────────────────────────┘


gray_g = cv2.cvtColor(img_g, cv2.COLOR_BGR2GRAY)
gray_d = cv2.cvtColor(img_d, cv2.COLOR_BGR2GRAY)
# ┌─────────────────────────────────────────────────────────────────────────┐
# │ cv2.cvtColor(image, code_conversion)                                    │
# │                                                                         │
# │ BUT : convertir une image d'un espace couleur à un autre.               │
# │                                                                         │
# │ ICI :                                                                   │
# │   cv2.COLOR_BGR2GRAY                                                    │
# │   → convertit une image couleur BGR (3 canaux) en image grise (1 canal) │
# │                                                                         │
# │ POURQUOI ?                                                              │
# │   SIFT travaille classiquement sur les intensités (niveaux de gris),    │
# │   pas directement sur la couleur.                                       │
# │                                                                         │
# │ ENTRÉE :                                                                │
# │   image couleur shape (H, W, 3)                                         │
# │                                                                         │
# │ SORTIE :                                                                │
# │   image grise shape (H, W)                                              │
# └─────────────────────────────────────────────────────────────────────────┘

h, w = gray_g.shape

print("=" * 70)
print("CHARGEMENT DES IMAGES")
print("=" * 70)
print(f"Image gauche : {LEFT_PATH}")
print(f"Image droite : {RIGHT_PATH}")
print(f"Taille image : {w} x {h}")


# =============================================================================
# ██████████████████████████████████████████████████████████████████████████
# ÉTAPE 3 : CRÉATION DU DÉTECTEUR SIFT
# ██████████████████████████████████████████████████████████████████████████
# =============================================================================

sift = cv2.SIFT_create(
    nfeatures=4000,
    contrastThreshold=0.01,
    edgeThreshold=10,
    sigma=1.6
)
# ┌─────────────────────────────────────────────────────────────────────────┐
# │ cv2.SIFT_create(...)                                                    │
# │                                                                         │
# │ BUT : créer un objet SIFT capable de :                                  │
# │   - détecter les points d'intérêt (keypoints)                           │
# │   - calculer les descripteurs associés                                  │
# │                                                                         │
# │ PRINCIPE DE SIFT :                                                      │
# │   SIFT = Scale-Invariant Feature Transform                              │
# │   → détecte des points remarquables : coins, textures, intersections    │
# │   → cherche des points robustes aux changements :                       │
# │        - d'échelle (zoom)                                               │
# │        - de rotation                                                    │
# │        - partiellement d'illumination                                   │
# │                                                                         │
# │ POUR CHAQUE POINT, SIFT CALCULE :                                       │
# │   1) sa position (x, y)                                                 │
# │   2) son échelle                                                        │
# │   3) son orientation                                                    │
# │   4) un descripteur de 128 valeurs                                      │
# │                                                                         │
# │ PARAMÈTRES :                                                            │
# │   nfeatures=4000                                                        │
# │      → nombre maximal de meilleurs points à conserver                   │
# │      → plus grand = plus de points, mais plus lent                      │
# │                                                                         │
# │   contrastThreshold=0.01                                                │
# │      → seuil de contraste minimum pour accepter un point                │
# │      → plus petit = plus de points détectés                             │
# │      → trop petit = plus de bruit / points faibles                      │
# │                                                                         │
# │   edgeThreshold=10                                                      │
# │      → rejette les points trop "linéaires" comme certaines arêtes       │
# │      → plus grand = garde plus de points proches des bords              │
# │                                                                         │
# │   sigma=1.6                                                             │
# │      → flou gaussien initial utilisé par SIFT                           │
# │      → valeur standard classique                                        │
# └─────────────────────────────────────────────────────────────────────────┘


kp_g, des_g = sift.detectAndCompute(gray_g, None)
kp_d, des_d = sift.detectAndCompute(gray_d, None)
# ┌─────────────────────────────────────────────────────────────────────────┐
# │ sift.detectAndCompute(image, mask)                                      │
# │                                                                         │
# │ BUT : faire en une seule étape :                                        │
# │   - detect()  → trouver les keypoints                                   │
# │   - compute() → calculer les descripteurs                               │
# │                                                                         │
# │ ENTRÉES :                                                               │
# │   image = image en niveaux de gris                                      │
# │   mask  = None                                                          │
# │      → None signifie : traiter toute l'image                            │
# │      → si on donnait un masque, SIFT travaillerait seulement            │
# │        dans une région précise                                          │
# │                                                                         │
# │ SORTIES :                                                               │
# │   kp_g / kp_d = liste de cv2.KeyPoint                                   │
# │      chaque KeyPoint contient notamment :                               │
# │      - pt      : position (x, y)                                        │
# │      - size    : taille/échelle                                         │
# │      - angle   : orientation                                            │
# │      - response: score d'importance                                     │
# │                                                                         │
# │   des_g / des_d = tableau NumPy de shape (N, 128)                       │
# │      N = nombre de keypoints                                            │
# │      128 = taille du descripteur SIFT                                   │
# │                                                                         │
# │ PRINCIPE :                                                              │
# │   Deux points représentant le même détail physique dans les deux vues   │
# │   devraient avoir des descripteurs similaires.                          │
# └─────────────────────────────────────────────────────────────────────────┘

if des_g is None or des_d is None:
    raise RuntimeError("Aucun descripteur SIFT détecté. Vérifie la qualité/texture des images.")

print("\n" + "=" * 70)
print("POINTS SIFT")
print("=" * 70)
print(f"Nb keypoints gauche : {len(kp_g)}")
print(f"Nb keypoints droite : {len(kp_d)}")


# =============================================================================
# ██████████████████████████████████████████████████████████████████████████
# ÉTAPE 4 : VISUALISATION DES KEYPOINTS
# ██████████████████████████████████████████████████████████████████████████
# =============================================================================

img_kp_g = cv2.drawKeypoints(
    img_g, kp_g, None,
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    color=(0, 255, 0)
)

img_kp_d = cv2.drawKeypoints(
    img_d, kp_d, None,
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    color=(0, 255, 0)
)
# ┌─────────────────────────────────────────────────────────────────────────┐
# │ cv2.drawKeypoints(image, keypoints, outImage, flags, color)             │
# │                                                                         │
# │ BUT : dessiner visuellement les points détectés.                        │
# │                                                                         │
# │ ENTRÉES :                                                               │
# │   image     = image source                                              │
# │   keypoints = liste des cv2.KeyPoint                                    │
# │   outImage  = None → OpenCV crée l'image de sortie                      │
# │   color     = (0,255,0) → vert en BGR                                   │
# │                                                                         │
# │ FLAGS :                                                                 │
# │   DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS                                │
# │   → dessine des cercles dont la taille reflète l'échelle du point       │
# │   → parfois aussi l'orientation                                          │
# │                                                                         │
# │ SORTIE :                                                                │
# │   image couleur avec keypoints dessinés                                 │
# └─────────────────────────────────────────────────────────────────────────┘

cv2.imwrite(os.path.join(OUT_DIR, "keypoints_gauche.png"), img_kp_g)
cv2.imwrite(os.path.join(OUT_DIR, "keypoints_droite.png"), img_kp_d)
# ┌─────────────────────────────────────────────────────────────────────────┐
# │ cv2.imwrite(chemin, image)                                              │
# │                                                                         │
# │ BUT : sauvegarder une image sur disque                                  │
# │                                                                         │
# │ ENTRÉES :                                                               │
# │   chemin = nom complet du fichier de sortie                             │
# │   image  = tableau NumPy représentant l'image                           │
# │                                                                         │
# │ Le format est déduit de l'extension : .png, .jpg, ...                   │
# └─────────────────────────────────────────────────────────────────────────┘


# =============================================================================
# ██████████████████████████████████████████████████████████████████████████
# ÉTAPE 5 : MATCHING DES DESCRIPTEURS AVEC FLANN
# ██████████████████████████████████████████████████████████████████████████
# =============================================================================

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=100)
# ┌─────────────────────────────────────────────────────────────────────────┐
# │ FLANN = Fast Library for Approximate Nearest Neighbors                  │
# │                                                                         │
# │ BUT : trouver rapidement, pour chaque descripteur de l'image gauche,    │
# │       les descripteurs les plus proches dans l'image droite.            │
# │                                                                         │
# │ Comme les descripteurs SIFT sont des vecteurs float (128D),             │
# │ on utilise un index KD-Tree.                                            │
# │                                                                         │
# │ index_params : paramètres de la structure de recherche                  │
# │   algorithm = FLANN_INDEX_KDTREE                                        │
# │      → on choisit l'algorithme KD-Tree                                  │
# │   trees = 5                                                             │
# │      → nombre d'arbres construits                                       │
# │      → plus grand = meilleure recherche mais plus lent                  │
# │                                                                         │
# │ search_params : paramètres de recherche                                 │
# │   checks = 100                                                          │
# │      → nombre de nœuds explorés pendant la recherche                    │
# │      → plus grand = plus précis mais plus lent                          │
# └─────────────────────────────────────────────────────────────────────────┘

flann = cv2.FlannBasedMatcher(index_params, search_params)
# ┌─────────────────────────────────────────────────────────────────────────┐
# │ cv2.FlannBasedMatcher(index_params, search_params)                      │
# │                                                                         │
# │ BUT : créer l'objet matcher FLANN                                       │
# │                                                                         │
# │ Cet objet permettra ensuite d'appeler knnMatch()                        │
# └─────────────────────────────────────────────────────────────────────────┘

matches_knn = flann.knnMatch(des_g, des_d, k=2)
# ┌─────────────────────────────────────────────────────────────────────────┐
# │ flann.knnMatch(queryDescriptors, trainDescriptors, k=2)                 │
# │                                                                         │
# │ BUT : pour chaque descripteur de l'image gauche (query),                │
# │       chercher les k meilleurs voisins dans l'image droite (train).     │
# │                                                                         │
# │ ENTRÉES :                                                               │
# │   des_g = descripteurs image gauche                                     │
# │   des_d = descripteurs image droite                                     │
# │   k=2   = on veut les 2 meilleurs voisins                               │
# │                                                                         │
# │ SORTIE :                                                                │
# │   liste de listes :                                                     │
# │     matches_knn[i] = [m, n]                                             │
# │       m = meilleur match                                                │
# │       n = deuxième meilleur match                                       │
# │                                                                         │
# │ Pourquoi demander 2 voisins ?                                           │
# │   Parce qu'on va appliquer le test de Lowe pour savoir                  │
# │   si le meilleur match est vraiment distinct du second.                 │
# └─────────────────────────────────────────────────────────────────────────┘


# =============================================================================
# ██████████████████████████████████████████████████████████████████████████
# ÉTAPE 6 : LOWE RATIO TEST
# ██████████████████████████████████████████████████████████████████████████
# =============================================================================

good_matches = []
pts_g = []
pts_d = []

ratio_thresh = 0.72
# ┌─────────────────────────────────────────────────────────────────────────┐
# │ PRINCIPLE DU LOWE RATIO TEST                                            │
# │                                                                         │
# │ Si le meilleur match m est beaucoup meilleur que le second n,           │
# │ alors l'appariement est probablement fiable.                            │
# │                                                                         │
# │ Condition :                                                             │
# │   m.distance < ratio_thresh * n.distance                                │
# │                                                                         │
# │ distance = distance entre deux descripteurs SIFT                        │
# │   → petite distance = descripteurs ressemblants                         │
# │                                                                         │
# │ Si m et n sont presque aussi proches :                                  │
# │   → le point est ambigu                                                 │
# │   → on le rejette                                                       │
# │                                                                         │
# │ ratio_thresh typique : 0.7 à 0.8                                        │
# └─────────────────────────────────────────────────────────────────────────┘

for pair in matches_knn:
    if len(pair) < 2:
        continue
    m, n = pair

    if m.distance < ratio_thresh * n.distance:
        good_matches.append(m)
        pts_g.append(kp_g[m.queryIdx].pt)
        pts_d.append(kp_d[m.trainIdx].pt)
# ┌─────────────────────────────────────────────────────────────────────────┐
# │ m.queryIdx = indice du point dans l'image gauche                        │
# │ m.trainIdx = indice du point correspondant dans l'image droite          │
# │                                                                         │
# │ kp_g[m.queryIdx].pt → coordonnées (x, y) du point gauche                │
# │ kp_d[m.trainIdx].pt → coordonnées (x, y) du point droite                │
# │                                                                         │
# │ On construit :                                                          │
# │   pts_g = liste des points 2D dans image gauche                         │
# │   pts_d = liste des points 2D correspondants dans image droite          │
# └─────────────────────────────────────────────────────────────────────────┘

pts_g = np.array(pts_g, dtype=np.float32)
pts_d = np.array(pts_d, dtype=np.float32)
# ┌─────────────────────────────────────────────────────────────────────────┐
# │ Conversion en tableaux NumPy float32                                    │
# │                                                                         │
# │ Pourquoi float32 ?                                                      │
# │   Beaucoup de fonctions OpenCV attendent ce type.                       │
# │                                                                         │
# │ Forme obtenue :                                                         │
# │   pts_g.shape = (N, 2)                                                  │
# │   pts_d.shape = (N, 2)                                                  │
# │ avec N = nombre de matches retenus                                      │
# └─────────────────────────────────────────────────────────────────────────┘

print("\n" + "=" * 70)
print("MATCHING APRÈS LOWE")
print("=" * 70)
print(f"Nb matches après ratio test : {len(good_matches)}")

if len(pts_g) < 8:
    raise RuntimeError("Pas assez de matches après Lowe ratio test pour estimer la matrice fondamentale.")


img_matches_lowe = cv2.drawMatches(
    img_g, kp_g, img_d, kp_d,
    good_matches, None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)
# ┌─────────────────────────────────────────────────────────────────────────┐
# │ cv2.drawMatches(img1, kp1, img2, kp2, matches, outImg, flags)           │
# │                                                                         │
# │ BUT : dessiner les correspondances entre deux images                    │
# │                                                                         │
# │ OpenCV place img1 à gauche et img2 à droite, puis trace                 │
# │ une ligne entre chaque paire de points correspondants.                  │
# │                                                                         │
# │ flags = NOT_DRAW_SINGLE_POINTS                                          │
# │   → n'affiche pas les points isolés                                     │
# │   → affiche uniquement les matches fournis                              │
# └─────────────────────────────────────────────────────────────────────────┘

cv2.imwrite(os.path.join(OUT_DIR, "matches_apres_lowe.png"), img_matches_lowe)


# =============================================================================
# ██████████████████████████████████████████████████████████████████████████
# ÉTAPE 7 : FILTRAGE GÉOMÉTRIQUE PAR MATRICE FONDAMENTALE + RANSAC
# ██████████████████████████████████████████████████████████████████████████
# =============================================================================

F, mask = cv2.findFundamentalMat(
    pts_g,
    pts_d,
    cv2.FM_RANSAC,
    1.5,
    0.99
)
# ┌─────────────────────────────────────────────────────────────────────────┐
# │ cv2.findFundamentalMat(points1, points2, method, ransacReprojThreshold, │
# │                         confidence)                                      │
# │                                                                         │
# │ BUT : estimer la matrice fondamentale F entre deux images               │
# │       et rejeter les faux matches avec RANSAC.                          │
# │                                                                         │
# │ PRINCIPE DE LA MATRICE FONDAMENTALE :                                   │
# │   Si x dans image 1 correspond à x' dans image 2, alors                 │
# │   ils doivent respecter une contrainte géométrique appelée              │
# │   contrainte épipolaire.                                                │
# │                                                                         │
# │   En gros : un point dans l'image 1 ne peut pas correspondre            │
# │   à n'importe où dans l'image 2. Il doit tomber sur une certaine ligne. │
# │                                                                         │
# │ RANSAC :                                                                │
# │   - choisit aléatoirement des sous-ensembles de points                  │
# │   - calcule un modèle F                                                 │
# │   - garde le modèle qui explique le plus de points cohérents            │
# │   - rejette les outliers (faux matches)                                 │
# │                                                                         │
# │ PARAMÈTRES :                                                            │
# │   pts_g, pts_d                                                          │
# │      → tableaux (N, 2) contenant les points correspondants              │
# │                                                                         │
# │   cv2.FM_RANSAC                                                         │
# │      → méthode robuste à utiliser                                       │
# │                                                                         │
# │   1.5                                                                   │
# │      → seuil de reprojection en pixels                                  │
# │      → plus petit = plus strict                                         │
# │      → plus grand = plus permissif                                      │
# │                                                                         │
# │   0.99                                                                  │
# │      → niveau de confiance du RANSAC                                    │
# │                                                                         │
# │ SORTIES :                                                               │
# │   F    = matrice fondamentale 3x3                                       │
# │   mask = vecteur (N,1) avec :                                           │
# │          1 = inlier, 0 = outlier                                        │
# └─────────────────────────────────────────────────────────────────────────┘

if F is None or mask is None:
    raise RuntimeError("Échec du calcul de la matrice fondamentale.")

mask = mask.ravel().astype(bool)
# ┌─────────────────────────────────────────────────────────────────────────┐
# │ mask.ravel()                                                            │
# │   → transforme mask en vecteur 1D                                       │
# │                                                                         │
# │ astype(bool)                                                            │
# │   → convertit 1/0 en True/False                                         │
# │   → pratique pour filtrer directement des tableaux NumPy                │
# └─────────────────────────────────────────────────────────────────────────┘

pts_g_ransac = pts_g[mask]
pts_d_ransac = pts_d[mask]
good_matches_ransac = [good_matches[i] for i in range(len(good_matches)) if mask[i]]

print("\n" + "=" * 70)
print("FILTRAGE RANSAC")
print("=" * 70)
print(f"Nb inliers RANSAC : {len(pts_g_ransac)} / {len(pts_g)}")

if len(pts_g_ransac) < 8:
    raise RuntimeError("Pas assez de points après RANSAC.")


# =============================================================================
# ██████████████████████████████████████████████████████████████████████████
# ÉTAPE 8 : FILTRAGE STÉRÉO FINAL
# ██████████████████████████████████████████████████████████████████████████
# =============================================================================

disparites = pts_g_ransac[:, 0] - pts_d_ransac[:, 0]
dy = np.abs(pts_g_ransac[:, 1] - pts_d_ransac[:, 1])
# ┌─────────────────────────────────────────────────────────────────────────┐
# │ disparité = x_gauche - x_droite                                         │
# │                                                                         │
# │ INTERPRÉTATION :                                                        │
# │   Si un objet est vu dans deux images stéréo, son point n'apparaît      │
# │   pas à la même abscisse x dans les deux vues.                          │
# │                                                                         │
# │   Cette différence s'appelle la disparité.                              │
# │                                                                         │
# │   En stéréovision classique :                                           │
# │     - grande disparité → objet proche                                   │
# │     - petite disparité → objet loin                                     │
# │                                                                         │
# │ dy = |y_gauche - y_droite|                                              │
# │   → différence verticale entre les deux points                          │
# │                                                                         │
# │ Pour deux vues stéréo presque horizontales, les points correspondants   │
# │ doivent avoir des ordonnées très proches, donc dy petit.                │
# └─────────────────────────────────────────────────────────────────────────┘

max_vertical_diff = 8.0
min_disp = 1.0
# ┌─────────────────────────────────────────────────────────────────────────┐
# │ max_vertical_diff = tolérance verticale max (en pixels)                 │
# │   → si dy est trop grand, le match est suspect                          │
# │                                                                         │
# │ min_disp = disparité minimale                                           │
# │   → évite des points quasi sans déplacement horizontal                  │
# │   → utile pour virer quelques matches douteux                           │
# └─────────────────────────────────────────────────────────────────────────┘

mask_stereo = (disparites > min_disp) & (dy < max_vertical_diff)
# ┌─────────────────────────────────────────────────────────────────────────┐
# │ On garde uniquement les points qui respectent EN MÊME TEMPS :           │
# │   1) disparité > min_disp                                               │
# │   2) dy < max_vertical_diff                                             │
# │                                                                         │
# │ Le résultat est un masque booléen True/False                            │
# └─────────────────────────────────────────────────────────────────────────┘

pts_g_final = pts_g_ransac[mask_stereo]
pts_d_final = pts_d_ransac[mask_stereo]

print("\n" + "=" * 70)
print("FILTRAGE STÉRÉO FINAL")
print("=" * 70)
print(f"Disparité min  : {disparites.min():.2f} px")
print(f"Disparité max  : {disparites.max():.2f} px")
print(f"|dy| moyen     : {dy.mean():.2f} px")
print(f"Nb points finaux : {len(pts_g_final)} / {len(pts_g_ransac)}")

if len(pts_g_final) == 0:
    raise RuntimeError("Tous les points ont été supprimés par le filtrage final. Relâche un peu les seuils.")


# =============================================================================
# ██████████████████████████████████████████████████████████████████████████
# ÉTAPE 9 : VISUALISATION DES MATCHES FINAUX
# ██████████████████████████████████████████████████████████████████████████
# =============================================================================

kp_g_final = [cv2.KeyPoint(float(p[0]), float(p[1]), 5) for p in pts_g_final]
kp_d_final = [cv2.KeyPoint(float(p[0]), float(p[1]), 5) for p in pts_d_final]
# ┌─────────────────────────────────────────────────────────────────────────┐
# │ cv2.KeyPoint(x, y, size)                                                │
# │                                                                         │
# │ BUT : créer manuellement un objet KeyPoint OpenCV                       │
# │                                                                         │
# │ Pourquoi ici ?                                                          │
# │   Après les filtrages, on n'a plus les objets KeyPoint d'origine        │
# │   alignés 1 à 1 avec les points finaux.                                 │
# │   Donc on recrée de "petits keypoints" artificiels juste pour           │
# │   l'affichage avec drawMatches.                                         │
# │                                                                         │
# │ PARAMÈTRES :                                                            │
# │   x, y = position du point                                              │
# │   size = diamètre visuel du point                                       │
# └─────────────────────────────────────────────────────────────────────────┘

matches_final_vis = [cv2.DMatch(i, i, 0) for i in range(len(pts_g_final))]
# ┌─────────────────────────────────────────────────────────────────────────┐
# │ cv2.DMatch(queryIdx, trainIdx, distance)                                │
# │                                                                         │
# │ BUT : représenter un match entre un point de gauche et un point         │
# │       de droite.                                                        │
# │                                                                         │
# │ ICI :                                                                   │
# │   on relie artificiellement le point i de gauche avec le point i        │
# │   de droite, car ce sont déjà les paires finales correspondantes.       │
# │                                                                         │
# │ distance = 0 ici, car c'est seulement pour l'affichage                  │
# └─────────────────────────────────────────────────────────────────────────┘

img_matches_final = cv2.drawMatches(
    img_g, kp_g_final,
    img_d, kp_d_final,
    matches_final_vis,
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

cv2.imwrite(os.path.join(OUT_DIR, "matches_finaux.png"), img_matches_final)


# =============================================================================
# ██████████████████████████████████████████████████████████████████████████
# ÉTAPE 10 : SAUVEGARDE DES POINTS
# ██████████████████████████████████████████████████████████████████████████
# =============================================================================

np.save(os.path.join(OUT_DIR, "pts_gauche.npy"), pts_g_final)
np.save(os.path.join(OUT_DIR, "pts_droite.npy"), pts_d_final)
# ┌─────────────────────────────────────────────────────────────────────────┐
# │ np.save(fichier.npy, tableau)                                           │
# │                                                                         │
# │ BUT : sauvegarder un tableau NumPy au format binaire .npy               │
# │                                                                         │
# │ AVANTAGE :                                                              │
# │   - rapide à relire                                                     │
# │   - conserve exactement le type et la forme                             │
# │                                                                         │
# │ ICI :                                                                   │
# │   pts_gauche.npy : shape (N, 2), coordonnées (u, v) dans image gauche   │
# │   pts_droite.npy : shape (N, 2), coordonnées (u, v) dans image droite   │
# │                                                                         │
# │ Ces fichiers seront utilisés dans qst4.py pour le calcul 3D.            │
# └─────────────────────────────────────────────────────────────────────────┘

np.savetxt(os.path.join(OUT_DIR, "pts_gauche.txt"), pts_g_final, fmt="%.4f")
np.savetxt(os.path.join(OUT_DIR, "pts_droite.txt"), pts_d_final, fmt="%.4f")
# ┌─────────────────────────────────────────────────────────────────────────┐
# │ np.savetxt(...)                                                         │
# │   → sauvegarde aussi en texte lisible                                   │
# │   → utile pour vérifier à la main les coordonnées                       │
# └─────────────────────────────────────────────────────────────────────────┘

print("\n" + "=" * 70)
print("SAUVEGARDE")
print("=" * 70)
print(f"Dossier de sortie : {OUT_DIR}")
print(f"pts_gauche.npy : {pts_g_final.shape}")
print(f"pts_droite.npy : {pts_d_final.shape}")
print("Images sauvegardées :")
print("  - keypoints_gauche.png")
print("  - keypoints_droite.png")
print("  - matches_apres_lowe.png")
print("  - matches_finaux.png")


# =============================================================================
# ██████████████████████████████████████████████████████████████████████████
# ÉTAPE 11 : AFFICHAGE FINAL AVEC MATPLOTLIB
# ██████████████████████████████████████████████████████████████████████████
# =============================================================================

plt.figure(figsize=(15, 7))
plt.imshow(cv2.cvtColor(img_matches_final, cv2.COLOR_BGR2RGB))
# ┌─────────────────────────────────────────────────────────────────────────┐
# │ matplotlib attend généralement des images en RGB,                       │
# │ alors qu'OpenCV stocke en BGR.                                          │
# │                                                                         │
# │ Donc avant l'affichage avec plt.imshow, on reconvertit BGR -> RGB.      │
# └─────────────────────────────────────────────────────────────────────────┘

plt.title("Matches SIFT finaux après Lowe + RANSAC + filtrage stéréo")
plt.axis("off")
plt.tight_layout()
plt.show()
# ┌─────────────────────────────────────────────────────────────────────────┐
# │ plt.figure(figsize=(15,7))                                              │
# │   → crée une figure de taille 15x7 pouces                               │
# │                                                                         │
# │ plt.imshow(...)                                                         │
# │   → affiche l'image                                                     │
# │                                                                         │
# │ plt.axis("off")                                                         │
# │   → enlève les axes                                                     │
# │                                                                         │
# │ plt.tight_layout()                                                      │
# │   → ajuste les marges automatiquement                                   │
# │                                                                         │
# │ plt.show()                                                              │
# │   → ouvre la fenêtre / affiche la figure                                │
# └─────────────────────────────────────────────────────────────────────────┘