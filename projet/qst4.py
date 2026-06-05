#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
=============================================================================
QST4 - RECONSTRUCTION 3D À PARTIR DES POINTS SIFT APPARIÉS
=============================================================================

PLAN GLOBAL DU TRAVAIL
----------------------

Dans qst3.py, on a déjà fait une partie très importante du pipeline :
1) détecter des points SIFT dans les deux images,
2) apparier les points,
3) supprimer les faux appariements,
4) sauvegarder les bonnes paires de points 2D.

Maintenant, dans qst4.py, le but est d'aller plus loin :
on veut passer de points 2D correspondants dans deux images
à des points 3D dans l’espace.

AUTREMENT DIT :
- entrée  : deux ensembles de points 2D appariés + la calibration K
- sortie  : un nuage de points 3D

IDÉE GÉNÉRALE
-------------

Quand un même point physique de la scène est visible dans les deux images,
on connaît sa projection :
- dans l'image 1 : (u1, v1)
- dans l'image 2 : (u2, v2)

Si on connaît aussi :
- la géométrie entre les deux caméras,
- leurs paramètres intrinsèques,

alors on peut "remonter" des images vers l’espace 3D.
Cette opération s'appelle la triangulation.

LOGIQUE DU SCRIPT
-----------------

Le script suit exactement cet ordre :

1) Charger les points 2D appariés
2) Charger la matrice intrinsèque K
3) Estimer la matrice fondamentale F
4) Convertir F en matrice essentielle E
5) Extraire la pose relative entre les deux vues : rotation R, translation t
6) Corriger l’échelle de t
7) Construire les matrices de projection P1 et P2
8) Trianguler les points 3D
9) Nettoyer les points aberrants
10) Sauvegarder et visualiser le nuage de points

IMPORTANT SUR L’ÉCHELLE
-----------------------

Quand on utilise la matrice essentielle et recoverPose,
la translation t obtenue n’est pas une vraie distance métrique.
Elle donne seulement :
- la direction du déplacement,
- mais pas sa longueur réelle.

Donc si on veut un nuage de points plus lisible ou plus cohérent,
on peut imposer nous-mêmes une baseline choisie :
par exemple 10 cm entre les deux positions de la caméra.

Dans ce script :
- on garde la direction de t,
- on normalise t,
- puis on lui donne une longueur choisie : BASELINE_REAL.

=============================================================================
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# =============================================================================
# 1) CHEMINS DES DOSSIERS
# =============================================================================

BASE_DIR   = r"C:\Users\SALOUA\Desktop\TPS\Tp Vision\projet\TP3CV_2026\TP3CV_2026\projet"
SIFT_DIR   = os.path.join(BASE_DIR, "resultats_sift")
PARAMS_DIR = os.path.join(BASE_DIR, "..", "camera_params")
OUTPUT_DIR = os.path.join(BASE_DIR, "resultats_3D")

# ---------------------------------------------------------------------------
# EXPLICATION
# ---------------------------------------------------------------------------
# BASE_DIR :
#   dossier principal du projet
#
# SIFT_DIR :
#   dossier qui contient les résultats de qst3.py
#   notamment :
#   - pts_gauche.npy
#   - pts_droite.npy
#
# PARAMS_DIR :
#   dossier contenant les paramètres de calibration,
#   en particulier la matrice intrinsèque K sauvegardée dans mtx.npy
#
# OUTPUT_DIR :
#   dossier dans lequel on sauvegardera les résultats 3D
#
# os.path.join(...)
#   sert à construire les chemins correctement,
#   sans écrire les "\" à la main.
#
# C’est plus propre et plus portable.
# ---------------------------------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# os.makedirs(dossier, exist_ok=True)
#
# BUT :
#   créer le dossier s’il n’existe pas.
#
# POURQUOI ?
#   Parce qu’on va sauvegarder plusieurs fichiers :
#   - points_3D.npy
#   - points_3D.txt
#   - nuage_points.ply
#   - images PNG
#
# exist_ok=True :
#   évite une erreur si le dossier existe déjà.
# ---------------------------------------------------------------------------


# =============================================================================
# 2) CHOIX DE LA BASELINE
# =============================================================================

BASELINE_REAL = 10.0

# ---------------------------------------------------------------------------
# EXPLICATION FONDAMENTALE
# ---------------------------------------------------------------------------
# recoverPose donne une translation t "à l’échelle près".
#
# Cela signifie :
#   la direction de t est fiable,
#   mais sa longueur n’est pas la vraie distance physique.
#
# Donc si on ne corrige pas cela,
# le nuage 3D sera reconstruit dans une échelle arbitraire.
#
# Pour mieux interpréter les résultats,
# on choisit une baseline :
#
#   BASELINE_REAL = 10.0
#
# Ici, cela veut dire :
#   "je décide que la distance entre les deux positions
#    de la caméra est de 10 unités"
#
# Si tu travailles en centimètres :
#   10.0 = 10 cm
#
# Si tu travailles en mètres :
#   0.10 = 10 cm
#
# Ce choix agit sur :
#   - la taille globale du nuage de points,
#   - les coordonnées reconstruites,
#   - la lisibilité de P2
#
# IMPORTANT :
#   cela ne change pas la forme relative du nuage,
#   seulement son échelle.
# ---------------------------------------------------------------------------


# =============================================================================
# 3) CHARGEMENT DES DONNÉES
# =============================================================================

pts_gauche = np.load(os.path.join(SIFT_DIR, "pts_gauche.npy"))
pts_droite = np.load(os.path.join(SIFT_DIR, "pts_droite.npy"))
K = np.load(os.path.join(PARAMS_DIR, "mtx.npy"))

# ---------------------------------------------------------------------------
# np.load(...)
#
# BUT :
#   relire un tableau NumPy sauvegardé avec np.save(...)
#
# ICI :
#   pts_gauche.npy  -> points 2D de l’image gauche
#   pts_droite.npy  -> points 2D correspondants dans l’image droite
#   mtx.npy         -> matrice intrinsèque K
#
# FORMAT ATTENDU :
#   pts_gauche : tableau (N, 2)
#   pts_droite : tableau (N, 2)
#
# où chaque ligne contient :
#   [x, y]
#
# K :
#   matrice 3x3 contenant les paramètres intrinsèques de la caméra,
#   par exemple :
#   - focale fx
#   - focale fy
#   - centre principal cx
#   - centre principal cy
#
# Cette matrice vient de la calibration de la caméra.
# Elle est indispensable pour passer de la géométrie image
# à la géométrie caméra.
# ---------------------------------------------------------------------------

if pts_gauche.shape != pts_droite.shape:
    raise ValueError("pts_gauche et pts_droite n'ont pas la même taille.")

# ---------------------------------------------------------------------------
# Pourquoi vérifier la taille ?
#
# Parce que chaque point de gauche doit correspondre exactement
# à un point de droite.
#
# Donc si pts_gauche et pts_droite n'ont pas la même taille,
# cela veut dire qu'il y a une incohérence dans les données.
# ---------------------------------------------------------------------------

if len(pts_gauche) < 8:
    raise RuntimeError("Pas assez de points pour estimer F/E et trianguler.")

# ---------------------------------------------------------------------------
# Pourquoi au moins 8 points ?
#
# Parce que l’estimation de la matrice fondamentale nécessite
# un nombre minimal de correspondances.
#
# En pratique :
#   plus on a de points fiables, mieux c’est.
# ---------------------------------------------------------------------------

print("=" * 70)
print("CHARGEMENT DES DONNÉES")
print("=" * 70)
print(f"Points gauche : {pts_gauche.shape}")
print(f"Points droite : {pts_droite.shape}")
print("Matrice intrinsèque K :")
print(K)


# =============================================================================
# 4) ESTIMATION DE LA MATRICE FONDAMENTALE F
# =============================================================================

F, maskF = cv2.findFundamentalMat(
    pts_gauche,
    pts_droite,
    cv2.FM_RANSAC,
    1.0,
    0.99
)

# ---------------------------------------------------------------------------
# cv2.findFundamentalMat(points1, points2, method, threshold, confidence)
#
# RÔLE :
#   calculer la matrice fondamentale F.
#
# IDÉE GÉOMÉTRIQUE :
#   La matrice fondamentale relie les points correspondants
#   dans deux images.
#
# Si x est un point dans l’image 1
# et x' son correspondant dans l’image 2,
# alors ils satisfont la contrainte :
#
#   x'^T F x = 0
#
# Cela veut dire que :
#   le point de la 2ème image doit se trouver sur
#   la droite épipolaire associée au point de la 1ère image.
#
# POURQUOI COMMENCER PAR F ?
#   Parce que F encode la relation géométrique
#   entre les deux images directement à partir des points 2D.
#
# PARAMÈTRES :
#
# 1) pts_gauche
#    ensemble des points 2D de l’image 1
#
# 2) pts_droite
#    ensemble des points 2D de l’image 2
#
# 3) cv2.FM_RANSAC
#    on demande à OpenCV d’utiliser RANSAC
#
#    RANSAC sert à :
#    - tester des sous-ensembles de points,
#    - estimer plusieurs modèles candidats,
#    - garder le modèle qui explique le plus de points
#      de manière cohérente,
#    - rejeter les outliers (fausses correspondances)
#
#    C’est très important car même après qst3,
#    il peut rester quelques mauvais matches.
#
# 4) 1.0
#    seuil RANSAC en pixels
#
#    Interprétation :
#    si un point est trop loin de la contrainte épipolaire,
#    on le considère comme faux.
#
#    Plus petit :
#      filtrage plus sévère
#    Plus grand :
#      filtrage plus permissif
#
# 5) 0.99
#    niveau de confiance
#
# SORTIES :
#
# F :
#   matrice fondamentale 3x3
#
# maskF :
#   masque indiquant pour chaque point s’il est :
#   - inlier  (bon)
#   - outlier (mauvais)
#
# INTÉRÊT DE CETTE ÉTAPE :
#   nettoyer encore les correspondances avant la reconstruction 3D.
# ---------------------------------------------------------------------------

if F is None or maskF is None:
    raise RuntimeError("Impossible de calculer la matrice fondamentale.")

maskF = maskF.ravel().astype(bool)

# ---------------------------------------------------------------------------
# maskF est souvent retourné comme un tableau colonne de forme (N,1).
#
# ravel() :
#   transforme ce tableau en vecteur 1D de taille N
#
# astype(bool) :
#   convertit les 0/1 en False/True
#
# Cela permet ensuite de filtrer directement les tableaux NumPy.
# ---------------------------------------------------------------------------

pts_g = pts_gauche[maskF]
pts_d = pts_droite[maskF]

# ---------------------------------------------------------------------------
# On ne garde que les inliers validés par RANSAC.
#
# À partir d’ici :
#   pts_g et pts_d sont des correspondances plus fiables.
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("MATRICE FONDAMENTALE")
print("=" * 70)
print("F =")
print(F)
print(f"Points inliers pour F : {len(pts_g)} / {len(pts_gauche)}")

if len(pts_g) < 8:
    raise RuntimeError("Pas assez de points après estimation de F.")


# =============================================================================
# 5) CALCUL DE LA MATRICE ESSENTIELLE E
# =============================================================================

E = K.T @ F @ K

# ---------------------------------------------------------------------------
# IDÉE :
#   La matrice fondamentale F travaille avec des points exprimés
#   dans les coordonnées image (pixels).
#
#   La matrice essentielle E travaille avec des coordonnées
#   normalisées par la caméra, donc dans le repère caméra.
#
# RELATION :
#   E = K^T F K
#
# POURQUOI C’EST IMPORTANT ?
#   Parce que E contient l’information géométrique
#   liée à la pose relative entre les deux caméras :
#   - rotation R
#   - translation t
#
# AUTREMENT DIT :
#   F = géométrie entre images
#   E = géométrie entre caméras calibrées
#
# K.T :
#   transpose de K
#
# L’opérateur @ :
#   multiplication matricielle en Python / NumPy
#
# Donc :
#   K.T @ F @ K
# signifie :
#   K transposée × F × K
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("MATRICE ESSENTIELLE")
print("=" * 70)
print("E =")
print(E)


# =============================================================================
# 6) ESTIMATION DE LA POSE RELATIVE : R ET t
# =============================================================================

retval, R, t, mask_pose = cv2.recoverPose(E, pts_g, pts_d, K)

# ---------------------------------------------------------------------------
# cv2.recoverPose(E, points1, points2, K)
#
# RÔLE :
#   retrouver la pose relative entre les deux caméras :
#   - R : rotation
#   - t : translation
#
# Cela veut dire :
#   "comment la caméra 2 est placée par rapport à la caméra 1 ?"
#
# PARAMÈTRES :
#
# 1) E
#    matrice essentielle calculée juste avant
#
# 2) pts_g
#    points de la première vue
#
# 3) pts_d
#    points correspondants de la deuxième vue
#
# 4) K
#    matrice intrinsèque
#
# SORTIES :
#
# retval :
#   nombre de points retenus / cohérents
#
# R :
#   matrice de rotation 3x3
#
# t :
#   vecteur de translation 3x1
#
# mask_pose :
#   masque indiquant quels points sont conservés
#
# POINT IMPORTANT :
#   à partir d’une matrice essentielle,
#   il existe plusieurs solutions possibles pour R et t.
#
# recoverPose choisit la solution correcte
# en appliquant un test appelé :
#   cheirality check
#
# IDÉE DU CHEIRALITY CHECK :
#   les points 3D reconstruits doivent se trouver
#   devant les deux caméras.
#
# Si une solution place beaucoup de points derrière une caméra,
# elle est rejetée.
#
# POURQUOI CETTE ÉTAPE EST ESSENTIELLE ?
#   Parce qu’on ne peut trianguler correctement les points
#   que si on connaît la position relative des deux vues.
# ---------------------------------------------------------------------------

if retval is None or retval <= 0:
    raise RuntimeError("Échec de recoverPose.")

mask_pose = mask_pose.ravel() > 0

pts_g_pose = pts_g[mask_pose]
pts_d_pose = pts_d[mask_pose]

print("\n" + "=" * 70)
print("POSE RELATIVE BRUTE")
print("=" * 70)
print(f"Nb points retenus par recoverPose : {len(pts_g_pose)}")
print("R =")
print(R)
print("t brut =")
print(t)

if len(pts_g_pose) < 8:
    raise RuntimeError("Pas assez de points après recoverPose.")


# =============================================================================
# 7) CORRECTION DE L'ÉCHELLE DE LA TRANSLATION
# =============================================================================

norm_t = np.linalg.norm(t)

# ---------------------------------------------------------------------------
# np.linalg.norm(t)
#
# RÔLE :
#   calculer la norme euclidienne du vecteur t
#
# Mathématiquement :
#   ||t|| = sqrt(tx² + ty² + tz²)
#
# POURQUOI ?
#   On veut d’abord isoler la direction de t,
#   puis lui imposer une longueur choisie.
# ---------------------------------------------------------------------------

if norm_t < 1e-12:
    raise RuntimeError("Norme de t trop petite, impossible de normaliser.")

t_unit = t / norm_t

# ---------------------------------------------------------------------------
# t_unit :
#   vecteur unitaire
#
# Cela signifie :
#   même direction que t,
#   mais longueur égale à 1
#
# Pourquoi faire ça ?
#   Parce que recoverPose donne surtout une direction utile.
# ---------------------------------------------------------------------------

t_scaled = t_unit * BASELINE_REAL

# ---------------------------------------------------------------------------
# t_scaled :
#   translation corrigée
#
# Interprétation :
#   on prend la direction estimée par recoverPose,
#   puis on impose une longueur égale à BASELINE_REAL.
#
# C’est ce vecteur-là qu’on utilisera ensuite
# pour construire la deuxième matrice de projection P2.
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("TRANSLATION CORRIGÉE")
print("=" * 70)
print(f"Norme de t brut : {norm_t:.6f}")
print(f"Baseline choisie : {BASELINE_REAL:.4f}")
print("t unitaire =")
print(t_unit)
print("t corrigé =")
print(t_scaled)


# =============================================================================
# 8) CONSTRUCTION DES MATRICES DE PROJECTION
# =============================================================================

P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
P2 = K @ np.hstack((R, t_scaled))

# ---------------------------------------------------------------------------
# MATRICE DE PROJECTION
# ---------------------
# Une matrice de projection sert à projeter un point 3D
# sur le plan image.
#
# Sa forme générale est :
#   P = K [R | t]
#
# où :
#   K = intrinsèques
#   R = rotation
#   t = translation
#
# PREMIÈRE CAMÉRA
# ---------------
# On choisit comme convention que la première caméra
# définit le repère de référence.
#
# Donc :
#   R = I
#   t = 0
#
# ce qui donne :
#   P1 = K [I | 0]
#
# DEUXIÈME CAMÉRA
# ---------------
# Elle est décrite par :
#   P2 = K [R | t_scaled]
#
# np.eye(3)
#   crée la matrice identité 3x3
#
# np.zeros((3,1))
#   crée un vecteur colonne nul
#
# np.hstack((A, B))
#   colle A et B horizontalement
#
# Exemple :
#   A = matrice 3x3
#   B = vecteur 3x1
#   résultat = matrice 3x4
#
# POURQUOI CETTE ÉTAPE ?
#   Parce que la triangulation a besoin des deux matrices
#   qui décrivent les deux vues.
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("MATRICES DE PROJECTION")
print("=" * 70)
print("P1 =")
print(P1)
print("P2 =")
print(P2)


# =============================================================================
# 9) TRIANGULATION DES POINTS 3D
# =============================================================================

pts1 = pts_g_pose.T
pts2 = pts_d_pose.T

# ---------------------------------------------------------------------------
# Pourquoi la transposition ?
#
# pts_g_pose et pts_d_pose sont de forme :
#   (N, 2)
#
# c’est-à-dire :
#   N lignes, chaque ligne = [x, y]
#
# Mais cv2.triangulatePoints attend :
#   (2, N)
#
# Donc on transpose avec .T
#
# Résultat :
#   pts1 devient :
#   ligne 0 = tous les x
#   ligne 1 = tous les y
# ---------------------------------------------------------------------------

points_4D = cv2.triangulatePoints(P1, P2, pts1, pts2)

# ---------------------------------------------------------------------------
# cv2.triangulatePoints(P1, P2, pts1, pts2)
#
# RÔLE :
#   reconstruire les points 3D à partir :
#   - des deux matrices de projection,
#   - des deux ensembles de points 2D correspondants
#
# PRINCIPE :
#   Chaque point 2D dans chaque image définit un rayon visuel.
#   La triangulation cherche l’intersection des rayons
#   venant des deux caméras.
#
# En pratique, à cause du bruit,
# les rayons ne se coupent pas parfaitement.
# L’algorithme cherche donc la meilleure solution.
#
# SORTIE :
#   points_4D : tableau de forme (4, N)
#
# Chaque colonne contient :
#   [X, Y, Z, W]
#
# Ce sont des coordonnées homogènes.
# ---------------------------------------------------------------------------

points_4D /= points_4D[3]

# ---------------------------------------------------------------------------
# PASSAGE HOMOGÈNE -> EUCLIDIEN
#
# Les coordonnées homogènes [X, Y, Z, W]
# ne sont pas directement les coordonnées finales.
#
# On doit diviser chaque coordonnée par W :
#   X = X / W
#   Y = Y / W
#   Z = Z / W
#
# En Python :
#   points_4D /= points_4D[3]
#
# Cela divise chaque ligne par la 4ème ligne.
# ---------------------------------------------------------------------------

X = points_4D[0]
Y = points_4D[1]
Z = points_4D[2]

points_3D = np.column_stack((X, Y, Z)).astype(np.float32)

# ---------------------------------------------------------------------------
# np.column_stack((X, Y, Z))
#
# RÔLE :
#   reconstruire un tableau de forme (N, 3)
#
# où chaque ligne est :
#   [X, Y, Z]
#
# astype(np.float32)
#   convertit en flottants 32 bits
#
# POURQUOI ?
#   format plus compact,
#   pratique pour sauvegarde / visualisation
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("POINTS 3D")
print("=" * 70)
print(f"Nombre de points 3D : {len(points_3D)}")
print(f"X : min={X.min():.4f}  max={X.max():.4f}")
print(f"Y : min={Y.min():.4f}  max={Y.max():.4f}")
print(f"Z : min={Z.min():.4f}  max={Z.max():.4f}")


# =============================================================================
# 10) FILTRAGE OPTIONNEL DES OUTLIERS 3D
# =============================================================================

z_min = np.percentile(Z, 2)
z_max = np.percentile(Z, 98)

# ---------------------------------------------------------------------------
# np.percentile(Z, p)
#
# RÔLE :
#   calculer une valeur seuil statistique
#
# Ici :
#   percentile 2  -> borne basse
#   percentile 98 -> borne haute
#
# Pourquoi ?
#   Certains points triangulés peuvent être aberrants :
#   - trop loin,
#   - trop proches,
#   - incohérents
#
# En gardant seulement les profondeurs comprises entre ces bornes,
# on enlève les extrêmes.
# ---------------------------------------------------------------------------

mask_3d = (Z >= z_min) & (Z <= z_max)

points_3D_clean = points_3D[mask_3d]

Xc = points_3D_clean[:, 0]
Yc = points_3D_clean[:, 1]
Zc = points_3D_clean[:, 2]

print("\n" + "=" * 70)
print("FILTRAGE 3D")
print("=" * 70)
print(f"Points avant filtrage 3D : {len(points_3D)}")
print(f"Points après filtrage 3D : {len(points_3D_clean)}")


# =============================================================================
# 11) SAUVEGARDE DES POINTS 3D
# =============================================================================

np.save(os.path.join(OUTPUT_DIR, "points_3D.npy"), points_3D_clean)

# ---------------------------------------------------------------------------
# np.save(...)
#
# sauvegarde le tableau NumPy dans un fichier binaire .npy
#
# avantage :
#   - rapide
#   - précis
#   - facile à recharger
# ---------------------------------------------------------------------------

np.savetxt(
    os.path.join(OUTPUT_DIR, "points_3D.txt"),
    points_3D_clean,
    fmt="%.6f",
    delimiter="\t",
    header="X\tY\tZ"
)

# ---------------------------------------------------------------------------
# np.savetxt(...)
#
# sauvegarde une version texte lisible du nuage
#
# fmt="%.6f"
#   6 chiffres après la virgule
#
# delimiter="\t"
#   séparation par tabulation
#
# header="X\tY\tZ"
#   première ligne descriptive
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("SAUVEGARDE")
print("=" * 70)
print(f"points_3D.npy et points_3D.txt sauvegardés dans : {OUTPUT_DIR}")


# =============================================================================
# 12) EXPORT DU NUAGE DE POINTS AU FORMAT PLY
# =============================================================================

ply_path = os.path.join(OUTPUT_DIR, "nuage_points.ply")

with open(ply_path, "w", encoding="utf-8") as f:
    f.write("ply\n")
    f.write("format ascii 1.0\n")
    f.write(f"element vertex {len(points_3D_clean)}\n")
    f.write("property float x\n")
    f.write("property float y\n")
    f.write("property float z\n")
    f.write("end_header\n")
    for p in points_3D_clean:
        f.write(f"{p[0]} {p[1]} {p[2]}\n")

# ---------------------------------------------------------------------------
# Pourquoi exporter en PLY ?
#
# Parce que le format PLY est très utilisé
# pour les nuages de points 3D.
#
# Il peut être ouvert dans :
#   - MeshLab
#   - CloudCompare
#   - Open3D
#   - d’autres logiciels 3D
#
# Structure du fichier :
#   un en-tête
#   puis la liste des sommets
#
# Ici, chaque point 3D devient une ligne :
#   X Y Z
# ---------------------------------------------------------------------------

print(f"Fichier PLY sauvegardé : {ply_path}")


# =============================================================================
# 13) VISUALISATION 3D DU NUAGE
# =============================================================================

X_vis = Xc - np.mean(Xc)
Y_vis = Yc - np.mean(Yc)
Z_vis = Zc - np.mean(Zc)

# ---------------------------------------------------------------------------
# Pourquoi recentrer ?
#
# Pour l’affichage seulement.
#
# Si les coordonnées sont très décalées,
# le nuage peut apparaître mal centré dans la figure.
#
# Donc on soustrait la moyenne :
#   nouveau_X = X - moyenne(X)
#
# Cela ne change pas la forme du nuage,
# seulement sa position dans la fenêtre d’affichage.
# ---------------------------------------------------------------------------

z_norm = (Z_vis - Z_vis.min()) / (Z_vis.max() - Z_vis.min() + 1e-9)

# ---------------------------------------------------------------------------
# z_norm :
#   profondeur normalisée entre 0 et 1
#
# Pourquoi ?
#   pour colorer les points selon leur profondeur.
#
# 1e-9 :
#   petit terme ajouté pour éviter une division par zéro
#   si jamais Zmax = Zmin
# ---------------------------------------------------------------------------

fig = plt.figure(figsize=(11, 8))
ax = fig.add_subplot(111, projection='3d')
fig.suptitle("Nuage de points 3D reconstruit", fontsize=16, fontweight='bold')

sc = ax.scatter(
    X_vis, Z_vis, -Y_vis,
    c=z_norm,
    cmap='plasma',
    s=16,
    alpha=0.9,
    edgecolors='none'
)

# ---------------------------------------------------------------------------
# ax.scatter(...)
#
# affiche un nuage de points 3D
#
# PARAMÈTRES :
#
# X_vis, Z_vis, -Y_vis
#   coordonnées affichées
#
# Pourquoi cet ordre ?
#   C’est un choix d’affichage pour rendre la vue
#   plus intuitive.
#
# c=z_norm
#   couleur des points dépend de la profondeur
#
# cmap='plasma'
#   palette de couleurs
#
# s=16
#   taille des points
#
# alpha=0.9
#   transparence légère
#
# edgecolors='none'
#   pas de contour autour des points
# ---------------------------------------------------------------------------

ax.set_xlabel("X")
ax.set_ylabel("Z")
ax.set_zlabel("Y")

cbar = fig.colorbar(sc, ax=ax, shrink=0.70, pad=0.08)
cbar.set_label("Profondeur normalisée")

ax.view_init(elev=20, azim=-35)

# ---------------------------------------------------------------------------
# view_init(elev, azim)
#
# elev :
#   angle vertical de vue
#
# azim :
#   angle horizontal
#
# Cela change juste la façon dont on regarde le nuage.
# ---------------------------------------------------------------------------

plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "nuage_3D.png"),
    dpi=160,
    bbox_inches='tight'
)

print("Image du nuage 3D sauvegardée : nuage_3D.png")


# =============================================================================
# 14) VUES ORTHOGONALES
# =============================================================================

fig2, axes = plt.subplots(1, 3, figsize=(16, 5))
fig2.suptitle("Vues orthogonales du nuage 3D", fontsize=14, fontweight='bold')

axes[0].scatter(X_vis, -Y_vis, c=z_norm, cmap='plasma', s=10)
axes[0].set_title("Vue X-Y")
axes[0].set_xlabel("X")
axes[0].set_ylabel("Y")
axes[0].grid(True, alpha=0.3)

axes[1].scatter(X_vis, Z_vis, c=z_norm, cmap='plasma', s=10)
axes[1].set_title("Vue X-Z")
axes[1].set_xlabel("X")
axes[1].set_ylabel("Z")
axes[1].grid(True, alpha=0.3)

axes[2].scatter(Z_vis, -Y_vis, c=z_norm, cmap='plasma', s=10)
axes[2].set_title("Vue Z-Y")
axes[2].set_xlabel("Z")
axes[2].set_ylabel("Y")
axes[2].grid(True, alpha=0.3)

# ---------------------------------------------------------------------------
# Pourquoi ces vues 2D ?
#
# Le nuage 3D n’est pas toujours facile à lire.
#
# Les projections orthogonales permettent de voir :
#   - la répartition selon X et Y
#   - la profondeur Z
#   - la structure générale du nuage
#
# C’est utile pour détecter :
#   - les points aberrants,
#   - l’étalement du nuage,
#   - la forme globale de la scène.
# ---------------------------------------------------------------------------

plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "vues_orthogonales.png"),
    dpi=160,
    bbox_inches='tight'
)

print("Image des vues orthogonales sauvegardée : vues_orthogonales.png")


# =============================================================================
# 15) HISTOGRAMME DE LA PROFONDEUR
# =============================================================================

fig3, ax3 = plt.subplots(figsize=(8, 4))
ax3.hist(Zc, bins=30, color='steelblue', edgecolor='white', alpha=0.9)
ax3.set_title("Distribution de la profondeur Z")
ax3.set_xlabel("Z")
ax3.set_ylabel("Nombre de points")
ax3.grid(True, alpha=0.3)

# ---------------------------------------------------------------------------
# ax3.hist(...)
#
# RÔLE :
#   afficher la distribution de Z
#
# Pourquoi c’est intéressant ?
#   Parce que Z représente la profondeur.
#
# Cet histogramme permet de voir :
#   - où se concentrent les points,
#   - si la plupart des points sont proches ou loin,
#   - s’il reste des valeurs extrêmes.
# ---------------------------------------------------------------------------

plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "histogramme_Z.png"),
    dpi=160,
    bbox_inches='tight'
)

print("Histogramme Z sauvegardé : histogramme_Z.png")


print("\n" + "=" * 70)
print("RÉSUMÉ FINAL")
print("=" * 70)
print(f"Points 3D reconstruits : {len(points_3D_clean)}")
print(f"Baseline utilisée      : {BASELINE_REAL}")
print(f"Fichier du nuage       : {ply_path}")
print("=" * 70)

plt.show()