# 🔬 Melanoma Detection - Test Scripts

Documentation complète pour tester votre application de détection de mélanome.

## 📋 Fichiers de Test Disponibles

### 1. `test_image_prediction.py` - Script en Ligne de Commande
**Meilleur pour**: Tester une seule image avec contrôle complet

```bash
# Syntaxe
python test_image_prediction.py <image_path> [model_name]

# Exemples
python test_image_prediction.py ./melanoma.jpg EfficientNetV2B2
python test_image_prediction.py ./test/image.png ResNet50_ML_ENSEMBLE
python test_image_prediction.py image.jpg  # Utilise le modèle par défaut

# Lister les modèles disponibles
python test_image_prediction.py --list
```

**Avantages**:
- ✅ Interface simple en ligne de commande
- ✅ Sortie formatée et claire
- ✅ Support de tous les modèles
- ✅ Affichage des probabilités en pourcentage
- ✅ Indicateur de confiance

---

### 2. `test_interactive.py` - Interface Interactive
**Meilleur pour**: Tests répétés et exploration

```bash
# Lancer
python test_interactive.py

# Le script vous guidera:
# 1. Choisir un modèle
# 2. Entrer le chemin de l'image
# 3. Voir les résultats
# 4. Tester une autre image
```

**Avantages**:
- ✅ Interface guidée et conviviale
- ✅ Sélection facile du modèle
- ✅ Boucle de test continue
- ✅ Validation des entrées
- ✅ Affichage colorisé des résultats

---

### 3. `test_batch.py` - Test par Lot (Batch)
**Meilleur pour**: Tester plusieurs images à la fois

```bash
# Syntaxe
python test_batch.py <image_dir_or_file> [model_name]

# Exemples
python test_batch.py ./test_images/ EfficientNetV2B2
python test_batch.py ./melanoma_set/
python test_batch.py image.jpg ResNet50

# Lister les modèles
python test_batch.py --list
```

**Avantages**:
- ✅ Traitement de plusieurs images
- ✅ Tableau récapitulatif automatique
- ✅ Statistiques globales
- ✅ Sortie formatée et lisible

---

## 🎯 Modèles Disponibles

### Modèles Deep Learning Seuls
- `ResNet50`
- `MobileNetV2`
- `EfficientNetV2B0`
- `EfficientNetV2B2` ⭐ (par défaut)
- `EfficientNetV2B3`
- `EfficientNetV2M`
- `DenseNet121`
- `DenseNet169`
- `VGG16`
- `VGG19`
- `CNN`
- `ConvNeXtSmall`

### Modèles Hybrides (DL + ML)
- `EfficientNetV2B2_ML`
- `ResNet50_ML`
- `DenseNet169_ML`

### Modèles Ensemble
- `EfficientNetV2B2_DenseNet169_ENSEMBLE`
- `ResNet50_EfficientNetV2B2_ENSEMBLE`
- `ResNet50_DenseNet169_ENSEMBLE`
- `ResNet50_EfficientNetV2B2_DenseNet169_ENSEMBLE`

### Modèles Ensemble Hybrides
- `EfficientNetV2B2_ML_ENSEMBLE`
- `EfficientNetV2B2_DenseNet169_ML_ENSEMBLE`
- `ResNet50_ML_ENSEMBLE`
- `ResNet50_EfficientNetV2B2_ML_ENSEMBLE`
- `ResNet50_EfficientNetV2B2_DenseNet169_ML_ENSEMBLE`

---

## 📊 Format de Sortie

### Exemple de Résultat
```
════════════════════════════════════════════════════════════════════════════════
  🔬 MELANOMA DETECTION RESULTS
════════════════════════════════════════════════════════════════════════════════

  📁 Image:        ./melanoma.jpg
  🤖 Model:        EfficientNetV2B2

────────────────────────────────────────────────────────────────────────────────

  🎯 DIAGNOSIS: MALIGNANT

  📊 PROBABILITIES:
     • Benign:     35.42%
     • Malignant:  64.58%

  ⚙️  Threshold:    50.00%

  💪 Confidence:   29.2%

════════════════════════════════════════════════════════════════════════════════
```

### Signification des Résultats
- **Diagnosis**: Classification finale (Benign ou Malignant)
- **Probabilities**: 
  - Benign: Probabilité que l'image soit bénigne
  - Malignant: Probabilité que l'image soit maligne
- **Threshold**: Seuil utilisé pour la classification (50% par défaut)
- **Confidence**: Niveau de confiance du modèle (0-100%)

---

## 🖼️ Format d'Image Acceptés

Formats supportés:
- ✅ JPG / JPEG
- ✅ PNG
- ✅ BMP
- ✅ GIF
- ✅ TIFF

Taille recommandée: Toute taille (redimensionnée automatiquement à 224×224)

---

## 📝 Exemples Complets

### Exemple 1: Test simple d'une image
```bash
python test_image_prediction.py ./melanoma.jpg
```

### Exemple 2: Test avec un modèle spécifique
```bash
python test_image_prediction.py ./test.png EfficientNetV2B2_ML_ENSEMBLE
```

### Exemple 3: Tests interactifs multiples
```bash
python test_interactive.py
# Choisir modèle → Entrer chemin image → Voir résultats → Recommencer
```

### Exemple 4: Tester tout un dossier
```bash
python test_batch.py ./dataset/test_images/ ResNet50
```

### Exemple 5: Afficher les modèles disponibles
```bash
python test_image_prediction.py --list
# ou
python test_batch.py --list
```

---

## 🔧 Dépendances

Les scripts utilisent les bibliothèques suivantes (déjà installées):
- `tensorflow` - Pour les modèles de deep learning
- `PIL/Pillow` - Pour le chargement et prétraitement des images
- `numpy` - Pour les calculs numériques
- `scikit-learn` - Pour les modèles ML hybrides
- `tabulate` - Pour l'affichage des tableaux (test_batch.py)

---

## ⚠️ Réglage du Seuil

Le seuil par défaut est **0.5** (50%), ce qui signifie:
- Probabilité Malignant < 50% → **Benign**
- Probabilité Malignant ≥ 50% → **Malignant**

Pour modifier le seuil, éditez la variable `THRESHOLD` en haut du script:

```python
THRESHOLD = 0.5  # Modifier cette valeur (ex: 0.6 pour 60%)
```

---

## 🚨 Dépannage

### Erreur: "Image not found"
→ Vérifiez le chemin de l'image
→ Utilisez le chemin absolu si nécessaire

### Erreur: "Model not found"
→ Utilisez `--list` pour voir les modèles disponibles
→ Vérifiez l'orthographe du nom du modèle

### Erreur: "Unsupported image format"
→ Convertissez l'image en JPG, PNG ou BMP
→ Vérifiez l'extension du fichier

### Problème de mémoire
→ Testez une image à la fois (au lieu d'un dossier complet)
→ Utilisez un modèle plus léger (ex: MobileNetV2)

---

## 📞 Usage Tipps

1. **Pour la production**: Utilisez `test_image_prediction.py` avec le modèle optimisé
2. **Pour le développement**: Utilisez `test_interactive.py` 
3. **Pour les tests complets**: Utilisez `test_batch.py` sur un dossier
4. **Pour voir les options**: Ajoutez `--help` à n'importe quel script

```bash
python test_image_prediction.py --help
python test_interactive.py --help
python test_batch.py --help
```

---

## 📌 Notes Importantes

- ⚠️ Le code existant n'a pas été modifié
- ✅ Les scripts de test sont indépendants
- ✅ Tous les scripts utilisent les modèles existants dans `/Models/`
- ✅ Les probabilités sont calculées automatiquement
- ✅ Aucun fichier n'est écrit sur le disque (sauf pour les résultats optionnels)

---

**Créé**: 2026-06-06
**Version**: 1.0
**Status**: ✅ Prêt à utiliser
