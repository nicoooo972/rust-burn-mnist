# Burn MNIST - Classification de chiffres manuscrits

Projet de classification d'images MNIST utilisant le framework [Burn](https://burn.dev/) en Rust.
[burn github examples](https://github.com/tracel-ai/burn/tree/main/examples)



## Description

Ce projet implémente un réseau de neurones convolutif (CNN) pour la classification des chiffres manuscrits du dataset MNIST (0-9). Il utilise Burn 0.19.1.

## Architecture du modèle

Le modèle est composé de :

- **2 blocs convolutifs** avec :
  
  - Convolution 2D (kernels 3x3)
  - Batch Normalization
  - Activation ReLU
  - Max Pooling 2x2
- **3 couches fully connected** :
  
  - FC1: 1600 → 128
  - FC2: 128 → 128
  - FC3: 128 → 10 (classes)
- **Régularisation** :
  
  - Dropout (25%)
  - Activation GELU

## Fonctionnalités

- **Augmentation de données** : Translation, Shear, Scale, Rotation
- **Early stopping** : Arrêt automatique si pas d'amélioration pendant 5 epochs
- **Métriques** : Accuracy, Loss, Learning Rate
- **Sauvegarde** : Modèle et configuration sauvegardés dans `/tmp/burn-example-mnist`
- **Scheduler d'apprentissage** : Warmup linéaire + décroissance cosinusoïdale

## Installation

### Prérequis

- Rust (édition 2021)
- Cargo

### Dépendances

Les dépendances sont définies dans `Cargo.toml` :

- `burn` (0.19.1) avec features : train, vision, dataset
- `burn-ndarray` (0.19.1)
- `burn-autodiff` (0.19.1)
- `rand` (0.9.2)
- `log` (0.4)
- `env_logger` (0.11)

## Structure du projet

```
burn/
├── src/
│   ├── main.rs          # Point d'entrée principal
│   ├── lib.rs           # Module principal avec tests
│   ├── model.rs         # Architecture du CNN
│   ├── training.rs      # Configuration et boucle d'entraînement
│   ├── data.rs          # Chargement et transformation des données MNIST
│   └── learn/
│       └── test-basic.rs # Exemple de test basique
├── Cargo.toml
└── README.md
```

## Utilisation

### Entraînement

```bash
cargo run --release
```

L'entraînement utilise :

- 55 000 images pour l'entraînement
- 5 000 images pour la validation
- 10 000 images pour les tests
- Batch size : 256
- Nombre d'epochs : 20
- Optimiseur : AdamW avec weight decay

### Test basique

```bash
cargo run --example test-basic
```

Vérifie que Burn est correctement installé en effectuant des opérations tensorielles simples.

### Tests unitaires

```bash
cargo test
```

Tests disponibles :

- Création du modèle
- Opérations tensorielles
- Forward pass du modèle

## Configuration

La configuration d'entraînement peut être modifiée dans `training.rs` :

```rust
pub struct MnistTrainingConfig {
    pub num_epochs: usize,      // Par défaut : 20
    pub batch_size: usize,       // Par défaut : 256
    pub num_workers: usize,      // Par défaut : 8
    pub seed: u64,               // Par défaut : 42
    pub optimizer: AdamWConfig,
}
```

## Résultats

Les métriques sont affichées pendant l'entraînement et sauvegardées dans `/tmp/burn-example-mnist`. Le modèle entraîné est sauvegardé à la fin de l'entraînement.

## Backend

Le projet utilise :

- **Backend** : `NdArray` avec `Autodiff` pour le calcul automatique des gradients
- **Device** : CPU par défaut (peut être modifié pour GPU si disponible)

