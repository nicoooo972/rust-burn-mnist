use std::sync::Arc;

use crate::{
    data::{MnistBatcher, MnistItemPrepared, MnistMapper, Transform},
    model::Model,
};

use log::{info, debug};

use burn::{
    data::{
        dataloader::DataLoaderBuilder,
        dataset::{
            Dataset,
            transform::{ComposedDataset, MapperDataset, PartialDataset, SamplerDataset},
            vision::{MnistDataset, MnistItem},
        },
    },
    lr_scheduler::{
        composed::ComposedLrSchedulerConfig, cosine::CosineAnnealingLrSchedulerConfig,
        linear::LinearLrSchedulerConfig,
    },
    prelude::*,
    record::{CompactRecorder, NoStdTrainingRecorder},
    tensor::backend::AutodiffBackend,
    train::{
        EvaluatorBuilder, LearnerBuilder, MetricEarlyStoppingStrategy, StoppingCondition,
        metric::{
            AccuracyMetric, LearningRateMetric, LossMetric,
            store::{Aggregate, Direction, Split},
        },
        renderer::MetricsRenderer,
    },
};
use burn::{optim::AdamWConfig};

static ARTIFACT_DIR: &str = "/tmp/burn-example-mnist";

#[derive(Config, Debug)]
pub struct MnistTrainingConfig {
    #[config(default = 20)]
    pub num_epochs: usize,

    #[config(default = 256)]
    pub batch_size: usize,

    #[config(default = 8)]
    pub num_workers: usize,

    #[config(default = 42)]
    pub seed: u64,

    pub optimizer: AdamWConfig,
}

fn create_artifact_dir(artifact_dir: &str) {
    // Remove existing artifacts before to get an accurate learner summary
    info!("Nettoyage du répertoire d'artifacts: {}", artifact_dir);
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
    info!("Répertoire d'artifacts créé: {}", artifact_dir);
}

pub fn run<B: AutodiffBackend>(device: B::Device) {
    info!("=== Démarrage de l'entraînement ===");
    create_artifact_dir(ARTIFACT_DIR);
    
    // Config
    info!("Configuration de l'optimiseur AdamW");
    let config_optimizer = AdamWConfig::new()
        .with_cautious_weight_decay(true)
        .with_weight_decay(5e-5);

    let config = MnistTrainingConfig::new(config_optimizer);
    info!("Configuration: {} epochs, batch_size={}, seed={}", 
          config.num_epochs, config.batch_size, config.seed);
    
    B::seed(&device, config.seed);
    debug!("Seed initialisé: {}", config.seed);

    info!("Initialisation du modèle");
    let model = Model::<B>::new(&device);

    info!("Chargement des datasets MNIST");
    let dataset_train_original = Arc::new(MnistDataset::train());
    let dataset_train_plain = PartialDataset::new(dataset_train_original.clone(), 0, 55_000);
    let dataset_valid_plain = PartialDataset::new(dataset_train_original.clone(), 55_000, 60_000);
    info!("Dataset train: {} échantillons, validation: {} échantillons", 
          dataset_train_plain.len(), dataset_valid_plain.len());

    debug!("Génération des identifiants de transformation pour train");
    let ident_trains = generate_idents(Some(10000));
    debug!("Génération des identifiants de transformation pour validation");
    let ident_valid = generate_idents(None);
    info!("{} transformations appliquées au dataset d'entraînement", ident_trains.len());
    
    let dataset_train = DatasetIdent::compose(ident_trains, dataset_train_plain);
    let dataset_valid = DatasetIdent::compose(ident_valid, dataset_valid_plain);

    info!("Création des dataloaders (batch_size={}, workers={})", 
          config.batch_size, config.num_workers);
    let dataloader_train = DataLoaderBuilder::new(MnistBatcher::default())
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(dataset_train);
    let dataloader_valid = DataLoaderBuilder::new(MnistBatcher::default())
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(dataset_valid);
    
    info!("Configuration du scheduler de learning rate (cosine + warmup + decay)");
    let lr_scheduler = ComposedLrSchedulerConfig::new()
        .cosine(CosineAnnealingLrSchedulerConfig::new(1.0, 2000))
        // Warmup
        .linear(LinearLrSchedulerConfig::new(1e-8, 1.0, 2000))
        .linear(LinearLrSchedulerConfig::new(1e-2, 1e-6, 10000));

    info!("Construction du learner avec early stopping (5 epochs sans amélioration)");
    let learner = LearnerBuilder::new(ARTIFACT_DIR)
        .metrics((AccuracyMetric::new(), LossMetric::new()))
        .metric_train_numeric(LearningRateMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .early_stopping(MetricEarlyStoppingStrategy::new(
            &LossMetric::<B>::new(),
            Aggregate::Mean,
            Direction::Lowest,
            Split::Valid,
            StoppingCondition::NoImprovementSince { n_epochs: 5 },
        ))
        .num_epochs(config.num_epochs)
        .summary()
        .build(
            model,
            config.optimizer.init(),
            lr_scheduler.init().unwrap(),
            //LearningStrategy::SingleDevice(device),
        );

    info!("=== Début de l'entraînement ===");
    let result = learner.fit(dataloader_train, dataloader_valid);
    info!("=== Entraînement terminé ===");

    info!("Chargement du dataset de test");
    let dataset_test_plain = Arc::new(MnistDataset::test());
    info!("Dataset de test: {} échantillons", dataset_test_plain.len());
    
    let mut renderer = result.renderer;

    let idents_tests = generate_idents(None);
    info!("Évaluation sur {} configurations de transformation", idents_tests.len());

    for (ident, _) in idents_tests {
        let name = ident.to_string();
        info!("Évaluation avec transformation: {}", name);
        renderer = evaluate::<B::InnerBackend>(
            name.as_str(),
            ident,
            result.model.clone(),
            renderer,
            dataset_test_plain.clone(),
            config.batch_size,
        );
    }

    info!("Sauvegarde du modèle dans {}", ARTIFACT_DIR);
    result
        .model
        .save_file(
            format!("{ARTIFACT_DIR}/model"),
            &NoStdTrainingRecorder::new(),
        )
        .expect("Failed to save trained model");
    info!("Modèle sauvegardé avec succès");

    info!("Sauvegarde de la configuration");
    config
        .save(format!("{ARTIFACT_DIR}/config.json").as_str())
        .unwrap();
    info!("Configuration sauvegardée");

    renderer.manual_close();
    info!("=== Fin du processus d'entraînement ===");
}

fn evaluate<B: Backend>(
    name: &str,
    ident: DatasetIdent,
    model: Model<B>,
    renderer: Box<dyn MetricsRenderer>,
    dataset: impl Dataset<MnistItem> + 'static,
    batch_size: usize,
) -> Box<dyn MetricsRenderer> {
    debug!("Préparation du dataset pour l'évaluation: {}", name);
    let batcher = MnistBatcher::default();
    let dataset_test = DatasetIdent::prepare(ident, dataset);
    let dataset_len = dataset_test.len();
    let dataloader_test = DataLoaderBuilder::new(batcher)
        .batch_size(batch_size)
        .num_workers(2)
        .build(dataset_test);
    debug!("Dataloader créé avec {} échantillons", dataset_len);

    let evaluator = EvaluatorBuilder::new(ARTIFACT_DIR)
        .renderer(renderer)
        .metrics((AccuracyMetric::new(), LossMetric::new()))
        .build(model);

    debug!("Démarrage de l'évaluation: {}", name);
    let renderer = evaluator.eval(name, dataloader_test);
    info!("Évaluation terminée: {}", name);
    renderer
}

enum DatasetIdent {
    Plain,
    Transformed(Vec<Transform>),
    All,
}

impl core::fmt::Display for DatasetIdent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DatasetIdent::Plain => f.write_str("Plain")?,
            DatasetIdent::Transformed(items) => {
                for i in items {
                    f.write_fmt(format_args!("{i}"))?;
                    f.write_str(" ")?;
                }
            }
            DatasetIdent::All => f.write_str("All")?,
        };

        Ok(())
    }
}

impl DatasetIdent {
    pub fn many(transforms: Vec<Transform>) -> Self {
        Self::Transformed(transforms)
    }

    pub fn prepare(self, dataset: impl Dataset<MnistItem>) -> impl Dataset<MnistItemPrepared> {
        let items = match self {
            DatasetIdent::Plain => Vec::new(),
            DatasetIdent::All => {
                vec![
                    Transform::Translate,
                    Transform::Shear,
                    Transform::Scale,
                    Transform::Rotation,
                ]
            }
            DatasetIdent::Transformed(items) => items.clone(),
        };
        MapperDataset::new(dataset, MnistMapper::default().transform(&items))
    }

    pub fn compose(
        idents: Vec<(Self, Option<usize>)>,
        dataset: PartialDataset<Arc<MnistDataset>, MnistItem>,
    ) -> impl Dataset<MnistItemPrepared> {
        let datasets = idents
            .into_iter()
            .map(|(ident, size)| match size {
                Some(size) => {
                    SamplerDataset::with_replacement(ident.prepare(dataset.clone()), size)
                }
                None => {
                    let dataset = ident.prepare(dataset.clone());
                    let size = dataset.len();
                    SamplerDataset::without_replacement(dataset, size)
                }
            })
            .collect();
        ComposedDataset::new(datasets)
    }
}

fn generate_idents(num_samples_base: Option<usize>) -> Vec<(DatasetIdent, Option<usize>)> {
    let mut current = Vec::new();
    let mut idents = Vec::new();

    for shear in [None, Some(Transform::Shear)] {
        for scale in [None, Some(Transform::Scale)] {
            for rotation in [None, Some(Transform::Rotation)] {
                for translate in [None, Some(Transform::Translate)] {
                    if let Some(tr) = shear {
                        current.push(tr);
                    }
                    if let Some(tr) = scale {
                        current.push(tr);
                    }
                    if let Some(tr) = rotation {
                        current.push(tr);
                    }
                    if let Some(tr) = translate {
                        current.push(tr);
                    }

                    let num_samples = num_samples_base.map(|val| val * current.len());

                    if current.len() == 4 {
                        idents.push((DatasetIdent::All, num_samples));
                    } else if current.is_empty() {
                        idents.push((DatasetIdent::Plain, num_samples));
                    } else {
                        idents.push((DatasetIdent::many(current.clone()), num_samples));
                    }

                    current.clear();
                }
            }
        }
    }

    idents
}