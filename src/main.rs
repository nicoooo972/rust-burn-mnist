use burn_autodiff::Autodiff;
use burn_ndarray::NdArray;

use burn_mnist::training;

fn main() {
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .init();
    
    println!("🔥 Démarrage de l'entraînement MNIST avec Burn 0.19\n");
    
    type Backend = Autodiff<NdArray>;
    let device = Default::default();
    
    training::run::<Backend>(device);
}