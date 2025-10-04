pub mod data;
pub mod features;
pub mod models;
pub mod strategies;

pub use models::linear_model::PricePredictor;
pub use strategies::ml_strategy::{MLStrategy, Signal, BacktestResult};
