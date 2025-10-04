use alphaforge::MLStrategy;
use clap::{Parser, Subcommand};
use ndarray::Array1;
use tracing::{info, Level};
use tracing_subscriber;

#[derive(Parser)]
#[command(name = "AlphaForge")]
#[command(about = "Machine Learning Trading Bot", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Train a new model
    Train {
        /// Number of data points to generate
        #[arg(short, long, default_value = "1000")]
        samples: usize,
    },
    /// Run backtest
    Backtest {
        /// Initial capital
        #[arg(short, long, default_value = "10000.0")]
        capital: f64,
    },
    /// Generate trading signal
    Signal {
        /// Recent prices (comma-separated)
        #[arg(short, long)]
        prices: String,
    },
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(Level::INFO)
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Train { samples } => {
            run_training(samples)?;
        }
        Commands::Backtest { capital } => {
            run_backtest(capital)?;
        }
        Commands::Signal { prices } => {
            run_signal_generation(&prices)?;
        }
    }

    Ok(())
}

fn run_training(samples: usize) -> anyhow::Result<()> {
    info!("Training model with {} samples", samples);

    // Generate synthetic price data (trending upward with noise and volatility)
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let prices: Array1<f64> = Array1::from_vec(
        (0..samples)
            .map(|i| {
                let trend = 100.0 + i as f64 * 0.1;
                let seasonal = (i as f64 * 0.1).sin() * 5.0;
                let noise = rng.gen_range(-2.0..2.0);
                trend + seasonal + noise
            })
            .collect(),
    );

    let mut strategy = MLStrategy::new(0.02, 20);
    strategy.train(&prices)?;

    info!("Model trained successfully");
    std::fs::create_dir_all("data/trained_models")?;
    strategy.predictor.save("data/trained_models/model.json")?;
    info!("Model saved to data/trained_models/model.json");

    Ok(())
}

fn run_backtest(initial_capital: f64) -> anyhow::Result<()> {
    info!("Running backtest with initial capital: ${}", initial_capital);

    // Generate synthetic price data
    let prices: Array1<f64> = Array1::from_vec(
        (0..500)
            .map(|i| 100.0 + i as f64 * 0.1 + (i as f64 * 0.1).sin() * 5.0)
            .collect(),
    );

    let mut strategy = MLStrategy::new(0.02, 20);
    strategy.train(&prices)?;

    let result = strategy.backtest(&prices, initial_capital);

    info!("Backtest Results:");
    info!("  Initial Capital: ${:.2}", result.initial_capital);
    info!("  Final Value: ${:.2}", result.final_value);
    info!("  Total Return: {:.2}%", result.total_return);
    info!("  Number of Trades: {}", result.num_trades);

    Ok(())
}

fn run_signal_generation(prices_str: &str) -> anyhow::Result<()> {
    let prices: Vec<f64> = prices_str
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();

    if prices.len() < 20 {
        return Err(anyhow::anyhow!("Need at least 20 price points"));
    }

    let prices_array = Array1::from_vec(prices.clone());

    // Train on historical data
    let mut strategy = MLStrategy::new(0.02, 20);
    strategy.train(&prices_array)?;

    // Generate signal for recent prices
    let recent = Array1::from_vec(prices[prices.len() - 20..].to_vec());
    let signal = strategy.generate_signal(&recent)?;

    info!("Generated Signal: {:?}", signal);

    Ok(())
}
