use crate::models::linear_model::PricePredictor;
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use tracing::info;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Signal {
    Buy,
    Sell,
    Hold,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MLStrategy {
    pub predictor: PricePredictor,
    pub threshold: f64,
    pub lookback_period: usize,
}

impl MLStrategy {
    pub fn new(threshold: f64, lookback_period: usize) -> Self {
        Self {
            predictor: PricePredictor::new(),
            threshold,
            lookback_period,
        }
    }

    pub fn train(&mut self, prices: &Array1<f64>) -> Result<(), String> {
        let (features, targets) = self.prepare_data(prices)?;
        self.predictor.train(&features, &targets)?;
        
        let r2 = self.predictor.evaluate(&features, &targets)?;
        info!("Model trained with R² score: {:.4}", r2);
        
        Ok(())
    }

    pub fn generate_signal(&self, recent_prices: &Array1<f64>) -> Result<Signal, String> {
        if recent_prices.len() < self.lookback_period {
            return Err("Not enough price data".to_string());
        }

        let features = self.extract_features(recent_prices);
        let prediction = self.predictor.predict(&features)?;
        
        let current_price = recent_prices[recent_prices.len() - 1];
        let predicted_price = prediction[0];
        
        let price_change = (predicted_price - current_price) / current_price;

        let signal = if price_change > self.threshold {
            Signal::Buy
        } else if price_change < -self.threshold {
            Signal::Sell
        } else {
            Signal::Hold
        };

        info!(
            "Current: {:.2}, Predicted: {:.2}, Change: {:.2}%, Signal: {:?}",
            current_price,
            predicted_price,
            price_change * 100.0,
            signal
        );

        Ok(signal)
    }

    fn prepare_data(&self, prices: &Array1<f64>) -> Result<(Array2<f64>, Array1<f64>), String> {
        let n = prices.len();
        if n < self.lookback_period + 1 {
            return Err("Not enough data for training".to_string());
        }

        let num_samples = n - self.lookback_period;
        let mut features = Array2::zeros((num_samples, self.lookback_period));
        let mut targets = Array1::zeros(num_samples);

        for i in 0..num_samples {
            for j in 0..self.lookback_period {
                features[[i, j]] = prices[i + j];
            }
            targets[i] = prices[i + self.lookback_period];
        }

        Ok((features, targets))
    }

    fn extract_features(&self, prices: &Array1<f64>) -> Array2<f64> {
        let start = prices.len() - self.lookback_period;
        let window = prices.slice(ndarray::s![start..]);
        
        let mut features = Array2::zeros((1, self.lookback_period));
        for (i, &price) in window.iter().enumerate() {
            features[[0, i]] = price;
        }
        
        features
    }

    pub fn backtest(&self, prices: &Array1<f64>, initial_capital: f64) -> BacktestResult {
        let mut capital = initial_capital;
        let mut position = 0.0;
        let mut trades = Vec::new();

        for i in self.lookback_period..prices.len() {
            let window = prices.slice(ndarray::s![i-self.lookback_period..i]);
            let signal = match self.generate_signal(&window.to_owned()) {
                Ok(s) => s,
                Err(_) => Signal::Hold,
            };

            let current_price = prices[i];

            match signal {
                Signal::Buy if position == 0.0 => {
                    position = capital / current_price;
                    capital = 0.0;
                    trades.push((i, "BUY".to_string(), current_price));
                }
                Signal::Sell if position > 0.0 => {
                    capital = position * current_price;
                    position = 0.0;
                    trades.push((i, "SELL".to_string(), current_price));
                }
                _ => {}
            }
        }

        // Close position if still open
        if position > 0.0 {
            capital = position * prices[prices.len() - 1];
            position = 0.0;
        }

        let final_value = capital;
        let total_return = (final_value - initial_capital) / initial_capital * 100.0;

        BacktestResult {
            initial_capital,
            final_value,
            total_return,
            num_trades: trades.len(),
            trades,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestResult {
    pub initial_capital: f64,
    pub final_value: f64,
    pub total_return: f64,
    pub num_trades: usize,
    pub trades: Vec<(usize, String, f64)>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ml_strategy_train() {
        let prices = Array1::from_vec((0..100).map(|i| 100.0 + i as f64).collect());
        let mut strategy = MLStrategy::new(0.01, 10);
        
        let result = strategy.train(&prices);
        assert!(result.is_ok());
    }

    #[test]
    fn test_generate_signal() {
        let prices = Array1::from_vec((0..100).map(|i| 100.0 + i as f64).collect());
        let mut strategy = MLStrategy::new(0.01, 10);
        strategy.train(&prices).unwrap();

        let recent = Array1::from_vec((90..100).map(|i| 100.0 + i as f64).collect());
        let signal = strategy.generate_signal(&recent);
        assert!(signal.is_ok());
    }
}
