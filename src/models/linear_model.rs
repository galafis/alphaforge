use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PricePredictor {
    pub coefficients: Vec<f64>,
    pub intercept: f64,
    pub feature_count: usize,
}

impl PricePredictor {
    pub fn new() -> Self {
        Self {
            coefficients: Vec::new(),
            intercept: 0.0,
            feature_count: 0,
        }
    }

    pub fn train(&mut self, features: &Array2<f64>, targets: &Array1<f64>) -> Result<(), String> {
        if features.nrows() != targets.len() {
            return Err("Features and targets must have the same length".to_string());
        }

        // Simple linear regression using normal equation: θ = (X^T X)^-1 X^T y
        // For simplicity, we'll use gradient descent instead
        let n_samples = features.nrows();
        let n_features = features.ncols();
        
        self.feature_count = n_features;
        self.coefficients = vec![0.0; n_features];
        self.intercept = 0.0;
        
        // Gradient descent
        let learning_rate = 0.01;
        let iterations = 1000;
        
        for _ in 0..iterations {
            let predictions = self.predict_internal(features);
            let errors: Array1<f64> = &predictions - targets;
            
            // Update coefficients
            for j in 0..n_features {
                let gradient: f64 = (0..n_samples)
                    .map(|i| errors[i] * features[[i, j]])
                    .sum::<f64>() / n_samples as f64;
                self.coefficients[j] -= learning_rate * gradient;
            }
            
            // Update intercept
            let intercept_gradient: f64 = errors.sum() / n_samples as f64;
            self.intercept -= learning_rate * intercept_gradient;
        }

        Ok(())
    }
    
    fn predict_internal(&self, features: &Array2<f64>) -> Array1<f64> {
        let coeffs = Array1::from_vec(self.coefficients.clone());
        features.dot(&coeffs) + self.intercept
    }

    pub fn predict(&self, features: &Array2<f64>) -> Result<Array1<f64>, String> {
        if features.ncols() != self.feature_count {
            return Err(format!(
                "Expected {} features, got {}",
                self.feature_count,
                features.ncols()
            ));
        }

        let coeffs = Array1::from_vec(self.coefficients.clone());
        let predictions = features.dot(&coeffs) + self.intercept;

        Ok(predictions)
    }

    pub fn evaluate(&self, features: &Array2<f64>, targets: &Array1<f64>) -> Result<f64, String> {
        let predictions = self.predict(features)?;
        
        // Calculate R² score
        let mean = targets.mean().unwrap_or(0.0);
        let ss_tot: f64 = targets.iter().map(|&y| (y - mean).powi(2)).sum();
        let ss_res: f64 = targets
            .iter()
            .zip(predictions.iter())
            .map(|(&y, &pred)| (y - pred).powi(2))
            .sum();

        let r2 = 1.0 - (ss_res / ss_tot);
        Ok(r2)
    }

    pub fn save(&self, path: &str) -> anyhow::Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    pub fn load(path: &str) -> anyhow::Result<Self> {
        let json = std::fs::read_to_string(path)?;
        let model = serde_json::from_str(&json)?;
        Ok(model)
    }
}

impl Default for PricePredictor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn test_train_and_predict() {
        let features = arr2(&[
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
        ]);
        let targets = Array1::from_vec(vec![3.0, 5.0, 7.0, 9.0]);

        let mut predictor = PricePredictor::new();
        let result = predictor.train(&features, &targets);
        assert!(result.is_ok());

        let test_features = arr2(&[[5.0, 6.0]]);
        let predictions = predictor.predict(&test_features);
        assert!(predictions.is_ok());
    }

    #[test]
    fn test_evaluate() {
        let features = arr2(&[
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
        ]);
        let targets = Array1::from_vec(vec![3.0, 5.0, 7.0, 9.0]);

        let mut predictor = PricePredictor::new();
        predictor.train(&features, &targets).unwrap();

        let r2 = predictor.evaluate(&features, &targets).unwrap();
        assert!(r2 > 0.8); // Good fit
    }
}
