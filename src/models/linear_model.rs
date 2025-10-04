use linfa::prelude::*;
use linfa_linear::LinearRegression;
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

        let dataset = Dataset::new(features.clone(), targets.clone());
        
        let model = LinearRegression::default()
            .fit(&dataset)
            .map_err(|e| format!("Training failed: {:?}", e))?;

        self.coefficients = model.params().to_vec();
        self.intercept = model.intercept();
        self.feature_count = features.ncols();

        Ok(())
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

    pub fn save(&self, path: &str) -> Result<(), String> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| format!("Serialization failed: {}", e))?;
        
        std::fs::write(path, json)
            .map_err(|e| format!("Failed to write file: {}", e))?;

        Ok(())
    }

    pub fn load(path: &str) -> Result<Self, String> {
        let json = std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read file: {}", e))?;
        
        let model = serde_json::from_str(&json)
            .map_err(|e| format!("Deserialization failed: {}", e))?;

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
