use ndarray::{Array1, ArrayView1};

/// Calculate Simple Moving Average
pub fn sma(prices: &ArrayView1<f64>, period: usize) -> Array1<f64> {
    let len = prices.len();
    let mut result = Array1::zeros(len);
    
    for i in period..=len {
        let window = prices.slice(ndarray::s![i-period..i]);
        result[i-1] = window.mean().unwrap_or(0.0);
    }
    
    result
}

/// Calculate Exponential Moving Average
pub fn ema(prices: &ArrayView1<f64>, period: usize) -> Array1<f64> {
    let len = prices.len();
    let mut result = Array1::zeros(len);
    let multiplier = 2.0 / (period as f64 + 1.0);
    
    // Initialize with SMA
    let initial_sma: f64 = prices.slice(ndarray::s![0..period]).mean().unwrap_or(0.0);
    result[period-1] = initial_sma;
    
    for i in period..len {
        result[i] = (prices[i] - result[i-1]) * multiplier + result[i-1];
    }
    
    result
}

/// Calculate Relative Strength Index
pub fn rsi(prices: &ArrayView1<f64>, period: usize) -> Array1<f64> {
    let len = prices.len();
    let mut result = Array1::from_elem(len, 50.0);
    
    if len < period + 1 {
        return result;
    }
    
    let mut gains = Vec::new();
    let mut losses = Vec::new();
    
    for i in 1..len {
        let change = prices[i] - prices[i-1];
        if change > 0.0 {
            gains.push(change);
            losses.push(0.0);
        } else {
            gains.push(0.0);
            losses.push(change.abs());
        }
    }
    
    for i in period..gains.len() {
        let avg_gain: f64 = gains[i-period..i].iter().sum::<f64>() / period as f64;
        let avg_loss: f64 = losses[i-period..i].iter().sum::<f64>() / period as f64;
        
        if avg_loss == 0.0 {
            result[i+1] = 100.0;
        } else {
            let rs = avg_gain / avg_loss;
            result[i+1] = 100.0 - (100.0 / (1.0 + rs));
        }
    }
    
    result
}

/// Calculate MACD (Moving Average Convergence Divergence)
pub fn macd(prices: &ArrayView1<f64>, fast: usize, slow: usize, signal: usize) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
    let ema_fast = ema(prices, fast);
    let ema_slow = ema(prices, slow);
    
    let macd_line = &ema_fast - &ema_slow;
    let signal_line = ema(&macd_line.view(), signal);
    let histogram = &macd_line - &signal_line;
    
    (macd_line, signal_line, histogram)
}

/// Calculate Bollinger Bands
pub fn bollinger_bands(prices: &ArrayView1<f64>, period: usize, std_dev: f64) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
    let len = prices.len();
    let mut upper = Array1::zeros(len);
    let mut middle = Array1::zeros(len);
    let mut lower = Array1::zeros(len);
    
    for i in period..=len {
        let window = prices.slice(ndarray::s![i-period..i]);
        let mean = window.mean().unwrap_or(0.0);
        let std = window.std(0.0);
        
        middle[i-1] = mean;
        upper[i-1] = mean + std_dev * std;
        lower[i-1] = mean - std_dev * std;
    }
    
    (upper, middle, lower)
}

/// Calculate Average True Range (ATR)
pub fn atr(high: &ArrayView1<f64>, low: &ArrayView1<f64>, close: &ArrayView1<f64>, period: usize) -> Array1<f64> {
    let len = high.len();
    let mut tr = Array1::zeros(len);
    let mut result = Array1::zeros(len);
    
    for i in 1..len {
        let hl = high[i] - low[i];
        let hc = (high[i] - close[i-1]).abs();
        let lc = (low[i] - close[i-1]).abs();
        
        tr[i] = hl.max(hc).max(lc);
    }
    
    for i in period..len {
        result[i] = tr.slice(ndarray::s![i-period+1..=i]).mean().unwrap_or(0.0);
    }
    
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn test_sma() {
        let prices = arr1(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let sma_result = sma(&prices.view(), 3);
        
        assert!((sma_result[2] - 2.0).abs() < 0.01);
        assert!((sma_result[4] - 4.0).abs() < 0.01);
    }

    #[test]
    fn test_rsi() {
        let prices = arr1(&[44.0, 44.5, 44.2, 45.0, 45.5, 45.2, 46.0, 46.5]);
        let rsi_result = rsi(&prices.view(), 3);
        
        // RSI should be between 0 and 100
        for value in rsi_result.iter() {
            assert!(*value >= 0.0 && *value <= 100.0);
        }
    }
}
