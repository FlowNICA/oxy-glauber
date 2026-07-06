// src/random.rs
use rand::Rng;
use rand::rngs::ThreadRng;

/// Extension trait for random number generation
pub trait RngExt {
    fn uniform(&mut self, min: f64, max: f64) -> f64;
    fn gaussian(&mut self, mean: f64, sigma: f64) -> f64;
    fn poisson(&mut self, lambda: f64) -> i32;
}

impl RngExt for ThreadRng {
    fn uniform(&mut self, min: f64, max: f64) -> f64 {
        self.gen_range(min..max)
    }

    fn gaussian(&mut self, mean: f64, sigma: f64) -> f64 {
        use rand_distr::{Distribution, Normal};
        let normal = Normal::new(mean, sigma).unwrap();
        normal.sample(self)
    }

    fn poisson(&mut self, lambda: f64) -> i32 {
        use rand_distr::{Distribution, Poisson};
        let poisson = Poisson::new(lambda).unwrap();
        poisson.sample(self) as i32
    }
}
