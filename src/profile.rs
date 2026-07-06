// src/profile.rs
use crate::constants::{MB_TO_FM2, PI};

/// Nucleon-nucleon profile types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NNProfileType {
    HardSphere,
    Gamma,
    Hijing,
    Pythia,
    Trento,
}

/// Nucleon-nucleon interaction profile
#[derive(Debug, Clone)]
pub struct NNProfile {
    pub profile_type: NNProfileType,
    pub sigma_nn: f64, // in mb
    pub omega: f64,    // Gamma parameter (0-2)
    pub w: f64,        // Trento W parameter
    pub mu: f64,       // HIJING mu parameter
    pub m: f64,        // PYTHIA m parameter
    pub rp: f64,       // PYTHIA rp parameter
    pub g: f64,        // Gamma G parameter
}

impl NNProfile {
    pub fn new_hard_sphere(sigma_nn: f64) -> Self {
        Self {
            profile_type: NNProfileType::HardSphere,
            sigma_nn,
            omega: 0.0,
            w: 0.5,
            mu: 3.9,
            m: 1.85,
            rp: 1.0,
            g: 1.0,
        }
    }

    pub fn new_gamma(sigma_nn: f64, omega: f64, g: f64) -> Self {
        Self {
            profile_type: NNProfileType::Gamma,
            sigma_nn,
            omega,
            w: 0.5,
            mu: 3.9,
            m: 1.85,
            rp: 1.0,
            g,
        }
    }

    pub fn new_hijing(sigma_nn: f64, mu: f64) -> Self {
        Self {
            profile_type: NNProfileType::Hijing,
            sigma_nn,
            omega: 7.0,
            w: 0.5,
            mu,
            m: 1.85,
            rp: 1.0,
            g: 1.0,
        }
    }

    pub fn new_pythia(sigma_nn: f64, m: f64, rp: f64) -> Self {
        Self {
            profile_type: NNProfileType::Pythia,
            sigma_nn,
            omega: 8.0,
            w: 0.5,
            mu: 3.9,
            m,
            rp,
            g: 1.0,
        }
    }

    pub fn new_trento(sigma_nn: f64, w: f64) -> Self {
        Self {
            profile_type: NNProfileType::Trento,
            sigma_nn,
            omega: 9.0 + w,
            w,
            mu: 3.9,
            m: 1.85,
            rp: 1.0,
            g: 1.0,
        }
    }

    /// Evaluate the profile at impact parameter b (in fm)
    pub fn eval(&self, b: f64) -> f64 {
        if b < 0.0 {
            return 0.0;
        }
        let b2 = b * b;

        match self.profile_type {
            NNProfileType::HardSphere => {
                let r2 = self.sigma_nn * MB_TO_FM2 / PI;
                if b2 < r2 { 1.0 } else { 0.0 }
            }
            NNProfileType::Gamma => {
                // Gamma distribution: P(b) = G * (1 - Γ(1/ω, G/(ω R^2) * b^2))
                if self.omega <= 0.0 || self.omega > 2.0 {
                    return 0.0;
                }
                let r2 = self.sigma_nn * MB_TO_FM2 / PI;
                let alpha = 1.0 / self.omega;
                let beta = self.g / (self.omega * r2);
                let arg = beta * b2;
                // Incomplete gamma function approximation
                // Using simplified form: 1 - exp(-arg) * sum_{k=0}^{alpha-1} arg^k / k!
                // For non-integer alpha, use approximation
                if self.omega > 0.0 {
                    // For omega close to 1, use exponential
                    self.g * (1.0 - (-arg).exp())
                } else {
                    // General case - approximate
                    let gamma_val = if self.omega < 0.5 {
                        // Approximate for small omega
                        1.0 - (-arg).exp()
                    } else {
                        1.0 - (-arg * 0.5).exp()
                    };
                    self.g * gamma_val
                }
            }
            NNProfileType::Hijing => {
                // HIJING profile: P(b) = 1 - exp(-A * (mu*b)^3 * K3(mu*b))
                // Simplified approximation
                let mu_b = self.mu * b;
                let k3 = if mu_b < 1.0 {
                    2.0 / (mu_b * mu_b * mu_b)
                } else {
                    (PI / (2.0 * mu_b)).sqrt() * (-mu_b).exp() * (1.0 + 15.0 / (8.0 * mu_b))
                };
                let exponent = mu_b * mu_b * mu_b * k3;
                1.0 - (-exponent).exp()
            }
            NNProfileType::Pythia => {
                // PYTHIA profile: P(b) = 1 - exp(-A * exp(-(b/rp)^m))
                let arg = (b / self.rp).powf(self.m);
                1.0 - (-arg).exp()
            }
            NNProfileType::Trento => {
                // TRENTO profile: P(b) = 1 - exp(-A * exp(-b^2/(4*w^2)))
                let arg = b * b / (4.0 * self.w * self.w);
                1.0 - (-arg).exp()
            }
        }
    }

    /// Get the maximum impact parameter where profile is non-zero
    pub fn max_b(&self) -> f64 {
        match self.profile_type {
            NNProfileType::HardSphere => (self.sigma_nn * MB_TO_FM2 / PI).sqrt(),
            _ => 10.0, // Default cutoff
        }
    }
}

/// Create a profile from omega parameter
pub fn profile_from_omega(sigma_nn: f64, omega: f64) -> Option<NNProfile> {
    if omega < 0.0 {
        Some(NNProfile::new_hard_sphere(sigma_nn))
    } else if omega < 2.0 {
        Some(NNProfile::new_gamma(sigma_nn, omega, 1.0))
    } else if omega == 7.0 {
        Some(NNProfile::new_hijing(sigma_nn, 3.9))
    } else if omega == 8.0 {
        Some(NNProfile::new_pythia(sigma_nn, 1.85, 1.0))
    } else if (9.0..=11.0).contains(&omega) {
        let w = omega - 9.0;
        Some(NNProfile::new_trento(sigma_nn, w))
    } else {
        None
    }
}
