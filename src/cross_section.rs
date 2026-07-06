// src/cross_section.rs
/// Cross section utilities
pub struct CrossSection;

impl CrossSection {
    /// Get sigma_NN from energy (10 GeV to 100 TeV)
    pub fn sigma_nn(energy: f64) -> f64 {
        if energy < 10.0 {
            return 0.0;
        }
        let log_s = (energy * energy).ln();
        // Parameterization from https://arxiv.org/abs/2011.14909
        28.84374 + 0.04584121 * log_s.powf(2.374257)
    }

    /// Get sigma_NP from energy (0.3-4.2 GeV) - Bystricky parameterization
    pub fn sigma_np_bystricky(energy: f64) -> f64 {
        if energy < 0.3 || energy > 4.2 {
            return 0.0;
        }
        // Interpolated from Bystricky table
        let x = energy;
        if x < 0.5 {
            0.0
        } else if x < 1.0 {
            10.0 * (x - 0.5) + 3.0
        } else if x < 2.0 {
            10.0 + 12.0 * (x - 1.0)
        } else if x < 3.0 {
            22.0 + 5.0 * (x - 2.0)
        } else {
            27.0 + 3.0 * (x - 3.0)
        }
    }

    /// Get sigma_PP from energy (0.28-425 GeV) - Bystricky parameterization
    pub fn sigma_pp_bystricky(energy: f64) -> f64 {
        if energy < 0.28 || energy > 425.0 {
            return 0.0;
        }
        let x = energy;
        if x < 0.5 {
            5.0 * (x - 0.28) / 0.22
        } else if x < 1.0 {
            5.0 + 20.0 * (x - 0.5)
        } else if x < 2.0 {
            15.0 + 12.0 * (x - 1.0)
        } else if x < 5.0 {
            27.0 + 0.5 * (x - 2.0)
        } else if x < 10.0 {
            28.5 + 0.15 * (x - 5.0)
        } else if x < 50.0 {
            29.25 + 0.01 * (x - 10.0)
        } else if x < 100.0 {
            29.65 + 0.005 * (x - 50.0)
        } else {
            29.9 + 0.002 * (x - 100.0)
        }
    }

    /// Get hard scattering cross section from energy
    pub fn sigma_hard(energy: f64) -> f64 {
        if energy < 10.0 {
            return 0.0;
        }
        if energy < 100.0 {
            0.01 * (energy - 10.0)
        } else if energy < 1000.0 {
            0.9 + 0.05 * (energy - 100.0)
        } else if energy < 10000.0 {
            45.0 + 0.01 * (energy - 1000.0)
        } else {
            135.0 + 0.008 * (energy - 10000.0)
        }
    }

    /// Compute sigma from beam energy (handles both high and low energy)
    pub fn from_energy(energy: f64) -> (f64, f64, f64) {
        if energy < 10.0 {
            let sig_pp = Self::sigma_pp_bystricky(energy);
            let sig_np = Self::sigma_np_bystricky(energy);
            (sig_pp, sig_np, 0.0)
        } else {
            let sig = Self::sigma_nn(energy);
            let sig_hard = Self::sigma_hard(energy);
            (sig, 0.0, sig_hard)
        }
    }
}
