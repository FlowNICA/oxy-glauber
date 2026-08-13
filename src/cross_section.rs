// src/cross_section.rs
/// Cross section utilities
pub struct CrossSection;

/// Linear interpolation over a table, matching ROOT's default `TGraph::Eval` behavior.
/// Returns 0.0 if `x` is outside `[xs[0], xs[last]]` (callers already range-check first).
fn lerp_table(x: f64, xs: &[f64], ys: &[f64]) -> f64 {
    if x <= xs[0] {
        return ys[0];
    }
    if x >= xs[xs.len() - 1] {
        return ys[ys.len() - 1];
    }
    // xs is sorted ascending; find the bracketing segment.
    let i = match xs.binary_search_by(|v| v.partial_cmp(&x).unwrap()) {
        Ok(i) => return ys[i],
        Err(i) => i,
    };
    let (x0, x1) = (xs[i - 1], xs[i]);
    let (y0, y1) = (ys[i - 1], ys[i]);
    y0 + (y1 - y0) * (x - x0) / (x1 - x0)
}

// Tkin (GeV) / sigma_inel(np) (mb), from Bystricky_JPhysique48 Table VI.
const NP_X: [f64; 52] = [
    0.300, 0.325, 0.350, 0.375, 0.400, 0.425, 0.450, 0.475, 0.500, 0.525, 0.550, 0.575, 0.60,
    0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.35,
    1.40, 1.45, 1.50, 1.60, 1.70, 1.80, 1.90, 2.00, 2.10, 2.20, 2.30, 2.40, 2.50, 2.60, 2.70,
    2.80, 2.90, 3.00, 3.20, 3.40, 3.60, 3.80, 4.00, 4.20,
];
const NP_Y: [f64; 52] = [
    0.003, 0.046, 0.175, 0.419, 0.783, 1.26, 1.84, 2.50, 3.23, 4.00, 4.81, 5.64, 6.48, 8.15, 9.77,
    11.31, 12.76, 14.10, 15.33, 16.46, 17.48, 18.42, 19.26, 20.03, 20.73, 21.36, 21.93, 22.44,
    22.91, 23.34, 23.72, 24.39, 24.95, 25.42, 25.81, 26.14, 26.43, 26.68, 26.90, 27.10, 27.28,
    27.45, 27.60, 27.75, 27.89, 28.03, 28.30, 28.57, 28.84, 29.13, 29.42, 29.73,
];

// Tkin (GeV) / sigma_inel(pp) (mb), from Bystricky_JPhysique48 Table VI.
const PP_X: [f64; 95] = [
    0.28, 0.29, 0.30, 0.31, 0.32, 0.34, 0.36, 0.38, 0.40, 0.42, 0.44, 0.46, 0.48, 0.50, 0.52,
    0.56, 0.60, 0.64, 0.68, 0.72, 0.76, 0.80, 0.84, 0.88, 0.92, 0.96, 1.00, 1.04, 1.08, 1.12,
    1.16, 1.20, 1.24, 1.28, 1.32, 1.36, 1.40, 1.44, 1.48, 1.52, 1.56, 1.64, 1.68, 1.72, 1.76,
    1.80, 1.84, 1.88, 1.92, 1.96, 2.00, 2.10, 2.20, 2.30, 2.40, 2.50, 2.60, 2.70, 2.80, 2.90,
    3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 25.0, 30.0, 40.0,
    50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 125.0, 150.0, 175.0, 200.0, 225.0, 250.0, 275.0, 300.0,
    325.0, 350.0, 375.0, 400.0, 425.0,
];
const PP_Y: [f64; 95] = [
    0.027, 0.051, 0.082, 0.120, 0.171, 0.321, 0.563, 0.921, 1.41, 2.05, 2.83, 3.74, 4.76, 5.87,
    7.04, 9.48, 11.89, 14.14, 16.15, 17.89, 19.34, 20.53, 21.51, 22.33, 23.03, 23.81, 24.39,
    24.89, 25.30, 25.66, 25.95, 26.21, 26.42, 26.59, 26.74, 26.86, 26.96, 27.04, 27.11, 27.17,
    27.21, 27.28, 27.30, 27.32, 27.33, 27.34, 27.35, 27.36, 27.36, 27.36, 27.36, 27.37, 27.37,
    27.38, 27.38, 27.40, 27.41, 27.43, 27.46, 27.49, 27.52, 27.97, 28.48, 28.92, 29.28, 29.55,
    29.75, 29.89, 30.04, 30.10, 30.09, 30.05, 30.00, 29.88, 29.80, 29.78, 29.91, 30.13, 30.38,
    30.66, 30.93, 31.19, 31.78, 32.23, 32.56, 32.79, 32.92, 32.97, 32.96, 32.89, 32.78, 32.63,
    32.44, 32.24, 32.01,
];

// Energy (GeV) / hard-scattering cross section (mb), from HIJING.
const HARD_X: [f64; 37] = [
    10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 200.0, 300.0, 400.0, 500.0,
    600.0, 700.0, 800.0, 900.0, 1000.0, 2000.0, 3000.0, 4000.0, 5000.0, 6000.0, 7000.0, 8000.0,
    9000.0, 10000.0, 20000.0, 30000.0, 40000.0, 50000.0, 60000.0, 70000.0, 80000.0, 90000.0,
    100000.0,
];
const HARD_Y: [f64; 37] = [
    4.41642385E-03,
    0.188464478,
    0.649059832,
    1.25692534,
    1.92962086,
    2.63335800,
    3.34105325,
    4.05063772,
    4.73708200,
    5.42901564,
    11.6562014,
    16.9493980,
    21.6152058,
    25.8751488,
    29.7382126,
    33.4025536,
    36.8270454,
    40.0877686,
    43.2026596,
    69.1051025,
    89.9556122,
    107.744072,
    124.087372,
    138.940948,
    153.014984,
    166.308075,
    178.939285,
    190.986938,
    294.216583,
    379.379303,
    455.911163,
    525.982910,
    592.784119,
    656.306885,
    717.265808,
    774.954590,
    830.319702,
];

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

    /// Get sigma_NP from energy (0.3-4.2 GeV) - Bystricky parameterization, linearly
    /// interpolated from the real experimental table (Bystricky_JPhysique48 Table VI),
    /// matching `TGlauberMC::GetSigmaNP_Bystricky`.
    pub fn sigma_np_bystricky(energy: f64) -> f64 {
        if energy < 0.3 || energy > 4.2 {
            return 0.0;
        }
        lerp_table(energy, &NP_X, &NP_Y)
    }

    /// Get sigma_PP from energy (0.28-425 GeV) - Bystricky parameterization, linearly
    /// interpolated from the real experimental table (Bystricky_JPhysique48 Table VI),
    /// matching `TGlauberMC::GetSigmaPP_Bystricky`.
    pub fn sigma_pp_bystricky(energy: f64) -> f64 {
        if energy < 0.28 || energy > 425.0 {
            return 0.0;
        }
        lerp_table(energy, &PP_X, &PP_Y)
    }

    /// Get hard scattering cross section from energy (10 GeV to 100 TeV), linearly
    /// interpolated from the real HIJING table, matching `TGlauberMC::GetSigmaHard`.
    pub fn sigma_hard(energy: f64) -> f64 {
        if energy < 10.0 {
            return 0.0;
        }
        lerp_table(energy, &HARD_X, &HARD_Y)
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
