// src/glauber3d.rs
//! 3D Monte Carlo Glauber mode, following 3dMCGlauber
//! (<https://github.com/FlowNICA/3dMCGlauber>, C. Shen et al.), at the nucleon level
//! (no valence quarks).
//!
//! Compared with the standard (McGlauber) mode it changes:
//! - the NN collision criterion: a Gaussian probability 1 - exp(-sigma_eff*T_nn(b))
//!   instead of a black disk (`Glauber::hit` / `get_sig_eff`);
//! - Ncoll: on top of the colliding pairs it runs 3dMCGlauber's time-ordered collision
//!   cascade of Lorentz-contracted nuclei, where every collision can produce a QCD
//!   string and decelerates the colliding nucleons (`make_collision_schedule`,
//!   `decide_QCD_strings_production`, `perform_string_production`), giving the number
//!   of executed collisions and of strings;
//! - the eccentricities: computed on a grid from Gaussian-smeared participant thickness
//!   functions T_A, T_B (`MakeDensity::outputTATBEccentricity`) and from the 3D energy
//!   density the strings deposit, as a function of eta_s (`MakeDensity::output_eccentricity`).
//!
//! The nuclei themselves are thrown exactly as in the standard mode (`data/nuclei.ron`).

use crate::constants::{MB_TO_FM2, PI};
use crate::glauber::TGlauberEvent;
use crate::nucleus::{TGlauNucleus, parse_ron};
use rand::Rng;
use rand::RngExt;
use rand::seq::SliceRandom;
use rand_distr::{Distribution, Gamma, Normal};
use serde::Deserialize;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Proton mass in GeV, as in 3dMCGlauber's `PhysConsts::MProton`
const M_PROTON: f64 = 0.938;

/// Labels of the six T_A/T_B weightings of the transverse density used for the
/// eccentricities, in 3dMCGlauber's `outputTATBEccentricity` order
pub const TATB_LABELS: [&str; 6] = ["TAplusTB", "sqrtTATB", "TATB23", "TATB", "TA", "TB"];

/// Errors from reading or validating a run configuration file
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("failed to read run configuration '{path}': {msg}")]
    Read { path: String, msg: String },
    #[error("failed to parse run configuration '{path}': {msg}")]
    Parse { path: String, msg: String },
    #[error("invalid 3D Glauber parameter: {0}")]
    Invalid(String),
}

/// Run configuration file (RON), passed to the binaries with `--config`.
/// See `config/run_3d.ron` for a documented example.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RunConfig {
    /// `false` (default): standard McGlauber mode; `true`: 3D Glauber mode
    pub three_d: bool,
    /// Parameters of the 3D mode (ignored unless `three_d` is true)
    pub three_d_params: Glauber3DParams,
}

impl RunConfig {
    pub fn from_file(path: &str) -> Result<Self, ConfigError> {
        let text = std::fs::read_to_string(path).map_err(|e| ConfigError::Read {
            path: path.to_string(),
            msg: e.to_string(),
        })?;
        parse_ron(&text).map_err(|e| ConfigError::Parse {
            path: path.to_string(),
            msg: e.to_string(),
        })
    }
}

/// Parameters of the 3D Glauber mode. Defaults are those of 3dMCGlauber's `input` file.
#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Glauber3DParams {
    /// Collision energy sqrt(s_NN) in GeV; sets the beam rapidity
    pub roots: f64,
    /// `false`: use the run's sigma_NN (`--signn`); `true`: use 3dMCGlauber's
    /// parameterization of sigma_NN^inel(sqrt(s)) at `roots`
    pub sigma_nn_from_roots: bool,
    /// Shadowing factor for producing strings from multiple scatterings, in [0, 1]
    pub shadowing_factor: f64,
    /// Order in which collisions are offered strings: 1 = random, 2 = anti-time
    /// ordered, anything else = time ordered
    pub string_production_mode: i32,
    /// Rapidity loss: 1 = LEXUS, 2 = parameterization, 3 = parameterization with
    /// logit-normal fluctuations, 4 = piecewise linear with logit-normal fluctuations
    pub rapidity_loss_method: i32,
    /// String evolution: 1 = fixed rapidity loss, 2 = tau_form and m/sigma fluctuate,
    /// 3 = tau_form fluctuates, 4 = m/sigma fluctuates
    pub string_evolution_mode: i32,
    /// Mean string formation time tau_form in fm
    pub tau_form_mean: f64,
    /// Shape/rate of the gamma distribution of tau_form fluctuations (mode 2)
    pub tau_form_fluct_gamma_beta: f64,
    /// Parameterization of methods 2/3: y_loss = slope * (y^a * tanh(y))^b,
    /// a = alpha/(1-alpha), alpha = alpha2/alpha1, b = alpha2/a
    pub yloss_param_slope: f64,
    pub yloss_param_alpha1: f64,
    pub yloss_param_alpha2: f64,
    /// Width of the logit-normal fluctuation at RHIC (y_init <= 5.5, method 3) and LHC
    /// (y_init >= 8, methods 3 and 4) energies, interpolated in between
    pub yloss_param_fluct_var_rhic: f64,
    pub yloss_param_fluct_var_lhc: f64,
    /// Piecewise linear mean rapidity loss of method 4 at y_init = 2, 4, 6, 8, 10;
    /// `yloss_param4_at8` defaults to the mean of the values at 6 and 10
    pub yloss_param4_at2: f64,
    pub yloss_param4_at4: f64,
    pub yloss_param4_at6: f64,
    pub yloss_param4_at8: Option<f64>,
    pub yloss_param4_at10: f64,
    /// Width of the logit-normal fluctuation of method 4 at RHIC energies
    pub yloss_param4_var: f64,
    /// Compute the 3D string energy density and eps_n(eta_s); this is the slow part
    /// of the 3D mode, turn it off if only Ncoll/Nstrings/T_A,T_B eccentricities matter
    pub eccentricity_eta: bool,
    /// Highest eccentricity order computed (3dMCGlauber: 3)
    pub max_order: usize,
    /// Transverse grid: `grid_nx` x `grid_nx` cells of `grid_dx` fm, centered at 0
    pub grid_nx: usize,
    pub grid_dx: f64,
    /// Space-time rapidity grid: `grid_neta` cells of `grid_deta`, centered at 0
    pub grid_neta: usize,
    pub grid_deta: f64,
    /// Gaussian smearing widths in the transverse plane (fm) and in eta_s
    pub sigma_x: f64,
    pub sigma_eta: f64,
    /// Fraction of the transverse separation of the string ends by which the string
    /// is tilted along eta_s (3dMCGlauber's `setStringTransShiftFrac`)
    pub string_transverse_shift_frac: f64,
}

impl Default for Glauber3DParams {
    fn default() -> Self {
        Self {
            roots: 200.0,
            sigma_nn_from_roots: false,
            shadowing_factor: 0.6,
            string_production_mode: 1,
            rapidity_loss_method: 4,
            string_evolution_mode: 4,
            tau_form_mean: 0.5,
            tau_form_fluct_gamma_beta: 1.0,
            yloss_param_slope: 1.32,
            yloss_param_alpha1: 1.8,
            yloss_param_alpha2: 0.35,
            yloss_param_fluct_var_rhic: 0.6,
            yloss_param_fluct_var_lhc: 0.6,
            yloss_param4_at2: 1.70,
            yloss_param4_at4: 2.00,
            yloss_param4_at6: 2.20,
            yloss_param4_at8: None,
            yloss_param4_at10: 2.95,
            yloss_param4_var: 0.5,
            eccentricity_eta: true,
            max_order: 3,
            grid_nx: 100,
            grid_dx: 0.2,
            grid_neta: 72,
            grid_deta: 0.2,
            sigma_x: 0.4,
            sigma_eta: 0.5,
            string_transverse_shift_frac: 0.0,
        }
    }
}

/// Per-event results of the 3D mode, stored in `TGlauberEvent::three_d`
#[derive(Debug, Clone, Default)]
pub struct Event3D {
    /// Collisions executed in the time-ordered cascade (3dMCGlauber's reported Ncoll)
    pub ncoll_exec: f32,
    /// Number of strings decided for the event (3dMCGlauber's centrality estimator)
    pub nstrings: f32,
    /// Transverse area pi*x_rms*y_rms of the string production points, in fm^2
    pub trans_area: f32,
    /// eps_n and Psi_n of the T_A/T_B weightings: `[TATB_LABELS index][n - 1]`
    pub ecc_tatb: Vec<Vec<f32>>,
    pub psi_tatb: Vec<Vec<f32>>,
    /// dE/deta_s in GeV per eta_s cell (empty unless `eccentricity_eta`)
    pub de_deta: Vec<f32>,
    /// eps_n(eta_s) and Psi_n(eta_s) of the string energy density: `[n - 1][eta cell]`
    pub ecc_eta: Vec<Vec<f32>>,
    pub psi_eta: Vec<Vec<f32>>,
}

/// One output column of the 3D mode: a value per event, or a fixed-length array per
/// event (the eta_s profiles, one entry per eta_s cell)
#[derive(Debug, Clone)]
pub enum Column3D {
    Scalar(Vec<f32>),
    Array(Vec<Vec<f32>>),
}

/// The 3D-mode output columns of `events`, named as written by the run binaries:
/// `NcollExec`, `Nstrings`, `TransArea`, `Ecc{n}_{label}` / `Psi{n}_{label}` for each
/// of `TATB_LABELS`, and, with `eccentricity_eta`, the arrays `dEdEtas`, `EccEta{n}`
/// and `PsiEta{n}` over the eta_s cells of `Glauber3D::eta_grid`. Events without 3D
/// results are written as zeros.
pub fn output_columns(events: &[TGlauberEvent], g3: &Glauber3D) -> Vec<(String, Column3D)> {
    let empty = Event3D::default();
    let e3: Vec<&Event3D> = events
        .iter()
        .map(|e| e.three_d.as_deref().unwrap_or(&empty))
        .collect();
    let max_order = g3.params.max_order;
    let neta = g3.params.grid_neta;
    let scalar = |f: &dyn Fn(&Event3D) -> f32| Column3D::Scalar(e3.iter().map(|e| f(e)).collect());
    let at = |v: &[Vec<f32>], i: usize, j: usize| {
        v.get(i).and_then(|r| r.get(j)).copied().unwrap_or(0.0)
    };
    let array = |v: &dyn Fn(&Event3D) -> &[f32]| {
        Column3D::Array(
            e3.iter()
                .map(|e| {
                    let row = v(e);
                    if row.len() == neta {
                        row.to_vec()
                    } else {
                        vec![0.0; neta]
                    }
                })
                .collect(),
        )
    };

    let mut columns = vec![
        ("NcollExec".to_string(), scalar(&|e| e.ncoll_exec)),
        ("Nstrings".to_string(), scalar(&|e| e.nstrings)),
        ("TransArea".to_string(), scalar(&|e| e.trans_area)),
    ];
    for (c, label) in TATB_LABELS.iter().enumerate() {
        for n in 1..=max_order {
            columns.push((
                format!("Ecc{n}_{label}"),
                scalar(&|e| at(&e.ecc_tatb, c, n - 1)),
            ));
            columns.push((
                format!("Psi{n}_{label}"),
                scalar(&|e| at(&e.psi_tatb, c, n - 1)),
            ));
        }
    }
    if g3.params.eccentricity_eta {
        columns.push(("dEdEtas".to_string(), array(&|e| &e.de_deta)));
        for n in 1..=max_order {
            columns.push((format!("EccEta{n}"), array(&|e| row(&e.ecc_eta, n - 1))));
            columns.push((format!("PsiEta{n}"), array(&|e| row(&e.psi_eta, n - 1))));
        }
    }
    columns
}

/// Row `i` of a per-order table, empty if missing
fn row(v: &[Vec<f32>], i: usize) -> &[f32] {
    v.get(i).map_or(&[], |r| r.as_slice())
}

/// 3D Glauber mode: validated parameters plus the quantities derived from them
#[derive(Debug, Clone)]
pub struct Glauber3D {
    params: Glauber3DParams,
    /// Inelastic NN cross section used, in mb
    sigma_nn: f64,
    /// Gaussian nucleon width w, in fm
    width: f64,
    /// Effective opacity sigma_eff, in fm^2
    sigma_eff: f64,
    /// Beam rapidity
    ybeam: f64,
    yloss_param_a: f64,
    yloss_param_b: f64,
    tau_form_gamma: Gamma<f64>,
}

impl Glauber3D {
    /// `run_sigma_nn` (mb) is used unless `sigma_nn_from_roots` is set.
    pub fn new(params: Glauber3DParams, run_sigma_nn: f64) -> Result<Self, ConfigError> {
        let invalid = |msg: String| Err(ConfigError::Invalid(msg));
        let p = &params;
        if p.roots <= 2.0 * M_PROTON {
            return invalid(format!(
                "roots = {} must exceed 2*m_p = {}",
                p.roots,
                2.0 * M_PROTON
            ));
        }
        if !(0.0..=1.0).contains(&p.shadowing_factor) {
            return invalid(format!(
                "shadowing_factor = {} must be in [0, 1]",
                p.shadowing_factor
            ));
        }
        if !(1..=4).contains(&p.rapidity_loss_method) {
            return invalid(format!(
                "rapidity_loss_method = {} must be 1-4",
                p.rapidity_loss_method
            ));
        }
        if !(1..=4).contains(&p.string_evolution_mode) {
            return invalid(format!(
                "string_evolution_mode = {} must be 1-4",
                p.string_evolution_mode
            ));
        }
        if p.tau_form_mean <= 0.0 || p.tau_form_fluct_gamma_beta <= 0.0 {
            return invalid("tau_form_mean and tau_form_fluct_gamma_beta must be > 0".into());
        }
        if p.yloss_param_alpha1 < 1.0 || !(0.0..1.0).contains(&p.yloss_param_alpha2) {
            return invalid("need yloss_param_alpha1 >= 1 and 0 <= yloss_param_alpha2 < 1".into());
        }
        if !(1..=9).contains(&p.max_order) {
            return invalid(format!("max_order = {} must be 1-9", p.max_order));
        }
        if p.grid_nx == 0 || p.grid_neta == 0 || p.grid_dx <= 0.0 || p.grid_deta <= 0.0 {
            return invalid("grid sizes and spacings must be > 0".into());
        }
        if p.sigma_x <= 0.0 || p.sigma_eta <= 0.0 {
            return invalid("sigma_x and sigma_eta must be > 0".into());
        }

        let sigma_nn = if p.sigma_nn_from_roots {
            sigma_nn_inel(p.roots)
        } else {
            run_sigma_nn
        };
        if sigma_nn <= 0.0 {
            return invalid(format!("sigma_NN = {sigma_nn} mb must be > 0"));
        }
        let (width, sigma_eff) = sig_eff(sigma_nn);
        let alpha = p.yloss_param_alpha2 / p.yloss_param_alpha1;
        let yloss_param_a = alpha / (1.0 - alpha);
        let yloss_param_b = p.yloss_param_alpha2 / yloss_param_a;
        // std::gamma_distribution(alpha, beta) is shape alpha, scale beta
        let tau_form_gamma =
            Gamma::new(p.tau_form_fluct_gamma_beta, p.tau_form_fluct_gamma_beta)
                .map_err(|e| ConfigError::Invalid(format!("tau_form gamma distribution: {e}")))?;
        Ok(Self {
            ybeam: (p.roots / (2.0 * M_PROTON)).acosh(),
            params,
            sigma_nn,
            width,
            sigma_eff,
            yloss_param_a,
            yloss_param_b,
            tau_form_gamma,
        })
    }

    pub fn params(&self) -> &Glauber3DParams {
        &self.params
    }
    /// Inelastic NN cross section used, in mb
    pub fn sigma_nn(&self) -> f64 {
        self.sigma_nn
    }
    /// Gaussian nucleon width, in fm
    pub fn width(&self) -> f64 {
        self.width
    }
    /// Effective opacity, in fm^2
    pub fn sigma_eff(&self) -> f64 {
        self.sigma_eff
    }
    pub fn ybeam(&self) -> f64 {
        self.ybeam
    }

    /// Space-time rapidity of the center of each eta_s cell
    pub fn eta_grid(&self) -> Vec<f64> {
        grid(self.params.grid_neta, self.params.grid_deta)
    }

    /// Probability that two nucleons at transverse distance^2 `d2` (fm^2) collide,
    /// 1 - exp(-sigma_eff * T_nn(d)) with the overlap T_nn of two Gaussian nucleons
    pub fn hit_probability(&self, d2: f64) -> f64 {
        let w2 = self.width * self.width;
        let t_nn = (-d2 / (4.0 * w2)).exp() / (4.0 * PI * w2);
        1.0 - (-self.sigma_eff * t_nn).exp()
    }

    /// Run the collision cascade and compute the eccentricities for an event whose
    /// colliding pairs are `bc[j][i]` (nucleon j of A, the projectile moving in +z,
    /// with nucleon i of B, the target).
    pub fn process_event<R: Rng>(
        &self,
        nucleus_a: &TGlauNucleus,
        nucleus_b: &TGlauNucleus,
        bc: &[Vec<bool>],
        rng: &mut R,
    ) -> Event3D {
        let mut cascade = Cascade::new(self, nucleus_a, nucleus_b, bc);
        let nstrings = cascade.decide_strings(self, rng);
        let ncoll_exec = cascade.perform_string_production(self, rng);
        let strings = cascade.finalize_strings();

        let mut event = Event3D {
            ncoll_exec: ncoll_exec as f32,
            nstrings: nstrings as f32,
            ..Default::default()
        };
        if strings.is_empty() {
            return event;
        }

        // Transverse positions relative to the string center of mass, as in
        // `Glauber::prepare_output_QCD_strings` / `prepareParticipantList`
        let n = strings.len() as f64;
        let x_o = strings.iter().map(|s| s.x_prod[1]).sum::<f64>() / n;
        let y_o = strings.iter().map(|s| s.x_prod[2]).sum::<f64>() / n;
        let x_rms = (strings
            .iter()
            .map(|s| (s.x_prod[1] - x_o).powi(2))
            .sum::<f64>()
            / n)
            .sqrt();
        let y_rms = (strings
            .iter()
            .map(|s| (s.x_prod[2] - y_o).powi(2))
            .sum::<f64>()
            / n)
            .sqrt();
        event.trans_area =
            (PI * (x_rms + self.params.sigma_x) * (y_rms + self.params.sigma_x)) as f32;

        let (ecc, psi) = self.tatb_eccentricities(&cascade.nucleons, x_o, y_o);
        event.ecc_tatb = ecc;
        event.psi_tatb = psi;

        if self.params.eccentricity_eta {
            let outputs: Vec<StringOutput> = strings
                .iter()
                .map(|s| s.output(&cascade.nucleons, x_o, y_o))
                .collect();
            event.de_deta = self.energy_eta_distribution(&outputs);
            let (ecc, psi) = self.eta_eccentricities(&outputs);
            event.ecc_eta = ecc;
            event.psi_eta = psi;
        }
        event
    }

    // ----- string formation and rapidity loss (Glauber.cpp) -----

    fn tau_form<R: Rng>(&self, rng: &mut R) -> f64 {
        let mean = self.params.tau_form_mean;
        if self.params.string_evolution_mode == 2 {
            (mean / self.tau_form_gamma.sample(rng)).clamp(0.1, 6.0 * mean)
        } else {
            mean
        }
    }

    /// `Glauber::get_tau_form_and_moversigma`: (tau_form, m/sigma, y_loss)
    fn tau_form_and_m_over_sigma<R: Rng>(&self, y_in_lrf: f64, rng: &mut R) -> (f64, f64, f64) {
        const EPS: f64 = 1e-16;
        let mut tau_form = self.tau_form(rng);
        let mut m_over_sigma = 1.0;
        let y_loss;
        match self.params.string_evolution_mode {
            1 => {
                let y = (tau_form * tau_form / (2.0 * m_over_sigma * m_over_sigma) + 1.0).acosh();
                y_loss = y.min(0.9 * y_in_lrf);
                m_over_sigma = tau_form / (2.0 * (y_loss.cosh() - 1.0)).sqrt().max(EPS);
            }
            3 => {
                y_loss = self.sample_rapidity_loss(y_in_lrf, rng);
                tau_form = m_over_sigma * (2.0 * (y_loss.cosh() - 1.0)).sqrt();
            }
            _ => {
                // modes 2 and 4
                y_loss = self.sample_rapidity_loss(y_in_lrf, rng);
                m_over_sigma = tau_form / (2.0 * (y_loss.cosh() - 1.0)).sqrt().max(EPS);
            }
        }
        (tau_form, m_over_sigma, y_loss)
    }

    /// `Glauber::sample_rapidity_loss_shell`
    fn sample_rapidity_loss<R: Rng>(&self, y_init: f64, rng: &mut R) -> f64 {
        match self.params.rapidity_loss_method {
            1 => {
                // LEXUS
                let sinh_y = y_init.sinh();
                let f = rng.random::<f64>() * ((2.0 * y_init).sinh() - sinh_y) + sinh_y;
                2.0 * y_init - f.asinh()
            }
            2 => self.yloss_parametrization(y_init),
            _ => self.yloss_with_fluct(y_init, rng),
        }
    }

    fn yloss_parametrization(&self, y_init: f64) -> f64 {
        self.params.yloss_param_slope
            * (y_init.powf(self.yloss_param_a) * y_init.tanh()).powf(self.yloss_param_b)
    }

    fn yloss_piecewise(&self, y_init: f64) -> f64 {
        let p = &self.params;
        let (y2, y4, y6, y10) = (
            p.yloss_param4_at2,
            p.yloss_param4_at4,
            p.yloss_param4_at6,
            p.yloss_param4_at10,
        );
        let y8 = p.yloss_param4_at8.unwrap_or((y6 + y10) / 2.0);
        if y_init < 2.0 {
            y2 / 2.0 * y_init
        } else if y_init < 4.0 {
            (y4 - y2) / 2.0 * y_init + (2.0 * y2 - y4)
        } else if y_init < 6.0 {
            (y6 - y4) / 2.0 * y_init + (3.0 * y4 - 2.0 * y6)
        } else if y_init < 8.0 {
            (y8 - y6) / 2.0 * y_init + (4.0 * y6 - 3.0 * y8)
        } else {
            (y10 - y8) / 2.0 * y_init + (5.0 * y8 - 4.0 * y10)
        }
    }

    /// `Glauber::sample_rapidity_loss_from_parametrization_with_fluct`: a logit-normal
    /// distribution of y_loss in [0, y_init] with the parameterized mean
    fn yloss_with_fluct<R: Rng>(&self, y_init: f64, rng: &mut R) -> f64 {
        let p = &self.params;
        let (y_mean, var_rhic) = if p.rapidity_loss_method == 3 {
            (
                self.yloss_parametrization(y_init),
                p.yloss_param_fluct_var_rhic,
            )
        } else {
            (self.yloss_piecewise(y_init), p.yloss_param4_var)
        };
        let var_lhc = p.yloss_param_fluct_var_lhc;
        const Y_RHIC: f64 = 5.5;
        const Y_LHC: f64 = 8.0;
        let var = if y_init > Y_LHC {
            var_lhc
        } else if y_init > Y_RHIC {
            var_rhic + (y_init - Y_RHIC) / (Y_LHC - Y_RHIC) * (var_lhc - var_rhic)
        } else {
            var_rhic
        };
        let x = Normal::new(0.0, var).map_or(0.0, |d| d.sample(rng));
        let logit = 1.0 / (1.0 + (-x).exp());
        if (2.0 * y_mean - y_init).abs() > 1e-15 {
            let aa = (2.0 * y_mean - y_init) / (2.0 * y_mean * y_init * (y_init - y_mean));
            let bb = (y_init * y_init - 2.0 * y_mean * y_mean)
                / (2.0 * y_init * y_mean * (y_init - y_mean));
            (-bb + (bb * bb + 4.0 * aa * logit).sqrt()) / (2.0 * aa)
        } else {
            logit * y_init
        }
    }

    // ----- eccentricities (MakeDensity.cpp) -----

    /// eps_n of the six T_A/T_B weightings (`MakeDensity::outputTATBEccentricity`)
    fn tatb_eccentricities(
        &self,
        nucleons: &[Nucleon3D],
        x_o: f64,
        y_o: f64,
    ) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let p = &self.params;
        let xg = grid(p.grid_nx, p.grid_dx);
        let cells = p.grid_nx * p.grid_nx;
        let mut ta = vec![0.0; cells];
        let mut tb = vec![0.0; cells];
        for nucleon in nucleons.iter().filter(|n| n.wounded) {
            let t = if nucleon.projectile { &mut ta } else { &mut tb };
            self.stamp(&xg, nucleon.x[1] - x_o, nucleon.x[2] - y_o, 1.0, t);
        }
        let mut ecc = Vec::with_capacity(TATB_LABELS.len());
        let mut psi = Vec::with_capacity(TATB_LABELS.len());
        for config in 0..TATB_LABELS.len() {
            let density: Vec<f64> = ta
                .iter()
                .zip(&tb)
                .map(|(&a, &b)| match config {
                    0 => a + b,
                    1 => (a * b).sqrt(),
                    2 => (a * b).powf(2.0 / 3.0),
                    3 => a * b,
                    4 => a,
                    _ => b,
                })
                .collect();
            let (e, s) = eccentricities(&xg, &density, p.max_order);
            ecc.push(e);
            psi.push(s);
        }
        (ecc, psi)
    }

    /// Longitudinal profiles of one string over the eta_s grid: the energy deposited
    /// by the string itself (between its ends, Gaussian tails beyond) and by the
    /// remnants at its left/right ends, each normalized to unit integral
    /// (`MakeDensity::compute_energyDensity_3D_distribution`). Returns the energy per
    /// unit eta_s in each cell.
    fn string_eta_profile(&self, s: &StringOutput, eta: &[f64]) -> Vec<f64> {
        let two_sigma2 = 2.0 * self.params.sigma_eta * self.params.sigma_eta;
        let cutoff = 5.0 * self.params.sigma_eta;
        let gauss = |d: f64| (-d * d / two_sigma2).exp();
        let e_string = M_PROTON
            * (s.y_i_left.cosh() + s.y_i_right.cosh() - s.y_left.cosh() - s.y_right.cosh());
        let e_rem_l = if s.remnant_left {
            M_PROTON * s.y_left.cosh()
        } else {
            0.0
        };
        let e_rem_r = if s.remnant_right {
            M_PROTON * s.y_right.cosh()
        } else {
            0.0
        };

        let n = eta.len();
        let (mut fc, mut fl, mut fr) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        for (k, &e) in eta.iter().enumerate() {
            if e > s.eta_left && e < s.eta_right {
                let y = s.y_left
                    + (s.y_right - s.y_left) / (s.eta_right - s.eta_left) * (e - s.eta_left);
                fc[k] = y.cosh();
            } else if e > s.eta_right && e - s.eta_right < cutoff {
                fc[k] = gauss(e - s.eta_right) * s.y_right.cosh();
            } else if e < s.eta_left && s.eta_left - e < cutoff {
                fc[k] = gauss(e - s.eta_left) * s.y_left.cosh();
            }
            if (e - s.eta_right).abs() < cutoff {
                fr[k] = gauss(e - s.eta_right) * s.y_right.cosh();
            }
            if (e - s.eta_left).abs() < cutoff {
                fl[k] = gauss(e - s.eta_left) * s.y_left.cosh();
            }
        }
        let deta = self.params.grid_deta;
        // 3dMCGlauber divides by these norms unguarded, which turns a string lying
        // entirely outside the eta_s grid into NaNs; here its energy is just dropped.
        let scale = |f: &[f64], energy: f64| {
            let norm = f.iter().sum::<f64>() * deta;
            if norm > 0.0 { energy / norm } else { 0.0 }
        };
        let (cc, cl, cr) = (
            scale(&fc, e_string),
            scale(&fl, e_rem_l),
            scale(&fr, e_rem_r),
        );
        (0..n)
            .map(|k| cc * fc[k] + cl * fl[k] + cr * fr[k])
            .collect()
    }

    /// dE/deta_s (`MakeDensity::output_energyDensity_eta_distribution`)
    fn energy_eta_distribution(&self, strings: &[StringOutput]) -> Vec<f32> {
        let eta = self.eta_grid();
        let mut de = vec![0.0; eta.len()];
        for s in strings {
            for (d, v) in de.iter_mut().zip(self.string_eta_profile(s, &eta)) {
                *d += v;
            }
        }
        de.into_iter().map(|v| v as f32).collect()
    }

    /// eps_n(eta_s) of the string energy density (`MakeDensity::output_eccentricity`)
    fn eta_eccentricities(&self, strings: &[StringOutput]) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let p = &self.params;
        let eta = self.eta_grid();
        let xg = grid(p.grid_nx, p.grid_dx);
        let slice = p.grid_nx * p.grid_nx;
        let mut ed = vec![0.0; slice * eta.len()];
        let mut stamp = vec![0.0; slice];
        for s in strings {
            let profile = self.string_eta_profile(s, &eta);
            let tilt = p.string_transverse_shift_frac != 0.0;
            if !tilt {
                stamp.iter_mut().for_each(|v| *v = 0.0);
                self.stamp(&xg, s.x_center, s.y_center, 1.0, &mut stamp);
            }
            for (k, &weight) in profile.iter().enumerate() {
                if weight == 0.0 {
                    continue;
                }
                let ed_k = &mut ed[k * slice..(k + 1) * slice];
                if tilt {
                    let frac = ((eta[k] - s.eta_left) / (s.eta_right - s.eta_left + 1e-16))
                        .clamp(0.0, 1.0);
                    let shift = p.string_transverse_shift_frac * (0.5 - frac) / 2.0;
                    let x_t = s.x_center + shift * (s.left_x - s.right_x);
                    let y_t = s.y_center + shift * (s.left_y - s.right_y);
                    self.stamp(&xg, x_t, y_t, weight, ed_k);
                } else {
                    for (e, &st) in ed_k.iter_mut().zip(&stamp) {
                        *e += weight * st;
                    }
                }
            }
        }

        let mut ecc = vec![vec![0.0f32; eta.len()]; p.max_order];
        let mut psi = vec![vec![0.0f32; eta.len()]; p.max_order];
        for k in 0..eta.len() {
            let (e, s) = eccentricities(&xg, &ed[k * slice..(k + 1) * slice], p.max_order);
            for n in 0..p.max_order {
                ecc[n][k] = e[n];
                psi[n][k] = s[n];
            }
        }
        (ecc, psi)
    }

    /// Add `weight` times a normalized 2D Gaussian of width sigma_x centered at
    /// (x0, y0) to the transverse grid `out`, truncated at 5 sigma as in 3dMCGlauber
    /// (only the cells within the cutoff are visited).
    fn stamp(&self, xg: &[f64], x0: f64, y0: f64, weight: f64, out: &mut [f64]) {
        let sigma = self.params.sigma_x;
        let two_sigma2 = 2.0 * sigma * sigma;
        let cut2 = 25.0 * sigma * sigma;
        let norm = weight / (2.0 * PI * sigma * sigma);
        let nx = xg.len();
        let range = |c: f64| {
            let lo = ((c - 5.0 * sigma - xg[0]) / self.params.grid_dx)
                .floor()
                .max(0.0) as usize;
            let hi = (((c + 5.0 * sigma - xg[0]) / self.params.grid_dx).ceil() as isize + 1)
                .clamp(0, nx as isize) as usize;
            lo.min(nx)..hi
        };
        for i in range(x0) {
            let dx = xg[i] - x0;
            for j in range(y0) {
                let dy = xg[j] - y0;
                let r2 = dx * dx + dy * dy;
                if r2 < cut2 {
                    out[i * nx + j] += norm * (-r2 / two_sigma2).exp();
                }
            }
        }
    }
}

/// Cell positions -N*d/2 + i*d of a grid, as in `MakeDensity`
fn grid(n: usize, d: f64) -> Vec<f64> {
    (0..n)
        .map(|i| -(n as f64) * d / 2.0 + i as f64 * d)
        .collect()
}

/// eps_n and Psi_n (n = 1..=max_order) of a density on the square transverse grid
/// `xg` x `xg`, about its center of mass, with weights r^3 for n = 1 and r^n
/// otherwise. Psi_n follows the standard mode's convention ((atan2 + pi)/n).
fn eccentricities(xg: &[f64], density: &[f64], max_order: usize) -> (Vec<f32>, Vec<f32>) {
    let nx = xg.len();
    let (mut sum, mut sx, mut sy) = (0.0, 0.0, 0.0);
    for (idx, &d) in density.iter().enumerate() {
        if d > 0.0 {
            sum += d;
            sx += d * xg[idx / nx];
            sy += d * xg[idx % nx];
        }
    }
    let mut ecc = vec![0.0f32; max_order];
    let mut psi = vec![0.0f32; max_order];
    if sum <= 0.0 {
        return (ecc, psi);
    }
    let (x_o, y_o) = (sx / sum, sy / sum);
    let mut re = vec![0.0; max_order];
    let mut im = vec![0.0; max_order];
    let mut norm = vec![0.0; max_order];
    for (idx, &d) in density.iter().enumerate() {
        if d <= 0.0 {
            continue;
        }
        let (x, y) = (xg[idx / nx] - x_o, xg[idx % nx] - y_o);
        let r2 = x * x + y * y;
        // z^n = r^n e^{i n phi}, built up incrementally
        let (mut zr, mut zi) = (1.0, 0.0);
        for n in 1..=max_order {
            (zr, zi) = (zr * x - zi * y, zr * y + zi * x);
            let rn = r2.powf(n as f64 / 2.0);
            // n = 1 uses r^3 e^{i phi} = r^2 z
            let (w, wr, wi) = if n == 1 {
                (r2 * rn, r2 * zr, r2 * zi)
            } else {
                (rn, zr, zi)
            };
            norm[n - 1] += d * w;
            re[n - 1] += d * wr;
            im[n - 1] += d * wi;
        }
    }
    for n in 1..=max_order {
        if norm[n - 1] > 0.0 {
            ecc[n - 1] = ((re[n - 1].powi(2) + im[n - 1].powi(2)).sqrt() / norm[n - 1]) as f32;
            psi[n - 1] = ((im[n - 1].atan2(re[n - 1]) + PI) / n as f64) as f32;
        }
    }
    (ecc, psi)
}

/// 3dMCGlauber's parameterization of the inelastic NN cross section (mb) at
/// sqrt(s) = `ecm` GeV (`Glauber::compute_NN_inelastic_cross_section`)
pub fn sigma_nn_inel(ecm: f64) -> f64 {
    let s = ecm * ecm;
    let ls = s.ln();
    let total = 44.4 - 2.9 * ls + 0.33 * ls * ls;
    let elastic = 11.4 - 1.52 * ls + 0.13 * ls * ls;
    total - elastic
}

/// Gaussian nucleon width w and effective opacity sigma_eff (fm^2) such that
/// int d^2b [1 - exp(-sigma_eff T_nn(b))] = sigma_NN (`Glauber::get_sig_eff`)
fn sig_eff(sigma_nn: f64) -> (f64, f64) {
    let width = (sigma_nn * MB_TO_FM2 / PI).sqrt() / 8.0f64.sqrt();
    let sigin = sigma_nn * MB_TO_FM2;
    const NINT: usize = 100;
    let db = 10.0 * width / NINT as f64;
    let b: Vec<f64> = (0..NINT).map(|i| (i as f64 + 0.5) * db).collect();
    let tnn: Vec<f64> = b
        .iter()
        .map(|&b| (-b * b / (4.0 * width * width)).exp() / (PI * 4.0 * width * width))
        .collect();
    let prefactor = 2.0 * PI * db;
    let mut sigeff = 10.0;
    for _ in 0..1000 {
        let previous = sigeff;
        let (mut sum, mut dn) = (0.0, 0.0);
        for (&b, &t) in b.iter().zip(&tnn) {
            sum += prefactor * b * (1.0 - (-sigeff * t).exp());
            dn += prefactor * b * t * (-sigeff * t).exp();
        }
        sigeff -= (sum - sigin) / dn;
        if (sigeff - previous).abs() <= 1e-4 {
            break;
        }
    }
    (width, sigeff)
}

// ----- the collision cascade -----

/// Space-time rapidity of a point (t, z)
fn eta_s(t: f64, z: f64) -> f64 {
    0.5 * ((t + z) / (t - z)).ln()
}

/// A nucleon of the cascade: projectile (A) nucleons come first, then target (B)
#[derive(Debug, Clone)]
struct Nucleon3D {
    projectile: bool,
    wounded: bool,
    /// (t, x, y, z) in fm
    x: [f64; 4],
    /// Longitudinal rapidity (transverse momenta are zero throughout)
    y: f64,
    /// Collisions executed so far (validates scheduled collisions)
    collided_times: u32,
    /// Nucleons it collides with
    partners: Vec<usize>,
    /// Strings decided per partner (first decision only, like
    /// `Nucleon::get_number_of_connections(partner)`)
    connections: Vec<(usize, u32)>,
    total_connections: u32,
    remnant_set: bool,
}

impl Nucleon3D {
    fn velocity(&self) -> f64 {
        self.y.tanh()
    }
    fn strings_with(&self, partner: usize) -> u32 {
        self.connections
            .iter()
            .find(|(p, _)| *p == partner)
            .map_or(0, |&(_, n)| n)
    }
}

/// A scheduled collision between projectile nucleon `proj` and target nucleon `targ`
#[derive(Debug, Clone)]
struct Collision {
    x: [f64; 4],
    proj: usize,
    targ: usize,
    /// `collided_times` of both nucleons when scheduled; the collision is skipped if
    /// either nucleon collided in between (`CollisionEvent::is_valid`)
    proj_collided: u32,
    targ_collided: u32,
    n_strings: u32,
    /// Insertion order, to break ties in time deterministically
    seq: u64,
}

impl PartialEq for Collision {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for Collision {}
impl PartialOrd for Collision {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Collision {
    /// Reversed, so that `BinaryHeap` pops the earliest collision first
    fn cmp(&self, other: &Self) -> Ordering {
        other.x[0]
            .total_cmp(&self.x[0])
            .then(other.seq.cmp(&self.seq))
    }
}

/// A QCD string between a projectile (right end) and a target (left end) nucleon
#[derive(Debug, Clone)]
struct QcdString {
    x_prod: [f64; 4],
    tau_form: f64,
    m_over_sigma: f64,
    proj: usize,
    targ: usize,
    y_i_left: f64,
    y_i_right: f64,
    y_f_left: f64,
    y_f_right: f64,
    eta_left: f64,
    eta_right: f64,
    remnant_left: bool,
    remnant_right: bool,
}

/// What the density calculation needs from a string (`QCD_string_output_arr_`)
#[derive(Debug, Clone)]
struct StringOutput {
    /// Transverse production point and end positions (left = target, right = projectile)
    x_center: f64,
    y_center: f64,
    left_x: f64,
    left_y: f64,
    right_x: f64,
    right_y: f64,
    eta_left: f64,
    eta_right: f64,
    y_left: f64,
    y_right: f64,
    y_i_left: f64,
    y_i_right: f64,
    remnant_left: bool,
    remnant_right: bool,
}

impl QcdString {
    /// `QCDString::evolve_QCD_string`: free streaming for m/sigma > 1e6, otherwise
    /// constant deceleration over tau_form, giving the final rapidities and eta_s
    fn evolve(&mut self) {
        let (t0, z0) = (self.x_prod[0], self.x_prod[3]);
        if self.m_over_sigma > 1e6 {
            self.y_f_left = self.y_i_left;
            self.y_f_right = self.y_i_right;
            self.eta_left = free_streaming_eta(self.tau_form, self.y_i_left, t0, z0);
            self.eta_right = free_streaming_eta(self.tau_form, self.y_i_right, t0, z0);
        } else {
            let y_lrf = (self.y_i_right - self.y_i_left).abs() / 2.0;
            let factor =
                self.tau_form * self.tau_form / (2.0 * self.m_over_sigma * self.m_over_sigma);
            let mut y_loss = (factor + 1.0).acosh();
            if y_loss > y_lrf {
                self.tau_form = self.m_over_sigma * (2.0 * (y_lrf.cosh() - 1.0)).sqrt();
                y_loss = y_lrf;
            }
            self.y_f_left = self.y_i_left + y_loss;
            self.y_f_right = self.y_i_right - y_loss;
            self.eta_left =
                decelerated_eta(-self.m_over_sigma, self.tau_form, self.y_i_left, t0, z0);
            self.eta_right =
                decelerated_eta(self.m_over_sigma, self.tau_form, self.y_i_right, t0, z0);
        }
    }

    fn output(&self, nucleons: &[Nucleon3D], x_o: f64, y_o: f64) -> StringOutput {
        let left = &nucleons[self.targ].x;
        let right = &nucleons[self.proj].x;
        StringOutput {
            x_center: self.x_prod[1] - x_o,
            y_center: self.x_prod[2] - y_o,
            left_x: left[1] - x_o,
            left_y: left[2] - y_o,
            right_x: right[1] - x_o,
            right_y: right[2] - y_o,
            eta_left: self.eta_left,
            eta_right: self.eta_right,
            y_left: self.y_f_left,
            y_right: self.y_f_right,
            y_i_left: self.y_i_left,
            y_i_right: self.y_i_right,
            remnant_left: self.remnant_left,
            remnant_right: self.remnant_right,
        }
    }
}

/// `QCDString::get_freestreaming_eta_f`
fn free_streaming_eta(delta_tau: f64, y_i: f64, t0: f64, z0: f64) -> f64 {
    let cosh2 = y_i.cosh().powi(2);
    let temp = t0 - z0 * y_i.tanh();
    let tau0 = (t0 * t0 - z0 * z0).sqrt();
    let tau_f = tau0 + delta_tau;
    let dt = cosh2 * (-temp + (temp * temp + (tau_f * tau_f - tau0 * tau0) / cosh2).sqrt());
    eta_s(t0 + dt, z0 + dt * y_i.tanh())
}

/// `QCDString::get_constant_decelerate_eta_f`
fn decelerated_eta(m_over_sigma: f64, delta_tau: f64, y_i: f64, t0: f64, z0: f64) -> f64 {
    let (c, s) = (y_i.cosh(), y_i.sinh());
    let temp = delta_tau / (2.0 * m_over_sigma);
    let root = (temp * temp + 1.0).sqrt();
    eta_s(
        t0 + delta_tau * (-temp * s + root * c),
        z0 + delta_tau * (-temp * c + root * s),
    )
}

struct Cascade {
    nucleons: Vec<Nucleon3D>,
    /// Initial collision list, in time order
    initial: Vec<Collision>,
    strings: Vec<QcdString>,
    seq: u64,
}

impl Cascade {
    /// Lorentz-contract both nuclei, place them just apart along z (projectile at
    /// z < 0 moving in +z, target at z > 0) and schedule the collisions of all pairs
    /// in `bc` (`Glauber::make_nuclei` + `make_collision_schedule`)
    fn new(
        g: &Glauber3D,
        nucleus_a: &TGlauNucleus,
        nucleus_b: &TGlauNucleus,
        bc: &[Vec<bool>],
    ) -> Self {
        let gamma = g.ybeam.cosh();
        let make = |n: &crate::nucleon::TGlauNucleon, projectile: bool| Nucleon3D {
            projectile,
            wounded: n.is_wounded(),
            x: [0.0, n.x(), n.y(), n.z() / gamma],
            y: if projectile { g.ybeam } else { -g.ybeam },
            collided_times: 0,
            partners: Vec::new(),
            connections: Vec::new(),
            total_connections: 0,
            remnant_set: false,
        };
        let mut nucleons: Vec<Nucleon3D> = nucleus_a
            .nucleons()
            .iter()
            .map(|n| make(n, true))
            .chain(nucleus_b.nucleons().iter().map(|n| make(n, false)))
            .collect();
        let an = nucleus_a.nucleons().len();
        let z_max_a = nucleons[..an]
            .iter()
            .map(|n| n.x[3])
            .fold(f64::NEG_INFINITY, f64::max);
        let z_min_b = nucleons[an..]
            .iter()
            .map(|n| n.x[3])
            .fold(f64::INFINITY, f64::min);
        for (k, n) in nucleons.iter_mut().enumerate() {
            n.x[3] += if k < an {
                -z_max_a - 1e-15
            } else {
                -z_min_b + 1e-15
            };
        }

        let mut cascade = Self {
            nucleons,
            initial: Vec::new(),
            strings: Vec::new(),
            seq: 0,
        };
        for (j, row) in bc.iter().enumerate() {
            for (i, &hit) in row.iter().enumerate() {
                if hit {
                    let targ = an + i;
                    cascade.nucleons[j].partners.push(targ);
                    cascade.nucleons[targ].partners.push(j);
                    if let Some(c) = cascade.schedule(j, targ) {
                        cascade.initial.push(c);
                    }
                }
            }
        }
        cascade
            .initial
            .sort_by(|a, b| a.x[0].total_cmp(&b.x[0]).then(a.seq.cmp(&b.seq)));
        cascade
    }

    /// The next collision of `proj` and `targ` given their current positions and
    /// velocities, if they are still approaching (`Glauber::create_a_collision_event`)
    fn schedule(&mut self, proj: usize, targ: usize) -> Option<Collision> {
        let (p, t) = (&self.nucleons[proj], &self.nucleons[targ]);
        let (v1, v2) = (p.velocity(), t.velocity());
        let gap = t.x[3] - p.x[3];
        // A pair that has just collided sits at the same z; require a real gap so that
        // rounding in the propagation can't make it collide again right away
        // (3dMCGlauber only requires a positive time).
        if gap <= 1e-10 || v1 <= v2 {
            return None;
        }
        let dt = gap / (v1 - v2);
        let x = [
            p.x[0] + dt,
            (p.x[1] + t.x[1]) / 2.0,
            (p.x[2] + t.x[2]) / 2.0,
            p.x[3] + v1 * dt,
        ];
        self.seq += 1;
        Some(Collision {
            x,
            proj,
            targ,
            proj_collided: p.collided_times,
            targ_collided: t.collided_times,
            n_strings: p.strings_with(targ),
            seq: self.seq,
        })
    }

    /// Decide how many strings each collision produces, until every participant is
    /// connected to at least one string (`Glauber::decide_QCD_strings_production`).
    /// Returns the number of strings decided.
    fn decide_strings<R: Rng>(&mut self, g: &Glauber3D, rng: &mut R) -> u32 {
        let mut order: Vec<usize> = (0..self.initial.len()).collect();
        match g.params.string_production_mode {
            1 => order.shuffle(rng),
            2 => order.reverse(),
            _ => {}
        }
        let s = g.params.shadowing_factor;
        let mut number_of_strings = 0;
        // Each pass connects at least one more participant, so this always ends; the
        // bound only protects against a participant without any scheduled collision.
        for _ in 0..10_000 {
            let mut finished = true;
            for &k in &order {
                let (proj, targ) = (self.initial[k].proj, self.initial[k].targ);
                let cp = self.nucleons[proj].total_connections;
                let ct = self.nucleons[targ].total_connections;
                // `Glauber::decide_produce_string_num` without valence quarks
                let n = if cp == 0 || ct == 0 {
                    1
                } else {
                    let prob = (1.0 - s) * (-s * (cp + ct - 2) as f64).exp();
                    u32::from(rng.random::<f64>() < prob)
                };
                if n > 0 {
                    number_of_strings += n;
                    self.nucleons[proj].connections.push((targ, n));
                    self.nucleons[proj].total_connections += n;
                    self.nucleons[targ].connections.push((proj, n));
                    self.nucleons[targ].total_connections += n;
                    self.initial[k].n_strings = n;
                } else if cp == 0 || ct == 0 {
                    finished = false;
                }
            }
            if finished {
                break;
            }
        }
        number_of_strings
    }

    /// Execute the collisions in time order: propagate all nucleons to each collision,
    /// produce its strings, decelerate the pair and reschedule their other collisions
    /// (`Glauber::perform_string_production`). Returns the executed collisions.
    fn perform_string_production<R: Rng>(&mut self, g: &Glauber3D, rng: &mut R) -> u32 {
        let mut schedule: BinaryHeap<Collision> = std::mem::take(&mut self.initial).into();
        let mut t_current = 0.0;
        let mut executed = 0;
        while let Some(c) = schedule.pop() {
            if c.proj_collided != self.nucleons[c.proj].collided_times
                || c.targ_collided != self.nucleons[c.targ].collided_times
            {
                continue;
            }
            executed += 1;
            let dt = c.x[0] - t_current;
            t_current = c.x[0];
            for n in &mut self.nucleons {
                let v = n.velocity();
                n.x[0] += dt;
                n.x[3] += v * dt;
            }
            if c.n_strings == 0 {
                continue;
            }
            for _ in 0..c.n_strings {
                let y_right = self.nucleons[c.proj].y;
                let y_left = self.nucleons[c.targ].y;
                let y_in_lrf = (y_right - y_left).abs() / 2.0;
                let (tau_form, m_over_sigma, y_loss) = g.tau_form_and_m_over_sigma(y_in_lrf, rng);
                self.strings.push(QcdString {
                    x_prod: c.x,
                    tau_form,
                    m_over_sigma,
                    proj: c.proj,
                    targ: c.targ,
                    y_i_left: y_left,
                    y_i_right: y_right,
                    y_f_left: y_left,
                    y_f_right: y_right,
                    eta_left: 0.0,
                    eta_right: 0.0,
                    remnant_left: false,
                    remnant_right: false,
                });
                self.nucleons[c.proj].y -= y_loss;
                self.nucleons[c.targ].y += y_loss;
            }
            // `Glauber::update_collision_schedule`
            self.nucleons[c.proj].collided_times += 1;
            for partner in self.nucleons[c.proj].partners.clone() {
                if let Some(next) = self.schedule(c.proj, partner) {
                    schedule.push(next);
                }
            }
            self.nucleons[c.targ].collided_times += 1;
            for partner in self.nucleons[c.targ].partners.clone() {
                if let Some(next) = self.schedule(partner, c.targ) {
                    schedule.push(next);
                }
            }
        }
        executed
    }

    /// Evolve the strings, and attach each nucleon's remnant to the last string it is
    /// connected to (end of `Glauber::perform_string_production`)
    fn finalize_strings(&mut self) -> Vec<QcdString> {
        let mut strings = std::mem::take(&mut self.strings);
        for s in &mut strings {
            s.evolve();
        }
        for s in strings.iter_mut().rev() {
            if !self.nucleons[s.proj].remnant_set {
                self.nucleons[s.proj].remnant_set = true;
                s.remnant_right = true;
            }
            if !self.nucleons[s.targ].remnant_set {
                self.nucleons[s.targ].remnant_set = true;
                s.remnant_left = true;
            }
        }
        strings
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn config_defaults_and_toggle() {
        let cfg: RunConfig = parse_ron("()").unwrap();
        assert!(!cfg.three_d);
        let cfg: RunConfig =
            parse_ron("(three_d: true, three_d_params: (roots: 5020.0, max_order: 4))").unwrap();
        assert!(cfg.three_d);
        assert_eq!(cfg.three_d_params.roots, 5020.0);
        assert_eq!(cfg.three_d_params.max_order, 4);
        assert_eq!(cfg.three_d_params.tau_form_mean, 0.5);
        assert!(parse_ron::<RunConfig>("(three_d: true, typo: 1)").is_err());
    }

    #[test]
    fn invalid_parameters_are_rejected() {
        let p = Glauber3DParams {
            rapidity_loss_method: 7,
            ..Default::default()
        };
        assert!(Glauber3D::new(p, 42.0).is_err());
        let p = Glauber3DParams {
            roots: 1.0,
            ..Default::default()
        };
        assert!(Glauber3D::new(p, 42.0).is_err());
    }

    #[test]
    fn hit_probability_integrates_to_sigma_nn() {
        for sigma_nn in [30.0, 42.0, 68.0] {
            let g = Glauber3D::new(Glauber3DParams::default(), sigma_nn).unwrap();
            let db = 0.001;
            let integral: f64 = (0..10_000)
                .map(|i| {
                    let b = (i as f64 + 0.5) * db;
                    2.0 * PI * b * db * g.hit_probability(b * b)
                })
                .sum();
            let rel = integral / (sigma_nn * MB_TO_FM2) - 1.0;
            assert!(
                rel.abs() < 1e-3,
                "sigma_NN = {sigma_nn}: relative error {rel}"
            );
        }
    }

    #[test]
    fn sigma_nn_parameterization() {
        // values of 3dMCGlauber's formula
        assert!((sigma_nn_inel(200.0) - 40.834).abs() < 1e-3);
        assert!((sigma_nn_inel(5020.0) - 67.570).abs() < 1e-3);
    }

    #[test]
    fn rapidity_loss_stays_in_range() {
        let g = Glauber3D::new(Glauber3DParams::default(), 42.0).unwrap();
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        for y in [0.5, 2.0, 5.36, 7.0, 9.0] {
            for _ in 0..1000 {
                let loss = g.sample_rapidity_loss(y, &mut rng);
                assert!((0.0..=y).contains(&loss), "y_init = {y}, y_loss = {loss}");
            }
        }
    }

    #[test]
    fn decelerated_string_ends_move_apart() {
        let mut s = QcdString {
            x_prod: [1.0, 0.0, 0.0, 0.2],
            tau_form: 0.5,
            m_over_sigma: 0.3,
            proj: 0,
            targ: 1,
            y_i_left: -5.0,
            y_i_right: 5.0,
            y_f_left: 0.0,
            y_f_right: 0.0,
            eta_left: 0.0,
            eta_right: 0.0,
            remnant_left: false,
            remnant_right: false,
        };
        s.evolve();
        assert!(s.y_f_left > -5.0 && s.y_f_right < 5.0);
        assert!(s.eta_left < s.eta_right);
    }

    #[test]
    fn eccentricity_of_an_elliptic_gaussian() {
        // density exp(-x^2/(2 sx^2) - y^2/(2 sy^2)) has eps_2 = (sy^2 - sx^2)/(sy^2 + sx^2)
        let xg = grid(200, 0.1);
        let (sx, sy) = (1.5f64, 2.5f64);
        let density: Vec<f64> = (0..xg.len() * xg.len())
            .map(|idx| {
                let (x, y) = (xg[idx / xg.len()], xg[idx % xg.len()]);
                (-x * x / (2.0 * sx * sx) - y * y / (2.0 * sy * sy)).exp()
            })
            .collect();
        let (ecc, psi) = eccentricities(&xg, &density, 3);
        let expected = (sy * sy - sx * sx) / (sy * sy + sx * sx);
        assert!(
            (ecc[1] as f64 - expected).abs() < 1e-3,
            "eps_2 = {}",
            ecc[1]
        );
        // not exactly 0: like 3dMCGlauber's, the grid runs from -10 to +9.9 fm
        assert!(ecc[2].abs() < 1e-3, "eps_3 = {}", ecc[2]);
        // short axis along x: Psi_2 = 0 (mod pi)
        let p = psi[1] as f64 % PI;
        assert!(p.min(PI - p) < 1e-3, "Psi_2 = {}", psi[1]);
    }
}
