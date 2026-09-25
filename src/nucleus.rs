// src/nucleus.rs
use crate::constants::{PI, TWO_PI};
use crate::nucleon::{NucleonType, TGlauNucleon};
use rand::Rng;
use rand::RngExt;
use serde::Deserialize;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

/// Nucleus configurations, embedded at compile time (see the file for the format).
const NUCLEI_RON: &str = include_str!("../data/nuclei.ron");

/// Errors from looking up a nucleus configuration
#[derive(Debug, thiserror::Error)]
pub enum NucleusError {
    #[error("unknown nucleus '{0}': no configuration with this name in data/nuclei.ron")]
    Unknown(String),
    #[error("failed to parse data/nuclei.ron: {0}")]
    Parse(String),
    #[error("nucleus '{name}': {msg}")]
    Config { name: String, msg: String },
    #[error("failed to read nucleon configurations from '{path}': {msg}")]
    ConfigFile { path: String, msg: String },
}

/// One entry of `data/nuclei.ron`
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct NucleusConfig {
    n: i32,
    z: i32,
    profile: DensityProfile,
    #[serde(default)]
    r: f64,
    #[serde(default)]
    a: f64,
    #[serde(default)]
    w: f64,
    #[serde(default)]
    r2: f64,
    #[serde(default)]
    a2: f64,
    #[serde(default)]
    w2: f64,
    #[serde(default)]
    beta2: f64,
    #[serde(default)]
    beta3: f64,
    #[serde(default)]
    beta4: f64,
    #[serde(default)]
    gamma: f64,
    #[serde(default = "default_max_r")]
    max_r: f64,
    #[serde(default = "default_recenter")]
    recenter: i32,
    #[serde(default = "default_smax")]
    smax: f64,
    #[serde(default)]
    r0: f64,
    #[serde(default)]
    r1: f64,
    #[serde(default)]
    r2_factor: f64,
    /// Nucleon configuration file for `profile: FromFile` (see `load_configurations`)
    #[serde(default)]
    file: Option<String>,
}

fn default_max_r() -> f64 {
    15.0
}
fn default_recenter() -> i32 {
    1
}
fn default_smax() -> f64 {
    99.0
}

/// Parse RON with `implicit_some`, so optional fields are written `file: "..."`
/// rather than `file: Some("...")`.
fn parse_ron<T: serde::de::DeserializeOwned>(text: &str) -> Result<T, ron::error::SpannedError> {
    ron::Options::default()
        .with_default_extension(ron::extensions::Extensions::IMPLICIT_SOME)
        .from_str(text)
}

/// Parsed nucleus table, parsed once on first use and shared by all threads
/// (the parallel path constructs fresh nuclei for every event).
fn nucleus_table() -> Result<&'static HashMap<String, NucleusConfig>, NucleusError> {
    static TABLE: OnceLock<Result<HashMap<String, NucleusConfig>, String>> = OnceLock::new();
    TABLE
        .get_or_init(|| parse_ron(NUCLEI_RON).map_err(|e| e.to_string()))
        .as_ref()
        .map_err(|e| NucleusError::Parse(e.clone()))
}

/// One nucleon read from a configuration file: position in fm, and whether it is a
/// proton (`None` if the file has no isospin column).
#[derive(Debug, Clone, Copy)]
struct FileNucleon {
    x: f64,
    y: f64,
    z: f64,
    is_proton: Option<bool>,
}

/// Nucleon configurations of one nucleus, one inner `Vec` (of length A) per configuration
type Configurations = Arc<Vec<Vec<FileNucleon>>>;

/// Read the nucleon configurations for a `FromFile` nucleus with `n` nucleons.
///
/// Format: one configuration per line, `x y z` (fm) for each of the `n` nucleons, i.e.
/// 3n numbers, or `x y z isospin` (4n numbers) with isospin 1 = proton, 0 = neutron.
/// Blank lines and lines starting with `#` are skipped. Relative paths are resolved
/// against the current working directory.
///
/// Files are cached per path, since the parallel path constructs fresh nuclei per event.
fn load_configurations(path: &str, n: i32) -> Result<Configurations, NucleusError> {
    static CACHE: OnceLock<Mutex<HashMap<(String, i32), Configurations>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let key = (path.to_string(), n);
    if let Some(configs) = cache.lock().unwrap().get(&key) {
        return Ok(configs.clone());
    }

    let err = |msg: String| NucleusError::ConfigFile { path: path.to_string(), msg };
    let text = std::fs::read_to_string(path).map_err(|e| err(e.to_string()))?;
    let n = n as usize;
    let mut configs = Vec::new();
    for (i, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let values = line
            .split_whitespace()
            .map(str::parse::<f64>)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| err(format!("line {}: {e}", i + 1)))?;
        let stride = match values.len() {
            len if len == 3 * n => 3,
            len if len == 4 * n => 4,
            len => {
                return Err(err(format!(
                    "line {}: expected {} (x y z) or {} (x y z isospin) numbers for A={n}, found {len}",
                    i + 1,
                    3 * n,
                    4 * n
                )));
            }
        };
        let nucleons = values
            .chunks_exact(stride)
            .map(|c| FileNucleon {
                x: c[0],
                y: c[1],
                z: c[2],
                is_proton: (stride == 4).then(|| c[3] > 0.5),
            })
            .collect();
        configs.push(nucleons);
    }
    if configs.is_empty() {
        return Err(err("no configurations found".to_string()));
    }

    let configs = Arc::new(configs);
    cache.lock().unwrap().insert(key, configs.clone());
    Ok(configs)
}

/// Lookup nucleus parameters by name in `data/nuclei.ron`
fn lookup(name: &str) -> Result<&'static NucleusConfig, NucleusError> {
    nucleus_table()?
        .get(name)
        .ok_or_else(|| NucleusError::Unknown(name.to_string()))
}

/// Nuclear density profile type
#[derive(Debug, Clone, Copy, PartialEq, Deserialize)]
pub enum DensityProfile {
    ProtonExp,
    WoodsSaxon3PF,
    WoodsSaxon3PG,
    Hulthen,
    HulthenConstrained,
    Ellipsoid,
    FromFile,
    DeformedBox,
    DeformedTF2,
    ProtonGaussian,
    ProtonDGaussian,
    ProtonNeutron3PF,
    Reweighted,
    ProtonNeutronReweighted,
    DeformedReweighted,
    HarmonicOscillator,
    Oxygen1970,
    FromGraph,
}

/// Represents a nucleus in the Glauber model
#[derive(Debug, Clone)]
pub struct TGlauNucleus {
    name: String,
    n: i32,
    z: i32,
    r: f64,
    a: f64,
    w: f64,
    r2: f64,
    a2: f64,
    w2: f64,
    beta2: f64,
    beta3: f64,
    beta4: f64,
    gamma: f64,
    min_dist: f64,
    node_dist: f64,
    smearing: f64,
    recenter: i32,
    lattice: i32,
    smax: f64,
    profile_type: DensityProfile,
    trials: i32,
    non_smeared: i32,
    weight: f64,
    nucleons: Vec<TGlauNucleon>,
    phi_rot: f64,
    theta_rot: f64,
    x_rot: f64,
    y_rot: f64,
    z_rot: f64,
    max_r: f64,
    // For reweighted profiles
    r0: f64,
    r1: f64,
    r2_factor: f64,
    // Cached rejection-sampling envelopes (max of rho(r)*r^2 over [0, max_r]), -1.0 = uncached
    density_env_p: f64,
    density_env_n: f64,
    density_env_deformed: f64,
    // Nucleon configurations for the FromFile profile
    configurations: Option<Configurations>,
}

impl TGlauNucleus {
    pub fn new(name: &str) -> Result<Self, NucleusError> {
        Self::from_config(name, lookup(name)?)
    }

    fn from_config(name: &str, cfg: &NucleusConfig) -> Result<Self, NucleusError> {
        let config_err = |msg: &str| NucleusError::Config {
            name: name.to_string(),
            msg: msg.to_string(),
        };
        let configurations = match (cfg.profile, &cfg.file) {
            (DensityProfile::FromFile, Some(path)) => Some(load_configurations(path, cfg.n)?),
            (DensityProfile::FromFile, None) => {
                return Err(config_err("profile FromFile requires a `file` parameter"));
            }
            (_, Some(_)) => {
                return Err(config_err("`file` is only used with profile FromFile"));
            }
            (_, None) => None,
        };
        Ok(Self {
            name: name.to_string(),
            n: cfg.n,
            z: cfg.z,
            r: cfg.r,
            a: cfg.a,
            w: cfg.w,
            r2: cfg.r2,
            a2: cfg.a2,
            w2: cfg.w2,
            beta2: cfg.beta2,
            beta3: cfg.beta3,
            beta4: cfg.beta4,
            gamma: cfg.gamma,
            min_dist: 0.4,
            node_dist: -1.0,
            smearing: 0.0,
            recenter: cfg.recenter,
            lattice: 0,
            smax: cfg.smax,
            profile_type: cfg.profile,
            trials: 0,
            non_smeared: 0,
            weight: 1.0,
            nucleons: Vec::new(),
            phi_rot: 0.0,
            theta_rot: 0.0,
            x_rot: 0.0,
            y_rot: 0.0,
            z_rot: 0.0,
            max_r: cfg.max_r,
            r0: cfg.r0,
            r1: cfg.r1,
            r2_factor: cfg.r2_factor,
            density_env_p: -1.0,
            density_env_n: -1.0,
            density_env_deformed: -1.0,
            configurations,
        })
    }

    /// Pick one nucleon configuration at random (FromFile profile only)
    fn random_configuration<R: Rng>(&self, rng: &mut R) -> Option<&[FileNucleon]> {
        let configs = self.configurations.as_ref()?;
        Some(&configs[rng.random_range(0..configs.len())])
    }

    /// Allocate nucleons for the nucleus
    fn allocate_nucleons(&mut self) {
        if self.n <= 0 {
            return;
        }
        self.nucleons.clear();
        self.nucleons.reserve(self.n as usize);
        for i in 0..self.n {
            let mut nucleon = TGlauNucleon::new();
            if i < self.z {
                nucleon.set_type(NucleonType::Proton);
            }
            self.nucleons.push(nucleon);
        }
    }

    /// Randomize the types of nucleons
    fn randomize_nucleons<R: Rng>(&mut self, rng: &mut R) {
        let mut iz = 0;
        for i in 0..self.n as usize {
            let frac = (self.z - iz) as f64 / (self.n - i as i32) as f64;
            let rn: f64 = rng.random();
            if rn < frac {
                self.nucleons[i].set_type(NucleonType::Proton);
                iz += 1;
            } else {
                self.nucleons[i].set_type(NucleonType::Neutron);
            }
        }
    }

    /// Test if a nucleon is within the minimum distance of existing nucleons
    #[allow(dead_code)]
    fn test_min_dist(&self, n: usize, x: f64, y: f64, z: f64) -> bool {
        if self.min_dist <= 0.0 {
            return true;
        }
        let md2 = self.min_dist * self.min_dist;
        for j in 0..n {
            let other = &self.nucleons[j];
            let dx = x - other.x();
            let dy = y - other.y();
            let dz = z - other.z();
            if dx * dx + dy * dy + dz * dz < md2 {
                return false;
            }
        }
        true
    }

    /// Woods-Saxon 3-parameter Fermi density (unnormalized), matching McGlauber fF=1 (3pF):
    /// rho(r) = (1 + w*(r/R)^2) / (1 + exp((r-R)/a))
    fn woods_saxon_3pf(r: f64, r_param: f64, a_param: f64, w_param: f64) -> f64 {
        let w_term = 1.0 + w_param * (r / r_param).powi(2);
        let denom = 1.0 + ((r - r_param) / a_param).exp();
        w_term / denom
    }

    /// Woods-Saxon 3-parameter Gaussian density (unnormalized), matching McGlauber fF=2 (3pG):
    /// rho(r) = (1 + w*(r/R)^2) / (1 + exp((r^2-R^2)/a^2))
    fn woods_saxon_3pg(r: f64, r_param: f64, a_param: f64, w_param: f64) -> f64 {
        let w_term = 1.0 + w_param * (r / r_param).powi(2);
        let denom = 1.0 + ((r * r - r_param * r_param) / (a_param * a_param)).exp();
        w_term / denom
    }

    /// Hulthen deuteron density (unnormalized), matching McGlauber fF=3/4:
    /// rho(r) = R*a*(R+a) / (2*pi*(R-a)^2) * ((exp(-R*r)-exp(-a*r))/r)^2
    fn hulthen_density(r: f64, r_param: f64, a_param: f64) -> f64 {
        if r <= 0.0 {
            return 0.0;
        }
        let diff = (-r_param * r).exp() - (-a_param * r).exp();
        let norm = r_param * a_param * (r_param + a_param) / (2.0 * PI * (r_param - a_param).powi(2));
        norm * (diff / r).powi(2)
    }

    /// Harmonic oscillator density (unnormalized), matching McGlauber fF=15:
    /// rho(r) = (1 + coeff*(r/scale)^2) * exp(-(r/scale)^2)
    /// Note: McGlauber sets par[0]=fR as the coefficient and par[1]=fA as the length scale.
    fn harmonic_oscillator_density(r: f64, coeff: f64, scale: f64) -> f64 {
        let x = r / scale;
        (1.0 + coeff * x * x) * (-x * x).exp()
    }

    /// Oxygen-1970 parameterization (unnormalized), matching McGlauber fF=16:
    /// rho(r) = (1 - 0.102*sin(2.76r)*exp(-(0.35r)^2)/(2.76r) + w*(r/R)^2) / (1 + exp((r-R)/a))
    fn oxygen_1970_density(r: f64, r_param: f64, a_param: f64, w_param: f64) -> f64 {
        let osc = if r.abs() < 1e-12 {
            0.102
        } else {
            0.102 * (2.76 * r).sin() / (2.76 * r) * (-(0.35 * r).powi(2)).exp()
        };
        let numerator = 1.0 - osc + w_param * (r / r_param).powi(2);
        let denom = 1.0 + ((r - r_param) / a_param).exp();
        numerator / denom
    }

    /// Double-Gaussian proton density (unnormalized), matching McGlauber fF=10:
    /// rho(r) = (1-p0)/R^3 * exp(-(r/R)^2) + p0/(0.4R)^3 * exp(-(r/(0.4R))^2)
    fn proton_dgaussian_density(r: f64, r_param: f64) -> f64 {
        const P0: f64 = 0.5;
        let scale2 = 0.4 * r_param;
        (1.0 - P0) / r_param.powi(3) * (-(r / r_param).powi(2)).exp()
            + P0 / scale2.powi(3) * (-(r / scale2).powi(2)).exp()
    }

    /// Radial density rho(r) (unnormalized) matching TGlauNucleus::fF in McGlauber's
    /// runglauber_v3.3.c. The nucleon radial probability distribution is proportional
    /// to rho(r) * r^2, over r in [0, max_r].
    fn radial_density(&self, r: f64, is_proton: bool) -> f64 {
        match self.profile_type {
            DensityProfile::ProtonExp => (-r / self.r).exp(),
            DensityProfile::ProtonGaussian => (-(r * r) / (2.0 * self.r * self.r)).exp(),
            DensityProfile::ProtonDGaussian => Self::proton_dgaussian_density(r, self.r),
            DensityProfile::Hulthen | DensityProfile::HulthenConstrained => {
                Self::hulthen_density(r, self.r, self.a)
            }
            DensityProfile::WoodsSaxon3PF => Self::woods_saxon_3pf(r, self.r, self.a, self.w),
            DensityProfile::WoodsSaxon3PG => Self::woods_saxon_3pg(r, self.r, self.a, self.w),
            DensityProfile::HarmonicOscillator => {
                Self::harmonic_oscillator_density(r, self.r, self.a)
            }
            DensityProfile::Oxygen1970 => Self::oxygen_1970_density(r, self.r, self.a, self.w),
            DensityProfile::ProtonNeutron3PF => {
                let (r_param, a_param, w_param) = if is_proton {
                    (self.r, self.a, self.w)
                } else {
                    (self.r2, self.a2, self.w2)
                };
                Self::woods_saxon_3pf(r, r_param, a_param, w_param)
            }
            DensityProfile::Reweighted | DensityProfile::DeformedReweighted => {
                let denom = self.r0 + self.r1 * r + self.r2_factor * r * r;
                Self::woods_saxon_3pf(r, self.r, self.a, self.w) / denom
            }
            DensityProfile::ProtonNeutronReweighted => {
                let (r_param, a_param, w_param) = if is_proton {
                    (self.r, self.a, self.w)
                } else {
                    (self.r2, self.a2, self.w2)
                };
                let denom = self.r0 + self.r1 * r + self.r2_factor * r * r;
                Self::woods_saxon_3pf(r, r_param, a_param, w_param) / denom
            }
            _ => Self::woods_saxon_3pf(r, self.r, self.a, self.w),
        }
    }

    /// Compute (and cache) the rejection-sampling envelope: the maximum of
    /// rho(r) * r^2 over [0, max_r], with a small safety margin.
    fn density_envelope(&mut self, is_proton: bool) -> f64 {
        let cached = if is_proton {
            self.density_env_p
        } else {
            self.density_env_n
        };
        if cached >= 0.0 {
            return cached;
        }
        const GRID: usize = 512;
        let mut peak = 0.0f64;
        if self.max_r > 0.0 {
            for i in 0..=GRID {
                let r = self.max_r * i as f64 / GRID as f64;
                let weight = self.radial_density(r, is_proton) * r * r;
                if weight.is_finite() && weight > peak {
                    peak = weight;
                }
            }
        }
        let envelope = if peak > 0.0 { peak * 1.05 } else { 0.0 };
        if is_proton {
            self.density_env_p = envelope;
        } else {
            self.density_env_n = envelope;
        }
        envelope
    }

    /// Sample a radius from the nuclear density profile using rejection sampling,
    /// mirroring TGlauNucleus::ThrowNucleons (via TF1::GetRandom) in McGlauber.
    fn sample_radius<R: Rng>(&mut self, rng: &mut R, is_proton: bool) -> f64 {
        if self.max_r <= 0.0 {
            return 0.0;
        }
        let envelope = self.density_envelope(is_proton);
        if envelope <= 0.0 {
            return self.max_r * rng.random::<f64>();
        }
        const MAX_TRIALS: usize = 100_000;
        for _ in 0..MAX_TRIALS {
            let r = rng.random::<f64>() * self.max_r;
            let u = rng.random::<f64>() * envelope;
            if u < self.radial_density(r, is_proton) * r * r {
                return r;
            }
        }
        self.max_r * rng.random::<f64>()
    }

    /// Deformed nuclear surface radius R(theta), matching McGlauber's TF2 formula for fF=8/14:
    /// R(theta) = R*(1 + beta2*0.315*(3cos^2(theta)-1) + beta4*0.105*(35cos^4(theta)-30cos^2(theta)+3))
    fn deformed_r_theta(&self, theta: f64) -> f64 {
        let ct = theta.cos();
        self.r
            * (1.0
                + self.beta2 * 0.315 * (3.0 * ct * ct - 1.0)
                + self.beta4 * 0.105 * (35.0 * ct.powi(4) - 30.0 * ct * ct + 3.0))
    }

    /// Unnormalized joint (r, theta) density r^2*sin(theta)/(1+exp((r-R(theta))/a)), optionally
    /// divided by the radial reweighting polynomial, matching McGlauber fF=8/14.
    fn deformed_weight(&self, r: f64, theta: f64) -> f64 {
        let r_theta = self.deformed_r_theta(theta);
        let mut w = r * r * theta.sin() / (1.0 + ((r - r_theta) / self.a).exp());
        if self.profile_type == DensityProfile::DeformedReweighted {
            let denom = self.r0 + self.r1 * r + self.r2_factor * r * r;
            w /= denom;
        }
        w
    }

    /// Compute (and cache) the 2D rejection-sampling envelope for the deformed TF2 profile.
    fn deformed_envelope(&mut self) -> f64 {
        if self.density_env_deformed >= 0.0 {
            return self.density_env_deformed;
        }
        const GRID_R: usize = 200;
        const GRID_T: usize = 200;
        let mut peak = 0.0f64;
        if self.max_r > 0.0 {
            for i in 0..=GRID_R {
                let r = self.max_r * i as f64 / GRID_R as f64;
                for j in 0..=GRID_T {
                    let theta = PI * j as f64 / GRID_T as f64;
                    let w = self.deformed_weight(r, theta);
                    if w.is_finite() && w > peak {
                        peak = w;
                    }
                }
            }
        }
        let envelope = if peak > 0.0 { peak * 1.05 } else { 0.0 };
        self.density_env_deformed = envelope;
        envelope
    }

    /// Generate spherical coordinates for a nucleon - returns (r, phi, theta) with theta as polar angle.
    /// cos(theta) is sampled uniformly in [-1, 1] so the angular distribution is isotropic,
    /// matching McGlauber's `ctheta = 2*Rndm()-1; theta = ACos(ctheta)`.
    fn generate_spherical_coordinates<R: Rng>(
        &mut self,
        rng: &mut R,
        is_proton: bool,
    ) -> (f64, f64, f64) {
        let r = self.sample_radius(rng, is_proton);
        let phi = rng.random::<f64>() * TWO_PI;
        let ctheta = 2.0 * rng.random::<f64>() - 1.0;
        let theta = ctheta.acos();
        (r, phi, theta)
    }

    /// Throw nucleons according to the density profile
    pub fn throw_nucleons<R: Rng>(&mut self, xshift: f64, rng: &mut R) -> [f64; 3] {
        self.allocate_nucleons();
        self.randomize_nucleons(rng);

        self.trials = 0;
        self.non_smeared = 0;
        self.phi_rot = rng.random::<f64>() * TWO_PI;
        let cos_theta = 2.0 * rng.random::<f64>() - 1.0;
        self.theta_rot = cos_theta.acos();
        self.x_rot = rng.random::<f64>() * TWO_PI;
        self.y_rot = rng.random::<f64>() * TWO_PI;
        self.z_rot = rng.random::<f64>() * TWO_PI;

        let is_hulthen = matches!(
            self.profile_type,
            DensityProfile::Hulthen | DensityProfile::HulthenConstrained
        );

        // Store nucleon positions temporarily
        let mut positions: Vec<(f64, f64, f64)> = Vec::with_capacity(self.n as usize);

        // Configurations read from a file: pick one at random
        if let Some(configs) = self.configurations.clone() {
            let config = &configs[rng.random_range(0..configs.len())];
            for (nucleon, fnuc) in self.nucleons.iter_mut().zip(config) {
                if let Some(is_proton) = fnuc.is_proton {
                    nucleon.set_type(if is_proton {
                        NucleonType::Proton
                    } else {
                        NucleonType::Neutron
                    });
                }
            }
            positions.extend(config.iter().map(|f| (f.x, f.y, f.z)));
            self.trials = 1;
        }
        // Special handling for Hulthen (deuteron)
        else if is_hulthen {
            let r = self.sample_radius(rng, true) / 2.0;
            let phi = rng.random::<f64>() * TWO_PI;
            let ctheta = 2.0 * rng.random::<f64>() - 1.0;
            let stheta = (1.0 - ctheta * ctheta).sqrt();

            let x1 = r * stheta * phi.cos();
            let y1 = r * stheta * phi.sin();
            let z1 = r * ctheta;
            positions.push((x1, y1, z1));

            if matches!(self.profile_type, DensityProfile::HulthenConstrained) {
                positions.push((-x1, -y1, -z1));
            } else {
                let r2 = self.sample_radius(rng, true) / 2.0;
                let phi2 = rng.random::<f64>() * TWO_PI;
                let ctheta2 = 2.0 * rng.random::<f64>() - 1.0;
                let stheta2 = (1.0 - ctheta2 * ctheta2).sqrt();
                positions.push((
                    r2 * stheta2 * phi2.cos(),
                    r2 * stheta2 * phi2.sin(),
                    r2 * ctheta2,
                ));
            }
            self.trials = 1;
        }
        // Deformed nuclei with box method (Uranium-like)
        else if matches!(
            self.profile_type,
            DensityProfile::Ellipsoid | DensityProfile::DeformedBox
        ) {
            for _i in 0..self.n as usize {
                let mut placed = false;
                while !placed {
                    let x = self.max_r * (2.0 * rng.random::<f64>() - 1.0);
                    let y = self.max_r * (2.0 * rng.random::<f64>() - 1.0);
                    let z = self.max_r * (2.0 * rng.random::<f64>() - 1.0);
                    let r = (x * x + y * y + z * z).sqrt();
                    let theta = (z / r).acos();

                    let r_theta = if self.profile_type == DensityProfile::Ellipsoid {
                        self.r + self.beta2 * theta.cos().powi(2)
                    } else {
                        // DeformedBox with beta2, beta3, beta4
                        let mut r_def = self.r;
                        if self.beta2 != 0.0 {
                            r_def +=
                                self.r * self.beta2 * (3.0 * theta.cos().powi(2) - 1.0) * 0.315;
                        }
                        if self.beta3 != 0.0 {
                            // sph_legendre(3,0,theta) = sqrt(7/(4*pi)) * (5cos^3(theta)-3cos(theta))/2
                            r_def += self.r
                                * self.beta3
                                * (5.0 * theta.cos().powi(3) - 3.0 * theta.cos())
                                * 0.373176;
                        }
                        if self.beta4 != 0.0 {
                            r_def += self.r
                                * self.beta4
                                * (35.0 * theta.cos().powi(4) - 30.0 * theta.cos().powi(2) + 3.0)
                                * 0.105;
                        }
                        r_def
                    };

                    let prob = (1.0 + self.w * (r / r_theta).powi(2))
                        / (1.0 + ((r - r_theta) / self.a).exp());
                    if rng.random::<f64>() < prob {
                        positions.push((x, y, z));
                        placed = true;
                    }
                    self.trials += 1;
                }
            }
        }
        // DeformedTF2 (Al, Cu2, Xe2, etc.): joint 2D rejection sampling in (r, theta),
        // matching McGlauber's TF2 GetRandom2 for fF=8/14.
        else if matches!(
            self.profile_type,
            DensityProfile::DeformedTF2 | DensityProfile::DeformedReweighted
        ) {
            let envelope = self.deformed_envelope();
            for _i in 0..self.n as usize {
                loop {
                    self.trials += 1;
                    let r = rng.random::<f64>() * self.max_r;
                    let theta = rng.random::<f64>() * PI;
                    let accept = envelope <= 0.0
                        || rng.random::<f64>() * envelope < self.deformed_weight(r, theta);
                    if accept {
                        let phi = rng.random::<f64>() * TWO_PI;
                        let x = r * theta.sin() * phi.cos();
                        let y = r * theta.sin() * phi.sin();
                        let z = r * theta.cos();
                        positions.push((x, y, z));
                        break;
                    }
                }
            }
        }
        // Standard spherical nuclei
        else {
            for i in 0..self.n as usize {
                let is_proton = self.nucleons[i].is_proton();
                let (r, phi, theta) = self.generate_spherical_coordinates(rng, is_proton);
                let x = r * theta.sin() * phi.cos();
                let y = r * theta.sin() * phi.sin();
                let z = r * theta.cos();
                positions.push((x, y, z));
                self.trials += 1;
            }
        }

        // Set positions in nucleons
        for (i, (x, y, z)) in positions.into_iter().enumerate() {
            if i < self.nucleons.len() {
                self.nucleons[i].set_position(x, y, z);
                // Apply rotation for deformed nuclei
                if matches!(
                    self.profile_type,
                    DensityProfile::Ellipsoid
                        | DensityProfile::DeformedBox
                        | DensityProfile::DeformedTF2
                        | DensityProfile::DeformedReweighted
                ) {
                    self.nucleons[i].rotate_2d(self.phi_rot, self.theta_rot);
                }
                // Isotropic random orientation for configurations read from a file:
                // uniform spin about z, then the z-axis carried onto a uniform direction
                if self.configurations.is_some() {
                    self.nucleons[i].rotate_3d(0.0, 0.0, self.z_rot);
                    self.nucleons[i].rotate_2d(self.phi_rot, self.theta_rot);
                }
            }
        }

        // Calculate center of mass
        let mut sumx = 0.0;
        let mut sumy = 0.0;
        let mut sumz = 0.0;
        for nucleon in &self.nucleons {
            sumx += nucleon.x();
            sumy += nucleon.y();
            sumz += nucleon.z();
        }
        sumx /= self.n as f64;
        sumy /= self.n as f64;
        sumz /= self.n as f64;

        let shift_mag = (sumx * sumx + sumy * sumy + sumz * sumz).sqrt();
        if shift_mag > self.smax {
            return self.throw_nucleons(xshift, rng);
        }

        // Recenter
        let mut fsumx = 0.0;
        let mut fsumy = 0.0;
        let mut fsumz = 0.0;

        match self.recenter {
            1 => {
                fsumx = sumx;
                fsumy = sumy;
                fsumz = sumz;
            }
            2 => {
                if let Some(last) = self.nucleons.last_mut() {
                    let x = last.x() - self.n as f64 * sumx;
                    let y = last.y() - self.n as f64 * sumy;
                    let z = last.z() - self.n as f64 * sumz;
                    last.set_position(x, y, z);
                }
            }
            3 | 4 => {
                if shift_mag > 1e-3 {
                    let shift_vec = [sumx, sumy, sumz];
                    let z_vec = [0.0, 0.0, 1.0];
                    let cross_x = shift_vec[1] * z_vec[2] - shift_vec[2] * z_vec[1];
                    let cross_y = shift_vec[2] * z_vec[0] - shift_vec[0] * z_vec[2];
                    let cross_z = shift_vec[0] * z_vec[1] - shift_vec[1] * z_vec[0];
                    let cross_mag =
                        (cross_x * cross_x + cross_y * cross_y + cross_z * cross_z).sqrt();

                    if cross_mag > 1e-10 {
                        let angle = (shift_mag).acos();
                        // Simplified rotation: rotate around the cross product axis
                        for nucleon in &mut self.nucleons {
                            let x = nucleon.x();
                            let y = nucleon.y();
                            let z = nucleon.z();
                            // Apply rotation (simplified)
                            let nx = cross_x * (cross_x * x + cross_y * y + cross_z * z)
                                / (cross_mag * cross_mag)
                                * (1.0 - angle.cos())
                                + x * angle.cos()
                                + (-cross_z * y + cross_y * z) / cross_mag * angle.sin();
                            let ny = cross_y * (cross_x * x + cross_y * y + cross_z * z)
                                / (cross_mag * cross_mag)
                                * (1.0 - angle.cos())
                                + y * angle.cos()
                                + (cross_z * x - cross_x * z) / cross_mag * angle.sin();
                            let nz = cross_z * (cross_x * x + cross_y * y + cross_z * z)
                                / (cross_mag * cross_mag)
                                * (1.0 - angle.cos())
                                + z * angle.cos()
                                + (-cross_y * x + cross_x * y) / cross_mag * angle.sin();
                            nucleon.set_position(nx, ny, nz);
                        }
                        if self.recenter == 3 {
                            fsumz = shift_mag;
                        }
                    }
                }
            }
            5 => {
                fsumx = sumx;
                fsumy = sumy;
            }
            _ => {}
        }

        // Apply shift
        for nucleon in &mut self.nucleons {
            nucleon.set_position(
                nucleon.x() - fsumx + xshift,
                nucleon.y() - fsumy,
                nucleon.z() - fsumz,
            );
        }

        // Return center of mass
        let mut cmx = 0.0;
        let mut cmy = 0.0;
        let mut cmz = 0.0;
        for nucleon in &self.nucleons {
            cmx += nucleon.x();
            cmy += nucleon.y();
            cmz += nucleon.z();
        }
        [
            cmx / self.n as f64,
            cmy / self.n as f64,
            cmz / self.n as f64,
        ]
    }

    /// Sample a single nucleon's distance from the nucleus center directly from the
    /// nuclear density profile rho(r) via rejection sampling (see `radial_density`),
    /// without the nucleus-level recentering, minimum-distance rejection, or rotation
    /// that `throw_nucleons` applies. This reflects the radial shape of the density
    /// profile (matching McGlauber's `fFunc1->GetRandom()`), which is what you want
    /// when visualizing/validating the density function itself rather than a fully
    /// assembled nucleus.
    ///
    /// For the Hulthen deuteron profile the density is sampled over the relative
    /// proton-neutron coordinate, so (as in `throw_nucleons`) the result is halved
    /// to give the nucleon's distance from the center of mass.
    pub fn sample_nucleon_radius<R: Rng>(&mut self, rng: &mut R, is_proton: bool) -> f64 {
        if self.configurations.is_some() {
            return self.sample_nucleon_direction(rng, is_proton).0;
        }
        let r = self.sample_radius(rng, is_proton);
        if matches!(
            self.profile_type,
            DensityProfile::Hulthen | DensityProfile::HulthenConstrained
        ) {
            r / 2.0
        } else {
            r
        }
    }

    /// Sample a single nucleon's (r, phi, theta) directly from the nuclear density
    /// function - r is the distance from the nucleus center, phi the azimuthal angle,
    /// theta the polar angle (measured from the z-axis) in the nucleus's own intrinsic
    /// frame. This uses exactly the same per-nucleon sampling as `throw_nucleons`
    /// (rejection sampling on the McGlauber `TGlauNucleus::fF` density - 3pF, 3pG,
    /// deformed profiles, etc. - see `radial_density`/`deformed_weight`), but skips the
    /// nucleus-level recentering, minimum-distance rejection, and the random per-event
    /// whole-nucleus orientation (`phi_rot`/`theta_rot`) that `throw_nucleons` applies
    /// afterward. Skipping that final orientation matters: it's what lets a deformed
    /// nucleus's theta-dependence (from beta2/beta3/beta4) actually show up here,
    /// instead of being washed out into an isotropic lab-frame distribution.
    pub fn sample_nucleon_direction<R: Rng>(
        &mut self,
        rng: &mut R,
        is_proton: bool,
    ) -> (f64, f64, f64) {
        // FromFile: a random nucleon of a random configuration (ignoring `is_proton`)
        if let Some(config) = self.random_configuration(rng) {
            let f = config[rng.random_range(0..config.len())];
            let r = (f.x * f.x + f.y * f.y + f.z * f.z).sqrt();
            let theta = if r > 0.0 { (f.z / r).acos() } else { 0.0 };
            return (r, f.y.atan2(f.x), theta);
        }

        let is_hulthen = matches!(
            self.profile_type,
            DensityProfile::Hulthen | DensityProfile::HulthenConstrained
        );
        if is_hulthen {
            let r = self.sample_radius(rng, true) / 2.0;
            let phi = rng.random::<f64>() * TWO_PI;
            let ctheta = 2.0 * rng.random::<f64>() - 1.0;
            return (r, phi, ctheta.acos());
        }

        if matches!(
            self.profile_type,
            DensityProfile::Ellipsoid | DensityProfile::DeformedBox
        ) {
            loop {
                let x = self.max_r * (2.0 * rng.random::<f64>() - 1.0);
                let y = self.max_r * (2.0 * rng.random::<f64>() - 1.0);
                let z = self.max_r * (2.0 * rng.random::<f64>() - 1.0);
                let r = (x * x + y * y + z * z).sqrt();
                if r == 0.0 {
                    continue;
                }
                let theta = (z / r).acos();

                let r_theta = if self.profile_type == DensityProfile::Ellipsoid {
                    self.r + self.beta2 * theta.cos().powi(2)
                } else {
                    let mut r_def = self.r;
                    if self.beta2 != 0.0 {
                        r_def += self.r * self.beta2 * (3.0 * theta.cos().powi(2) - 1.0) * 0.315;
                    }
                    if self.beta3 != 0.0 {
                        r_def += self.r
                            * self.beta3
                            * (5.0 * theta.cos().powi(3) - 3.0 * theta.cos())
                            * 0.373176;
                    }
                    if self.beta4 != 0.0 {
                        r_def += self.r
                            * self.beta4
                            * (35.0 * theta.cos().powi(4) - 30.0 * theta.cos().powi(2) + 3.0)
                            * 0.105;
                    }
                    r_def
                };

                let prob = (1.0 + self.w * (r / r_theta).powi(2))
                    / (1.0 + ((r - r_theta) / self.a).exp());
                if rng.random::<f64>() < prob {
                    let phi = y.atan2(x);
                    return (r, phi, theta);
                }
            }
        }

        if matches!(
            self.profile_type,
            DensityProfile::DeformedTF2 | DensityProfile::DeformedReweighted
        ) {
            let envelope = self.deformed_envelope();
            loop {
                let r = rng.random::<f64>() * self.max_r;
                let theta = rng.random::<f64>() * PI;
                let accept = envelope <= 0.0
                    || rng.random::<f64>() * envelope < self.deformed_weight(r, theta);
                if accept {
                    let phi = rng.random::<f64>() * TWO_PI;
                    return (r, phi, theta);
                }
            }
        }

        self.generate_spherical_coordinates(rng, is_proton)
    }

    pub fn name(&self) -> &str {
        &self.name
    }
    pub fn n(&self) -> i32 {
        self.n
    }
    pub fn z(&self) -> i32 {
        self.z
    }
    pub fn r(&self) -> f64 {
        self.r
    }
    pub fn a(&self) -> f64 {
        self.a
    }
    pub fn w(&self) -> f64 {
        self.w
    }
    pub fn min_dist(&self) -> f64 {
        self.min_dist
    }
    pub fn node_dist(&self) -> f64 {
        self.node_dist
    }
    pub fn recenter(&self) -> i32 {
        self.recenter
    }
    pub fn smax(&self) -> f64 {
        self.smax
    }
    pub fn weight(&self) -> f64 {
        self.weight
    }
    pub fn trials(&self) -> i32 {
        self.trials
    }
    pub fn non_smeared(&self) -> i32 {
        self.non_smeared
    }
    pub fn nucleons(&self) -> &[TGlauNucleon] {
        &self.nucleons
    }
    pub fn nucleons_mut(&mut self) -> &mut [TGlauNucleon] {
        &mut self.nucleons
    }
    pub fn phi_rot(&self) -> f64 {
        self.phi_rot
    }
    pub fn theta_rot(&self) -> f64 {
        self.theta_rot
    }

    pub fn set_min_dist(&mut self, d: f64) {
        self.min_dist = d;
    }
    pub fn set_node_dist(&mut self, d: f64) {
        self.node_dist = d;
    }
    pub fn set_recenter(&mut self, r: i32) {
        self.recenter = r;
    }
    pub fn set_smax(&mut self, s: f64) {
        self.smax = s;
    }
    pub fn set_smearing(&mut self, s: f64) {
        self.smearing = s;
    }
    pub fn set_lattice(&mut self, l: i32) {
        self.lattice = l;
    }
    pub fn set_weight(&mut self, w: f64) {
        self.weight = w;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nucleus_table_parses() {
        assert!(!nucleus_table().unwrap().is_empty());
    }

    #[test]
    fn known_nucleus_uses_ron_values_and_defaults() {
        let pb = TGlauNucleus::new("Pb").unwrap();
        assert_eq!((pb.n, pb.z), (208, 82));
        assert_eq!((pb.r, pb.a, pb.w), (6.62, 0.546, 0.0));
        assert_eq!(pb.max_r, 10.0);
        assert_eq!((pb.recenter, pb.smax), (1, 99.0));
        assert_eq!(pb.profile_type, DensityProfile::WoodsSaxon3PF);
    }

    /// Write `contents` to a unique temporary file and return its path
    fn temp_file(tag: &str, contents: &str) -> String {
        let path = std::env::temp_dir().join(format!(
            "oxy_glauber_test_{}_{tag}.dat",
            std::process::id()
        ));
        std::fs::write(&path, contents).unwrap();
        path.to_str().unwrap().to_string()
    }

    fn from_file_config(n: i32, z: i32, path: &str) -> NucleusConfig {
        parse_ron(&format!("(n: {n}, z: {z}, profile: FromFile, file: {path:?})")).unwrap()
    }

    #[test]
    fn from_file_with_isospin() {
        // Two He4-like configurations; protons are the nucleons with isospin 1
        let path = temp_file(
            "isospin",
            "# x y z isospin\n\
             1 0 0 1  -1 0 0 1  0 1 0 0  0 -1 0 0\n\
             \n\
             0 0 1 1  0 0 -1 0  2 0 0 1  -2 0 0 0\n",
        );
        let cfg = from_file_config(4, 2, &path);
        let mut nucleus = TGlauNucleus::from_config("test", &cfg).unwrap();
        assert_eq!(nucleus.configurations.as_ref().unwrap().len(), 2);

        let mut rng = rand::rng();
        for _ in 0..20 {
            nucleus.throw_nucleons(0.0, &mut rng);
            assert_eq!(nucleus.nucleons().len(), 4);
            assert_eq!(nucleus.nucleons().iter().filter(|n| n.is_proton()).count(), 2);
            // Both configurations are centered, so rotation keeps each nucleon at r=1 or 2
            for nucleon in nucleus.nucleons() {
                let r = (nucleon.x().powi(2) + nucleon.y().powi(2) + nucleon.z().powi(2)).sqrt();
                assert!((r - 1.0).abs() < 1e-9 || (r - 2.0).abs() < 1e-9, "r = {r}");
            }
        }
    }

    #[test]
    fn from_file_without_isospin() {
        let path = temp_file("no_isospin", "0.5 0 0  -0.5 0 0\n");
        let cfg = from_file_config(2, 1, &path);
        let mut nucleus = TGlauNucleus::from_config("test", &cfg).unwrap();
        nucleus.throw_nucleons(0.0, &mut rand::rng());
        assert_eq!(nucleus.nucleons().iter().filter(|n| n.is_proton()).count(), 1);
    }

    #[test]
    fn from_file_errors() {
        let bad = temp_file("bad_count", "1 2 3 4 5\n");
        assert!(matches!(
            TGlauNucleus::from_config("test", &from_file_config(2, 1, &bad)),
            Err(NucleusError::ConfigFile { .. })
        ));
        assert!(matches!(
            TGlauNucleus::from_config("test", &from_file_config(2, 1, "/nonexistent/file.dat")),
            Err(NucleusError::ConfigFile { .. })
        ));
        // Table entries with FromFile but no `file` are rejected when used
        assert!(matches!(TGlauNucleus::new("O"), Err(NucleusError::Config { .. })));
    }

    #[test]
    fn unknown_nucleus_is_an_error() {
        assert!(matches!(
            TGlauNucleus::new("NotANucleus"),
            Err(NucleusError::Unknown(name)) if name == "NotANucleus"
        ));
    }
}
