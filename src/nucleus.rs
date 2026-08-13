// src/nucleus.rs
use crate::constants::{PI, TWO_PI};
use crate::nucleon::{NucleonType, TGlauNucleon};
use rand::Rng;
use rand::RngExt;

/// Nuclear density profile type
#[derive(Debug, Clone, Copy, PartialEq)]
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
    Trajectum,
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
}

impl TGlauNucleus {
    pub fn new(name: &str) -> Self {
        let mut nucleus = Self {
            name: name.to_string(),
            n: 0,
            z: 0,
            r: 0.0,
            a: 0.0,
            w: 0.0,
            r2: 0.0,
            a2: 0.0,
            w2: 0.0,
            beta2: 0.0,
            beta3: 0.0,
            beta4: 0.0,
            gamma: 0.0,
            min_dist: 0.4,
            node_dist: -1.0,
            smearing: 0.0,
            recenter: 1,
            lattice: 0,
            smax: 99.0,
            profile_type: DensityProfile::WoodsSaxon3PF,
            trials: 0,
            non_smeared: 0,
            weight: 1.0,
            nucleons: Vec::new(),
            phi_rot: 0.0,
            theta_rot: 0.0,
            x_rot: 0.0,
            y_rot: 0.0,
            z_rot: 0.0,
            max_r: 15.0,
            r0: 0.0,
            r1: 0.0,
            r2_factor: 0.0,
            density_env_p: -1.0,
            density_env_n: -1.0,
            density_env_deformed: -1.0,
        };
        nucleus.lookup(name);
        nucleus
    }

    /// Lookup nucleus parameters by name
    fn lookup(&mut self, name: &str) {
        match name {
            // Protons
            "p" | "pi" => {
                self.n = 1;
                self.z = 1;
                self.r = 0.234;
                self.profile_type = DensityProfile::ProtonExp;
            }
            "pg" => {
                self.n = 1;
                self.z = 1;
                self.r = 0.514;
                self.profile_type = DensityProfile::ProtonGaussian;
            }
            "pdg" => {
                self.n = 1;
                self.z = 1;
                self.r = 1.0;
                self.profile_type = DensityProfile::ProtonDGaussian;
            }
            // Deuteron
            "dpf" => {
                self.n = 2;
                self.z = 1;
                self.r = 0.01;
                self.a = 0.5882;
                self.profile_type = DensityProfile::WoodsSaxon3PF;
            }
            "dh" => {
                self.n = 2;
                self.z = 1;
                self.r = 0.2283;
                self.a = 1.1765;
                self.profile_type = DensityProfile::Hulthen;
            }
            "d" => {
                self.n = 2;
                self.z = 1;
                self.r = 0.2283;
                self.a = 1.1765;
                self.profile_type = DensityProfile::HulthenConstrained;
            }
            // Light nuclei from files
            "He3" => {
                self.n = 3;
                self.z = 1;
                self.profile_type = DensityProfile::FromFile;
            }
            "H3" => {
                self.n = 3;
                self.z = 2;
                self.profile_type = DensityProfile::FromFile;
            }
            "He4" => {
                self.n = 4;
                self.z = 2;
                self.profile_type = DensityProfile::FromFile;
            }
            "C" => {
                self.n = 12;
                self.z = 6;
                self.profile_type = DensityProfile::FromFile;
            }
            // Nitrogen
            "Npar" => {
                self.n = 14;
                self.z = 7;
                self.r = 2.570;
                self.a = 0.0572;
                self.w = -0.0180;
                self.profile_type = DensityProfile::WoodsSaxon3PF;
            }
            // Oxygen
            "O" => {
                self.n = 16;
                self.z = 8;
                self.profile_type = DensityProfile::FromFile;
            }
            "Opar" => {
                self.n = 16;
                self.z = 8;
                self.r = 2.608;
                self.a = 0.513;
                self.w = -0.051;
                self.max_r = 7.5;
                self.profile_type = DensityProfile::WoodsSaxon3PF;
            }
            "Opar2" => {
                self.n = 16;
                self.z = 8;
                self.r = 1.850;
                self.a = 0.497;
                self.w = 0.912;
                self.max_r = 7.5;
                self.profile_type = DensityProfile::WoodsSaxon3PF;
            }
            "Osat" => {
                self.n = 16;
                self.z = 8;
                self.max_r = 7.5;
                self.profile_type = DensityProfile::FromGraph;
            }
            "Odat" => {
                self.n = 16;
                self.z = 8;
                self.r = 2.608;
                self.a = 0.513;
                self.w = -0.051;
                self.max_r = 7.5;
                self.profile_type = DensityProfile::Oxygen1970;
            }
            "Oho" => {
                self.n = 16;
                self.z = 8;
                self.r = 1.544;
                self.a = 1.833;
                self.max_r = 7.5;
                self.profile_type = DensityProfile::HarmonicOscillator;
            }
            "Oho2" => {
                self.n = 16;
                self.z = 8;
                self.r = 1.506;
                self.a = 1.819;
                self.max_r = 7.5;
                self.profile_type = DensityProfile::HarmonicOscillator;
            }
            // Neon
            "Ne" => {
                self.n = 20;
                self.z = 10;
                self.r = 2.805;
                self.a = 0.571;
                self.max_r = 8.5;
                self.profile_type = DensityProfile::WoodsSaxon3PF;
            }
            "Ne2" => {
                self.n = 20;
                self.z = 10;
                self.r = 2.740;
                self.a = 0.572;
                self.max_r = 8.5;
                self.profile_type = DensityProfile::WoodsSaxon3PF;
            }
            "Ne3" => {
                self.n = 20;
                self.z = 10;
                self.r = 2.791;
                self.a = 0.698;
                self.w = -0.168;
                self.max_r = 8.5;
                self.profile_type = DensityProfile::WoodsSaxon3PF;
            }
            "NeTr2" => {
                self.n = 20;
                self.z = 10;
                self.r = 2.8;
                self.a = 0.57;
                self.beta2 = 0.721;
                self.max_r = 10.0;
                self.profile_type = DensityProfile::DeformedTF2;
            }
            "NeTr3" => {
                self.n = 20;
                self.z = 10;
                self.r = 2.7243;
                self.a = 0.4982;
                self.beta2 = 0.4899;
                self.beta3 = 0.2160;
                self.beta4 = 0.3055;
                self.gamma = 0.0;
                self.max_r = 10.0;
                self.profile_type = DensityProfile::DeformedBox;
            }
            // Aluminum
            "Al" => {
                self.n = 27;
                self.z = 13;
                self.r = 3.34;
                self.a = 0.580;
                self.beta2 = -0.448;
                self.beta4 = 0.239;
                self.max_r = 10.0;
                self.profile_type = DensityProfile::DeformedTF2;
            }
            // Silicon
            "Si" => {
                self.n = 28;
                self.z = 14;
                self.r = 3.34;
                self.a = 0.580;
                self.w = -0.233;
                self.max_r = 10.0;
                self.profile_type = DensityProfile::WoodsSaxon3PF;
            }
            "Si2" => {
                self.n = 28;
                self.z = 14;
                self.r = 3.34;
                self.a = 0.580;
                self.beta2 = -0.478;
                self.beta4 = 0.250;
                self.max_r = 10.0;
                self.profile_type = DensityProfile::DeformedTF2;
            }
            // Sulfur
            "S" => {
                self.n = 32;
                self.z = 16;
                self.r = 2.54;
                self.a = 2.191;
                self.w = 0.16;
                self.max_r = 10.0;
                self.profile_type = DensityProfile::WoodsSaxon3PG;
            }
            // Argon
            "Ar" => {
                self.n = 40;
                self.z = 18;
                self.r = 3.53;
                self.a = 0.542;
                self.max_r = 10.0;
                self.profile_type = DensityProfile::WoodsSaxon3PF;
            }
            // Calcium
            "Ca" => {
                self.n = 40;
                self.z = 20;
                self.r = 3.766;
                self.a = 0.586;
                self.w = -0.161;
                self.max_r = 10.0;
                self.profile_type = DensityProfile::WoodsSaxon3PF;
            }
            // Nickel
            "Ni" => {
                self.n = 58;
                self.z = 28;
                self.r = 4.309;
                self.a = 0.517;
                self.w = -0.1308;
                self.max_r = 10.0;
                self.profile_type = DensityProfile::WoodsSaxon3PF;
            }
            // Copper
            "Cu" => {
                self.n = 63;
                self.z = 29;
                self.r = 4.20;
                self.a = 0.596;
                self.max_r = 10.0;
                self.profile_type = DensityProfile::WoodsSaxon3PF;
            }
            "Curw" => {
                self.n = 63;
                self.z = 29;
                self.r = 4.20;
                self.a = 0.596;
                self.max_r = 10.0;
                self.r0 = 1.00898;
                self.r1 = -0.000790403;
                self.r2_factor = -0.000389897;
                self.profile_type = DensityProfile::Reweighted;
            }
            "Cu2" => {
                self.n = 63;
                self.z = 29;
                self.r = 4.20;
                self.a = 0.596;
                self.beta2 = 0.162;
                self.beta4 = -0.006;
                self.max_r = 10.0;
                self.profile_type = DensityProfile::DeformedTF2;
            }
            "Cu2rw" => {
                self.n = 63;
                self.z = 29;
                self.r = 4.20;
                self.a = 0.596;
                self.beta2 = 0.162;
                self.beta4 = -0.006;
                self.max_r = 10.0;
                self.r0 = 1.01269;
                self.r1 = -0.00298083;
                self.r2_factor = -9.97222e-05;
                self.profile_type = DensityProfile::DeformedReweighted;
            }
            "CuHN" => {
                self.n = 63;
                self.z = 29;
                self.r = 4.28;
                self.a = 0.5;
                self.max_r = 10.0;
                self.profile_type = DensityProfile::WoodsSaxon3PF;
            }
            // Niobium
            "Nb93LB" => {
                self.n = 93;
                self.z = 41;
                self.r = 4.9853;
                self.a = 0.5234;
                self.profile_type = DensityProfile::WoodsSaxon3PF;
            }
            // Zirconium
            "Zr96LB" => {
                self.n = 96;
                self.z = 40;
                self.r = 5.0212;
                self.a = 0.5234;
                self.profile_type = DensityProfile::WoodsSaxon3PF;
            }
            // Ruthenium
            "Ru96LB" => {
                self.n = 96;
                self.z = 44;
                self.r = 5.0845;
                self.a = 0.5234;
                self.profile_type = DensityProfile::WoodsSaxon3PF;
            }
            // Silver
            "Ag107LB" => {
                self.n = 107;
                self.z = 47;
                self.r = 5.3006;
                self.a = 0.5234;
                self.profile_type = DensityProfile::WoodsSaxon3PF;
            }
            "Ag109LB" => {
                self.n = 109;
                self.z = 47;
                self.r = 5.3306;
                self.a = 0.5234;
                self.profile_type = DensityProfile::WoodsSaxon3PF;
            }
            "Ag107pn" => {
                self.n = 107;
                self.z = 47;
                self.r = 5.2731;
                self.a = 0.4749;
                self.r2 = 5.4262;
                self.a2 = 0.4776;
                self.profile_type = DensityProfile::ProtonNeutron3PF;
            }
            "Ag109pn" => {
                self.n = 109;
                self.z = 47;
                self.r = 5.2943;
                self.a = 0.4729;
                self.r2 = 5.4762;
                self.a2 = 0.4788;
                self.profile_type = DensityProfile::ProtonNeutron3PF;
            }
            "Ag107pnHFB14" => {
                self.n = 107;
                self.z = 47;
                self.r = 5.2875;
                self.a = 0.4788;
                self.r2 = 5.287;
                self.a2 = 0.5498;
                self.profile_type = DensityProfile::ProtonNeutron3PF;
            }
            "Ag109pnHFB14" => {
                self.n = 109;
                self.z = 47;
                self.r = 5.3160;
                self.a = 0.4776;
                self.r2 = 5.3246;
                self.a2 = 0.5593;
                self.profile_type = DensityProfile::ProtonNeutron3PF;
            }
            // Tin stable isotopes
            "Sn112" => {
                self.n = 112;
                self.z = 50;
                self.r = 5.3714;
                self.a = 0.5234;
                self.profile_type = DensityProfile::WoodsSaxon3PF;
            }
            "Sn114" => {
                self.n = 114;
                self.z = 50;
                self.r = 5.3943;
                self.a = 0.5234;
                self.profile_type = DensityProfile::WoodsSaxon3PF;
            }
            "Sn116" => {
                self.n = 116;
                self.z = 50;
                self.r = 5.4173;
                self.a = 0.5234;
                self.profile_type = DensityProfile::WoodsSaxon3PF;
            }
            "Sn117" => {
                self.n = 117;
                self.z = 50;
                self.r = 5.1241;
                self.a = 0.5234;
                self.profile_type = DensityProfile::WoodsSaxon3PF;
            }
            "Sn118" => {
                self.n = 118;
                self.z = 50;
                self.r = 5.4391;
                self.a = 0.5234;
                self.profile_type = DensityProfile::WoodsSaxon3PF;
            }
            "Sn119" => {
                self.n = 119;
                self.z = 50;
                self.r = 5.4431;
                self.a = 0.5234;
                self.profile_type = DensityProfile::WoodsSaxon3PF;
            }
            "Sn120" => {
                self.n = 120;
                self.z = 50;
                self.r = 5.4588;
                self.a = 0.5234;
                self.profile_type = DensityProfile::WoodsSaxon3PF;
            }
            "Sn122" => {
                self.n = 122;
                self.z = 50;
                self.r = 5.4761;
                self.a = 0.5234;
                self.profile_type = DensityProfile::WoodsSaxon3PF;
            }
            "Sn124" => {
                self.n = 124;
                self.z = 50;
                self.r = 5.4907;
                self.a = 0.5234;
                self.profile_type = DensityProfile::WoodsSaxon3PF;
            }
            // Tin with pr3 parametrization
            name if name.starts_with("Sn") && name.ends_with("pr3") => {
                let (n_val, r_val, a_val, w_val) = match name {
                    "Sn112pr3" => (112, 4.962, 2.638 / (4.0 * 3.0_f64.ln()), 0.285),
                    "Sn114pr3" => (114, 4.971, 2.636 / (4.0 * 3.0_f64.ln()), 0.320),
                    "Sn116pr3" => (116, 5.062, 2.625 / (4.0 * 3.0_f64.ln()), 0.272),
                    "Sn117pr3" => (117, 5.058, 2.625 / (4.0 * 3.0_f64.ln()), 0.295),
                    "Sn118pr3" => (118, 5.072, 2.623 / (4.0 * 3.0_f64.ln()), 0.304),
                    "Sn119pr3" => (119, 5.100, 2.618 / (4.0 * 3.0_f64.ln()), 0.290),
                    "Sn120pr3" => (120, 5.110, 2.619 / (4.0 * 3.0_f64.ln()), 0.292),
                    "Sn122pr3" => (122, 5.088, 2.611 / (4.0 * 3.0_f64.ln()), 0.378),
                    "Sn124pr3" => (124, 5.150, 2.615 / (4.0 * 3.0_f64.ln()), 0.311),
                    _ => (0, 0.0, 0.0, 0.0),
                };
                self.n = n_val;
                self.z = 50;
                self.r = r_val;
                self.a = a_val;
                self.w = w_val;
                self.profile_type = DensityProfile::WoodsSaxon3PF;
            }
            // Tin non-stable isotopes
            "Sn108" => {
                self.n = 108;
                self.z = 50;
                self.r = 5.3274;
                self.a = 0.5234;
                self.profile_type = DensityProfile::WoodsSaxon3PF;
            }
            "Sn132" => {
                self.n = 132;
                self.z = 50;
                self.r = 5.5387;
                self.a = 0.5234;
                self.profile_type = DensityProfile::WoodsSaxon3PF;
            }
            // Iodine
            "I" => {
                self.n = 127;
                self.z = 53;
                self.r = 5.66;
                self.a = 0.54;
                self.profile_type = DensityProfile::WoodsSaxon3PF;
            }
            "IHS" => {
                self.n = 127;
                self.z = 53;
                self.r = 5.66;
                self.a = 0.00001;
                self.profile_type = DensityProfile::WoodsSaxon3PF;
            }
            // Xenon
            "Xe" => {
                self.n = 129;
                self.z = 54;
                self.r = 5.36;
                self.a = 0.59;
                self.max_r = 10.72;
                self.profile_type = DensityProfile::WoodsSaxon3PF;
            }
            "XeDef" => {
                self.n = 129;
                self.z = 54;
                self.r = 5.36;
                self.a = 0.54;
                self.max_r = 10.72;
                self.profile_type = DensityProfile::WoodsSaxon3PF;
            }
            "XeDef2" => {
                self.n = 129;
                self.z = 54;
                self.r = 5.26;
                self.a = 0.54;
                self.max_r = 10.72;
                self.profile_type = DensityProfile::WoodsSaxon3PF;
            }
            "XeDef3" => {
                self.n = 129;
                self.z = 54;
                self.r = 5.46;
                self.a = 0.54;
                self.max_r = 10.72;
                self.profile_type = DensityProfile::WoodsSaxon3PF;
            }
            "XeDef4" => {
                self.n = 129;
                self.z = 54;
                self.r = 5.36;
                self.a = 0.59;
                self.max_r = 10.72;
                self.profile_type = DensityProfile::WoodsSaxon3PF;
            }
            "XeDef5" => {
                self.n = 129;
                self.z = 54;
                self.r = 5.36;
                self.a = 0.49;
                self.max_r = 10.72;
                self.profile_type = DensityProfile::WoodsSaxon3PF;
            }
            "XeDCM" => {
                self.n = 129;
                self.z = 54;
                self.r = 5.336;
                self.a = 0.545;
                self.max_r = 6.94;
                self.profile_type = DensityProfile::WoodsSaxon3PF;
            }
            "Xes" => {
                self.n = 129;
                self.z = 54;
                self.r = 5.42;
                self.a = 0.57;
                self.profile_type = DensityProfile::WoodsSaxon3PF;
            }
            "Xe2" => {
                self.n = 129;
                self.z = 54;
                self.r = 5.36;
                self.a = 0.59;
                self.beta2 = 0.161;
                self.beta4 = -0.003;
                self.profile_type = DensityProfile::DeformedTF2;
            }
            "Xe2a" => {
                self.n = 129;
                self.z = 54;
                self.r = 5.36;
                self.a = 0.59;
                self.beta2 = 0.18;
                self.beta4 = 0.0;
                self.profile_type = DensityProfile::DeformedTF2;
            }
            "Xerw" => {
                self.n = 129;
                self.z = 54;
                self.r = 5.36;
                self.a = 0.59;
                self.r0 = 1.00911;
                self.r1 = -0.000722999;
                self.r2_factor = -0.0002663;
                self.profile_type = DensityProfile::Reweighted;
            }
            "Xesrw" => {
                self.n = 129;
                self.z = 54;
                self.r = 5.42;
                self.a = 0.57;
                self.r0 = 1.0096;
                self.r1 = -0.000874123;
                self.r2_factor = -0.000256708;
                self.profile_type = DensityProfile::Reweighted;
            }
            "Xe2arw" => {
                self.n = 129;
                self.z = 54;
                self.r = 5.36;
                self.a = 0.59;
                self.beta2 = 0.18;
                self.beta4 = 0.0;
                self.r0 = 1.01246;
                self.r1 = -0.0024851;
                self.r2_factor = -5.72464e-05;
                self.profile_type = DensityProfile::DeformedReweighted;
            }
            "Xe124" => {
                self.n = 124;
                self.z = 54;
                self.r = 5.431;
                self.a = 0.5978;
                self.beta2 = 0.212;
                self.beta4 = -0.018;
                self.profile_type = DensityProfile::DeformedTF2;
            }
            "Xe124HS" => {
                self.n = 124;
                self.z = 54;
                self.r = 5.431;
                self.a = 0.00001;
                self.profile_type = DensityProfile::WoodsSaxon3PF;
            }
            // Cesium
            "CsI" => {
                self.n = 130;
                self.z = 54;
                self.r = 5.71;
                self.a = 0.54;
                self.profile_type = DensityProfile::WoodsSaxon3PF;
            }
            "CsIHS" => {
                self.n = 130;
                self.z = 54;
                self.r = 5.71;
                self.a = 0.00001;
                self.profile_type = DensityProfile::WoodsSaxon3PF;
            }
            "Cs" => {
                self.n = 133;
                self.z = 55;
                self.r = 5.76;
                self.a = 0.54;
                self.profile_type = DensityProfile::WoodsSaxon3PF;
            }
            "CsHS" => {
                self.n = 133;
                self.z = 55;
                self.r = 5.76;
                self.a = 0.00001;
                self.profile_type = DensityProfile::WoodsSaxon3PF;
            }
            // Tungsten
            "W184" => {
                self.n = 184;
                self.z = 74;
                self.r = 6.52;
                self.a = 0.535;
                self.profile_type = DensityProfile::WoodsSaxon3PF;
            }
            "W184LB" => {
                self.n = 184;
                self.z = 74;
                self.r = 6.3599;
                self.a = 0.523;
                self.profile_type = DensityProfile::WoodsSaxon3PF;
            }
            "W" => {
                self.n = 186;
                self.z = 74;
                self.r = 6.58;
                self.a = 0.480;
                self.profile_type = DensityProfile::WoodsSaxon3PF;
            }
            "W186LB" => {
                self.n = 186;
                self.z = 74;
                self.r = 6.3839;
                self.a = 0.523;
                self.profile_type = DensityProfile::WoodsSaxon3PF;
            }
            // Gold
            "Au" => {
                self.n = 197;
                self.z = 79;
                self.r = 6.38;
                self.a = 0.535;
                self.max_r = 10.0;
                self.profile_type = DensityProfile::WoodsSaxon3PF;
            }
            "Aurw" => {
                self.n = 197;
                self.z = 79;
                self.r = 6.38;
                self.a = 0.535;
                self.max_r = 10.0;
                self.r0 = 1.00899;
                self.r1 = -0.000590908;
                self.r2_factor = -0.000210598;
                self.profile_type = DensityProfile::Reweighted;
            }
            "Au2" => {
                self.n = 197;
                self.z = 79;
                self.r = 6.38;
                self.a = 0.535;
                self.beta2 = -0.131;
                self.beta4 = -0.031;
                self.max_r = 10.0;
                self.profile_type = DensityProfile::DeformedTF2;
            }
            "Au2rw" => {
                self.n = 197;
                self.z = 79;
                self.r = 6.38;
                self.a = 0.535;
                self.beta2 = -0.131;
                self.beta4 = -0.031;
                self.max_r = 10.0;
                self.r0 = 1.01261;
                self.r1 = -0.00225517;
                self.r2_factor = -3.71513e-05;
                self.profile_type = DensityProfile::DeformedReweighted;
            }
            "AuHN" => {
                self.n = 197;
                self.z = 79;
                self.r = 6.42;
                self.a = 0.44;
                self.max_r = 10.0;
                self.profile_type = DensityProfile::WoodsSaxon3PF;
            }
            "Au197LB" => {
                self.n = 197;
                self.z = 79;
                self.r = 6.5541;
                self.a = 0.523;
                self.profile_type = DensityProfile::WoodsSaxon3PF;
            }
            "Au4pn" => {
                self.n = 197;
                self.z = 79;
                self.r = 6.538;
                self.a = 0.465;
                self.r2 = 6.794;
                self.a2 = 0.483;
                self.profile_type = DensityProfile::ProtonNeutron3PF;
            }
            "Au197pnHFB14" => {
                self.n = 197;
                self.z = 79;
                self.r = 6.5831;
                self.a = 0.4628;
                self.r2 = 6.6604;
                self.a2 = 0.5464;
                self.profile_type = DensityProfile::ProtonNeutron3PF;
            }
            // Lead
            "Pb" => {
                self.n = 208;
                self.z = 82;
                self.r = 6.62;
                self.a = 0.546;
                self.max_r = 10.0;
                self.profile_type = DensityProfile::WoodsSaxon3PF;
            }
            "Pbrw" => {
                self.n = 208;
                self.z = 82;
                self.r = 6.62;
                self.a = 0.546;
                self.max_r = 10.0;
                self.r0 = 1.00863;
                self.r1 = -0.00044808;
                self.r2_factor = -0.000205872;
                self.profile_type = DensityProfile::Reweighted;
            }
            "Pb*" => {
                self.n = 208;
                self.z = 82;
                self.r = 6.624;
                self.a = 0.549;
                self.max_r = 10.0;
                self.profile_type = DensityProfile::WoodsSaxon3PF;
            }
            "PbHN" => {
                self.n = 208;
                self.z = 82;
                self.r = 6.65;
                self.a = 0.460;
                self.max_r = 10.0;
                self.profile_type = DensityProfile::WoodsSaxon3PF;
            }
            "Pbpn" => {
                self.n = 208;
                self.z = 82;
                self.r = 6.68;
                self.a = 0.447;
                self.r2 = 6.69;
                self.a2 = 0.56;
                self.profile_type = DensityProfile::ProtonNeutron3PF;
            }
            "Pbpnrw" => {
                self.n = 208;
                self.z = 82;
                self.r = 6.68;
                self.a = 0.447;
                self.r2 = 6.69;
                self.a2 = 0.56;
                self.recenter = 1;
                self.smax = 0.1;
                self.r0 = 1.00866;
                self.r1 = -0.000461484;
                self.r2_factor = -0.000203571;
                self.profile_type = DensityProfile::ProtonNeutronReweighted;
            }
            // Bismuth
            "Bi" => {
                self.n = 209;
                self.z = 83;
                self.r = 6.75;
                self.a = 0.468;
                self.profile_type = DensityProfile::WoodsSaxon3PF;
            }
            "BiGS" => {
                self.n = 209;
                self.z = 83;
                self.r = 6.315;
                self.a = 2.881;
                self.w = 0.39;
                self.profile_type = DensityProfile::WoodsSaxon3PG;
            }
            // Uranium
            "U" => {
                self.n = 238;
                self.z = 92;
                self.r = 6.188;
                self.a = 0.54;
                self.beta2 = 1.77;
                self.max_r = 10.0;
                self.profile_type = DensityProfile::Ellipsoid;
            }
            "U2" => {
                self.n = 238;
                self.z = 92;
                self.r = 6.67;
                self.a = 0.44;
                self.beta2 = 0.280;
                self.beta4 = 0.093;
                self.max_r = 10.0;
                self.profile_type = DensityProfile::DeformedTF2;
            }
            // Trajectum models
            name if name.starts_with("TR_") => {
                self.profile_type = DensityProfile::Trajectum;
            }
            // Input from file
            name if name.starts_with("input") => {
                self.profile_type = DensityProfile::FromFile;
            }
            // Unknown nucleus
            _ => {
                eprintln!("Warning: Could not find nucleus {} in lookup table", name);
            }
        }
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

        // Special handling for Hulthen (deuteron)
        if is_hulthen {
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
