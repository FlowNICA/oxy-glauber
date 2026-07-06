// src/lib.rs
#![deny(unsafe_code)]

pub mod cross_section;
pub mod glauber;
pub mod nucleon;
pub mod nucleus;
pub mod profile;
pub mod random;

pub use cross_section::CrossSection;
pub use glauber::{TGlauberEvent, TGlauberMC};
pub use nucleon::TGlauNucleon;
pub use nucleus::TGlauNucleus;
pub use profile::{NNProfile, NNProfileType, profile_from_omega};

pub const VERSION: &str = "0.3.3";

/// Physical constants
pub mod constants {
    pub const PI: f64 = std::f64::consts::PI;
    pub const TWO_PI: f64 = 2.0 * PI;
    pub const MB_TO_FM2: f64 = 0.1; // 1 mb = 0.1 fm^2
    pub const FM_TO_MB: f64 = 10.0;
}
