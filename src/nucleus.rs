// src/nucleus.rs
use crate::constants::{PI, TWO_PI};
use crate::nucleon::{NucleonType, TGlauNucleon};
use crate::random::RngExt;
use rand::Rng;
use rand::rngs::ThreadRng;
use std::collections::HashMap;

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
    n: i32,         // Number of nucleons
    z: i32,         // Number of protons
    r: f64,         // Radius of the 3pF function
    a: f64,         // Thickness of the 3pF function
    w: f64,         // Shape parameter of the 3pF function
    r2: f64,        // Radius for neutron distribution
    a2: f64,        // Thickness for neutron distribution
    w2: f64,        // Shape parameter for neutron distribution
    beta2: f64,     // Beta2 deformation
    beta3: f64,     // Beta3 deformation
    beta4: f64,     // Beta4 deformation
    gamma: f64,     // Gamma deformation
    min_dist: f64,  // Minimum separation distance
    node_dist: f64, // Average node distance (≤0 for continuous)
    smearing: f64,  // Node smearing
    recenter: i32, // Recentering method (0=none, 1=all, 2=displace one, 3=rotate+shift, 4=rotate only, 5=transverse)
    lattice: i32,  // Lattice type (0=HCP, 1=PCS, 2=BCC, 3=FCC)
    smax: f64,     // Maximum magnitude of cms shift tolerated
    profile_type: DensityProfile,
    trials: i32,      // Number of trials needed to complete nucleus
    non_smeared: i32, // Number of non-smeared-node nucleons
    weight: f64,      // Weight of nucleus
    nucleons: Vec<TGlauNucleon>,
    phi_rot: f64,
    theta_rot: f64,
    x_rot: f64,
    y_rot: f64,
    z_rot: f64,
    max_r: f64, // Maximum radius
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
        };
        nucleus.lookup(name);
        nucleus
    }

    /// Lookup nucleus parameters by name
    fn lookup(&mut self, name: &str) {
        match name {
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
            "Osat" => {
                self.n = 16;
                self.z = 8;
                self.max_r = 7.5;
                self.profile_type = DensityProfile::FromGraph;
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
            "Cu" => {
                self.n = 63;
                self.z = 29;
                self.r = 4.20;
                self.a = 0.596;
                self.max_r = 10.0;
                self.profile_type = DensityProfile::WoodsSaxon3PF;
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
            "Au" => {
                self.n = 197;
                self.z = 79;
                self.r = 6.38;
                self.a = 0.535;
                self.max_r = 10.0;
                self.profile_type = DensityProfile::WoodsSaxon3PF;
            }
            "Pb" => {
                self.n = 208;
                self.z = 82;
                self.r = 6.62;
                self.a = 0.546;
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
                self.profile_type = DensityProfile::ProtonNeutronReweighted;
            }
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
            _ if name.starts_with("TR_") => {
                self.profile_type = DensityProfile::Trajectum;
            }
            _ if name.starts_with("input") => {
                self.profile_type = DensityProfile::FromFile;
            }
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
    fn randomize_nucleons(&mut self, rng: &mut ThreadRng) {
        let mut iz = 0;
        for i in 0..self.n as usize {
            let frac = (self.z - iz) as f64 / (self.n - i as i32) as f64;
            let rn: f64 = rng.r#gen();
            if rn < frac {
                self.nucleons[i].set_type(NucleonType::Proton);
                iz += 1;
            } else {
                self.nucleons[i].set_type(NucleonType::Neutron);
            }
        }
    }

    /// Test if a nucleon is within the minimum distance of existing nucleons
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

    /// Woods-Saxon density evaluation
    fn woods_saxon_radius(&self, r: f64) -> f64 {
        let r = r.abs();
        if r > self.max_r {
            return 0.0;
        }
        let w_term = 1.0 + self.w * (r / self.r).powi(2);
        let denom = 1.0 + ((r - self.r) / self.a).exp();
        w_term / denom
    }

    /// Generate random radius from Woods-Saxon distribution (rejection sampling)
    fn random_woods_saxon(&self, rng: &mut ThreadRng) -> f64 {
        const MAX_TRIALS: usize = 10000;
        for _ in 0..MAX_TRIALS {
            let r = rng.r#gen_range(0.0..self.max_r);
            let rho = self.woods_saxon_radius(r);
            let r2 = r * r;
            let weight = rho * r2;
            let max_weight = self.max_r * self.max_r * 1.0; // Approximate
            if rng.r#gen::<f64>() * max_weight < weight {
                return r;
            }
        }
        self.max_r * rng.r#gen::<f64>()
    }

    /// Throw nucleons according to the density profile
    pub fn throw_nucleons(&mut self, xshift: f64, rng: &mut ThreadRng) -> [f64; 3] {
        self.allocate_nucleons();
        self.randomize_nucleons(rng);

        self.trials = 0;
        self.non_smeared = 0;
        self.phi_rot = rng.r#gen::<f64>() * TWO_PI;
        let cos_theta = 2.0 * rng.r#gen::<f64>() - 1.0;
        self.theta_rot = cos_theta.acos();
        self.x_rot = rng.r#gen::<f64>() * TWO_PI;
        self.y_rot = rng.r#gen::<f64>() * TWO_PI;
        self.z_rot = rng.r#gen::<f64>() * TWO_PI;

        let is_hulthen = matches!(
            self.profile_type,
            DensityProfile::Hulthen | DensityProfile::HulthenConstrained
        );

        // Store nucleon positions temporarily to avoid borrowing issues
        let mut positions: Vec<(f64, f64, f64)> = Vec::with_capacity(self.n as usize);

        match self.profile_type {
            DensityProfile::ProtonExp
            | DensityProfile::ProtonGaussian
            | DensityProfile::ProtonDGaussian
            | DensityProfile::WoodsSaxon3PF
            | DensityProfile::HarmonicOscillator => {
                for _ in 0..self.n as usize {
                    let r = self.random_woods_saxon(rng);
                    let phi = rng.r#gen::<f64>() * TWO_PI;
                    let ctheta = 2.0 * rng.r#gen::<f64>() - 1.0;
                    let stheta = (1.0 - ctheta * ctheta).sqrt();

                    let x = r * stheta * phi.cos();
                    let y = r * stheta * phi.sin();
                    let z = r * ctheta;

                    positions.push((x, y, z));
                    self.trials += 1;
                }
            }
            DensityProfile::Ellipsoid | DensityProfile::DeformedBox => {
                for _ in 0..self.n as usize {
                    let mut placed = false;
                    while !placed {
                        let x = self.max_r * (2.0 * rng.r#gen::<f64>() - 1.0);
                        let y = self.max_r * (2.0 * rng.r#gen::<f64>() - 1.0);
                        let z = self.max_r * (2.0 * rng.r#gen::<f64>() - 1.0);
                        let r = (x * x + y * y + z * z).sqrt();
                        let theta = (z / r).acos();
                        let r_theta = self.r + self.beta2 * theta.cos().powi(2);

                        let prob = (1.0 + self.w * (r / r_theta).powi(2))
                            / (1.0 + ((r - r_theta) / self.a).exp());
                        if rng.r#gen::<f64>() < prob {
                            positions.push((x, y, z));
                            placed = true;
                        }
                        self.trials += 1;
                    }
                }
            }
            DensityProfile::DeformedTF2 => {
                for _ in 0..self.n as usize {
                    let mut placed = false;
                    while !placed {
                        let r = self.random_woods_saxon(rng);
                        let theta = rng.r#gen::<f64>() * PI;
                        let phi = rng.r#gen::<f64>() * TWO_PI;

                        let r_theta =
                            self.r * (1.0 + self.beta2 * 0.315 * (3.0 * theta.cos().powi(2) - 1.0));
                        let prob = 1.0 / (1.0 + ((r - r_theta) / self.a).exp());

                        if rng.r#gen::<f64>() < prob {
                            let x = r * theta.sin() * phi.cos();
                            let y = r * theta.sin() * phi.sin();
                            let z = r * theta.cos();
                            positions.push((x, y, z));
                            placed = true;
                        }
                        self.trials += 1;
                    }
                }
            }
            DensityProfile::Hulthen | DensityProfile::HulthenConstrained => {
                // Hulthen distribution for deuteron
                let r = self.random_woods_saxon(rng) / 2.0;
                let phi = rng.r#gen::<f64>() * TWO_PI;
                let ctheta = 2.0 * rng.r#gen::<f64>() - 1.0;
                let stheta = (1.0 - ctheta * ctheta).sqrt();

                let x1 = r * stheta * phi.cos();
                let y1 = r * stheta * phi.sin();
                let z1 = r * ctheta;
                positions.push((x1, y1, z1));

                if matches!(self.profile_type, DensityProfile::HulthenConstrained) {
                    positions.push((-x1, -y1, -z1));
                } else {
                    let r2 = self.random_woods_saxon(rng) / 2.0;
                    let phi2 = rng.r#gen::<f64>() * TWO_PI;
                    let ctheta2 = 2.0 * rng.r#gen::<f64>() - 1.0;
                    let stheta2 = (1.0 - ctheta2 * ctheta2).sqrt();
                    positions.push((
                        r2 * stheta2 * phi2.cos(),
                        r2 * stheta2 * phi2.sin(),
                        r2 * ctheta2,
                    ));
                }
                self.trials = 1;
            }
            _ => {
                // Default: uniform distribution within a sphere
                for _ in 0..self.n as usize {
                    let r = self.max_r * rng.r#gen::<f64>().powf(1.0 / 3.0);
                    let phi = rng.r#gen::<f64>() * TWO_PI;
                    let ctheta = 2.0 * rng.r#gen::<f64>() - 1.0;
                    let stheta = (1.0 - ctheta * ctheta).sqrt();

                    positions.push((r * stheta * phi.cos(), r * stheta * phi.sin(), r * ctheta));
                    self.trials += 1;
                }
            }
        }

        // Now set positions in nucleons
        for (i, (x, y, z)) in positions.into_iter().enumerate() {
            if i < self.nucleons.len() {
                self.nucleons[i].set_position(x, y, z);
                // Apply rotation for deformed nuclei
                if matches!(
                    self.profile_type,
                    DensityProfile::Ellipsoid
                        | DensityProfile::DeformedBox
                        | DensityProfile::DeformedTF2
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
            // Retry if shift is too large - use recursion but with a limit
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
