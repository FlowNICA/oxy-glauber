// src/nucleon.rs
use serde::{Deserialize, Serialize};

/// Represents a single nucleon in the Glauber model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TGlauNucleon {
    x: f64,
    y: f64,
    z: f64,
    type_: NucleonType,
    in_nucleus_a: bool,
    n_coll: i32,
    energy: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NucleonType {
    Neutron = 0,
    Proton = 1,
}

impl TGlauNucleon {
    pub fn new() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            type_: NucleonType::Neutron,
            in_nucleus_a: false,
            n_coll: 0,
            energy: 0.0,
        }
    }

    pub fn with_position(x: f64, y: f64, z: f64) -> Self {
        Self {
            x,
            y,
            z,
            ..Self::new()
        }
    }

    pub fn collide(&mut self) {
        self.n_coll += 1;
    }

    pub fn reset(&mut self) {
        self.n_coll = 0;
    }

    pub fn get_2c_weight(&self, x: f64) -> f64 {
        2.0 * (0.5 * (1.0 - x) + 0.5 * x * self.n_coll as f64)
    }

    pub fn x(&self) -> f64 {
        self.x
    }
    pub fn y(&self) -> f64 {
        self.y
    }
    pub fn z(&self) -> f64 {
        self.z
    }
    pub fn type_(&self) -> NucleonType {
        self.type_
    }
    pub fn n_coll(&self) -> i32 {
        self.n_coll
    }
    pub fn energy(&self) -> f64 {
        self.energy
    }
    pub fn is_neutron(&self) -> bool {
        self.type_ == NucleonType::Neutron
    }
    pub fn is_proton(&self) -> bool {
        self.type_ == NucleonType::Proton
    }
    pub fn is_wounded(&self) -> bool {
        self.n_coll > 0
    }
    pub fn is_spectator(&self) -> bool {
        self.n_coll == 0
    }
    pub fn in_nucleus_a(&self) -> bool {
        self.in_nucleus_a
    }
    pub fn in_nucleus_b(&self) -> bool {
        !self.in_nucleus_a
    }

    pub fn set_position(&mut self, x: f64, y: f64, z: f64) {
        self.x = x;
        self.y = y;
        self.z = z;
    }

    pub fn set_type(&mut self, t: NucleonType) {
        self.type_ = t;
    }

    pub fn set_in_nucleus_a(&mut self) {
        self.in_nucleus_a = true;
    }

    pub fn set_in_nucleus_b(&mut self) {
        self.in_nucleus_a = false;
    }

    pub fn set_n_coll(&mut self, n: i32) {
        self.n_coll = n;
    }

    pub fn set_energy(&mut self, en: f64) {
        self.energy = en;
    }

    pub fn rotate_2d(&mut self, phi: f64, theta: f64) {
        // Simplified rotation - in full implementation would use rotation matrices
        let x = self.x;
        let y = self.y;
        let z = self.z;
        let cos_phi = phi.cos();
        let sin_phi = phi.sin();
        let cos_theta = theta.cos();
        let sin_theta = theta.sin();

        // Rotate around Z axis by phi, then around X axis by theta
        let x1 = x * cos_phi - y * sin_phi;
        let y1 = x * sin_phi + y * cos_phi;
        let z1 = z;

        let x2 = x1;
        let y2 = y1 * cos_theta - z1 * sin_theta;
        let z2 = y1 * sin_theta + z1 * cos_theta;

        self.x = x2;
        self.y = y2;
        self.z = z2;
    }

    pub fn rotate_3d(&mut self, psi_x: f64, psi_y: f64, psi_z: f64) {
        let x = self.x;
        let y = self.y;
        let z = self.z;

        // Rotation around X axis
        let (sx, cx) = psi_x.sin_cos();
        let y1 = y * cx - z * sx;
        let z1 = y * sx + z * cx;

        // Rotation around Y axis
        let (sy, cy) = psi_y.sin_cos();
        let x2 = x * cy + z1 * sy;
        let z2 = -x * sy + z1 * cy;

        // Rotation around Z axis
        let (sz, cz) = psi_z.sin_cos();
        let x3 = x2 * cz - y1 * sz;
        let y3 = x2 * sz + y1 * cz;

        self.x = x3;
        self.y = y3;
        self.z = z2;
    }
}

impl Default for TGlauNucleon {
    fn default() -> Self {
        Self::new()
    }
}
