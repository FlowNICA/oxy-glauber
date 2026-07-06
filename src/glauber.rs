// src/glauber.rs
use crate::constants::{MB_TO_FM2, PI};
use crate::cross_section::CrossSection;
use crate::nucleus::TGlauNucleus;
use crate::profile::{NNProfile, profile_from_omega};
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::ThreadRng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use std::sync::Arc;
use std::time::Instant;

/// Event results from Glauber simulation
#[derive(Debug, Clone)]
pub struct TGlauberEvent {
    pub npart: f32,
    pub ncoll: f32,
    pub nhard: f32,
    pub nmpi: f32,
    pub b: f32,
    pub bnn: f32,
    pub ncollpp: f32,
    pub ncollpn: f32,
    pub ncollnn: f32,
    pub var_x: f32,
    pub var_y: f32,
    pub var_xy: f32,
    pub npart_a: f32,
    pub npart_b: f32,
    pub npart0: f32,
    pub npart_an: f32,
    pub npart_bn: f32,
    pub npart0n: f32,
    pub area_w: f32,
    pub spec_a: f32,
    pub spec_b: f32,
    pub weight: f32,
    pub psi1: f32,
    pub ecc1: f32,
    pub psi2: f32,
    pub ecc2: f32,
    pub psi3: f32,
    pub ecc3: f32,
    pub psi4: f32,
    pub ecc4: f32,
    pub psi5: f32,
    pub ecc5: f32,
    pub area_o: f32,
    pub area_a: f32,
    pub x0: f32,
    pub y0: f32,
    pub phi0: f32,
    pub length: f32,
    pub mean_x: f32,
    pub mean_y: f32,
    pub mean_x2: f32,
    pub mean_y2: f32,
    pub mean_xy: f32,
    pub mean_x_system: f32,
    pub mean_y_system: f32,
    pub mean_x_a: f32,
    pub mean_y_a: f32,
    pub mean_x_b: f32,
    pub mean_y_b: f32,
    pub phi_a: f32,
    pub theta_a: f32,
    pub phi_b: f32,
    pub theta_b: f32,
}

impl Default for TGlauberEvent {
    fn default() -> Self {
        Self {
            npart: 0.0,
            ncoll: 0.0,
            nhard: 0.0,
            nmpi: 0.0,
            b: 0.0,
            bnn: 0.0,
            ncollpp: 0.0,
            ncollpn: 0.0,
            ncollnn: 0.0,
            var_x: 0.0,
            var_y: 0.0,
            var_xy: 0.0,
            npart_a: 0.0,
            npart_b: 0.0,
            npart0: 0.0,
            npart_an: 0.0,
            npart_bn: 0.0,
            npart0n: 0.0,
            area_w: 0.0,
            spec_a: 0.0,
            spec_b: 0.0,
            weight: 0.0,
            psi1: 0.0,
            ecc1: 0.0,
            psi2: 0.0,
            ecc2: 0.0,
            psi3: 0.0,
            ecc3: 0.0,
            psi4: 0.0,
            ecc4: 0.0,
            psi5: 0.0,
            ecc5: 0.0,
            area_o: 0.0,
            area_a: 0.0,
            x0: 0.0,
            y0: 0.0,
            phi0: 0.0,
            length: 0.0,
            mean_x: 0.0,
            mean_y: 0.0,
            mean_x2: 0.0,
            mean_y2: 0.0,
            mean_xy: 0.0,
            mean_x_system: 0.0,
            mean_y_system: 0.0,
            mean_x_a: 0.0,
            mean_y_a: 0.0,
            mean_x_b: 0.0,
            mean_y_b: 0.0,
            phi_a: 0.0,
            theta_a: 0.0,
            phi_b: 0.0,
            theta_b: 0.0,
        }
    }
}

/// Configuration for the Glauber model - used for parallel generation
#[derive(Clone)]
struct GlauberConfig {
    nucleus_a_name: String,
    nucleus_b_name: String,
    xsect: f64,
    xsect_np: f64,
    bmin: f64,
    bmax: f64,
    hard_frac: f64,
    min_dist: f64,
    node_dist: f64,
    omega: f64,
    two_c_x: f64,
    detail: i32,
    calc_area: bool,
    calc_length: bool,
    do_core: bool,
    sig_h: f64,
    nn_profile: Option<NNProfile>,
}

/// Main Glauber Monte Carlo simulation
pub struct TGlauberMC {
    nucleus_a: TGlauNucleus,
    nucleus_b: TGlauNucleus,
    xsect: f64,
    xsect_np: f64,
    bmin: f64,
    bmax: f64,
    hard_frac: f64,
    detail: i32,
    calc_area: bool,
    calc_length: bool,
    do_core: bool,
    sig_h: f64,
    omega: f64,
    max_npart_found: i32,
    two_c_x: f64,
    events: f64,
    total_events: f64,
    nn_profile: Option<NNProfile>,
    event: TGlauberEvent,
    bc: Vec<Vec<bool>>,
    config: GlauberConfig,
}

impl TGlauberMC {
    pub fn new(na: &str, nb: &str, xsect: f64, xsect_sigma: f64, xsect_np: f64) -> Self {
        let mut xsect_use = xsect;
        let mut xsect_np_use = xsect_np;
        let mut sig_h = 0.0;

        if xsect < 0.0 {
            let energy = -xsect;
            let (sig_nn, sig_np, sig_hard) = CrossSection::from_energy(energy);
            xsect_use = sig_nn;
            xsect_np_use = if xsect_np <= 0.0 { sig_np } else { xsect_np };
            sig_h = sig_hard;
            println!(
                "Using sigma_NN={:.1} mb and sigma_NP={:.1} mb for energy={:.1} GeV",
                xsect_use, xsect_np_use, energy
            );
        }

        if xsect_sigma > 0.0 {
            println!(
                "Using fluctuating cross section with sigma={:.1} mb",
                xsect_sigma
            );
        }

        let nucleus_a = TGlauNucleus::new(na);
        let nucleus_b = TGlauNucleus::new(nb);

        let config = GlauberConfig {
            nucleus_a_name: na.to_string(),
            nucleus_b_name: nb.to_string(),
            xsect: xsect_use,
            xsect_np: xsect_np_use,
            bmin: 0.0,
            bmax: 20.0,
            hard_frac: 0.65,
            min_dist: 0.4,
            node_dist: -1.0,
            omega: -1.0,
            two_c_x: 0.0,
            detail: 99,
            calc_area: false,
            calc_length: false,
            do_core: false,
            sig_h,
            nn_profile: None,
        };

        Self {
            nucleus_a,
            nucleus_b,
            xsect: xsect_use,
            xsect_np: xsect_np_use,
            bmin: 0.0,
            bmax: 20.0,
            hard_frac: 0.65,
            detail: 99,
            calc_area: false,
            calc_length: false,
            do_core: false,
            sig_h,
            omega: -1.0,
            max_npart_found: 0,
            two_c_x: 0.0,
            events: 0.0,
            total_events: 0.0,
            nn_profile: None,
            event: TGlauberEvent::default(),
            bc: Vec::new(),
            config,
        }
    }

    pub fn with_profile(mut self, profile: NNProfile) -> Self {
        self.nn_profile = Some(profile.clone());
        self.config.nn_profile = Some(profile);
        self
    }

    pub fn set_omega(&mut self, omega: f64) {
        if omega < 0.0 {
            println!("Using hard-sphere approximation (default)");
            self.omega = 0.0;
            self.config.omega = 0.0;
            return;
        }
        self.omega = omega;
        self.config.omega = omega;
        if let Some(profile) = profile_from_omega(self.xsect, omega) {
            self.nn_profile = Some(profile.clone());
            self.config.nn_profile = Some(profile);
        }
    }

    pub fn set_min_distance(&mut self, d: f64) {
        self.nucleus_a.set_min_dist(d);
        self.nucleus_b.set_min_dist(d);
        self.config.min_dist = d;
    }

    pub fn set_node_distance(&mut self, d: f64) {
        self.nucleus_a.set_node_dist(d);
        self.nucleus_b.set_node_dist(d);
        self.config.node_dist = d;
    }

    pub fn set_bmin(&mut self, bmin: f64) {
        self.bmin = bmin;
        self.config.bmin = bmin;
    }

    pub fn set_bmax(&mut self, bmax: f64) {
        self.bmax = bmax;
        self.config.bmax = bmax;
    }

    pub fn set_calc_area(&mut self, calc: bool) {
        self.calc_area = calc;
        self.config.calc_area = calc;
    }

    pub fn set_calc_length(&mut self, calc: bool) {
        self.calc_length = calc;
        self.config.calc_length = calc;
    }

    pub fn set_calc_core(&mut self, calc: bool) {
        self.do_core = calc;
        self.config.do_core = calc;
    }

    pub fn set_detail(&mut self, detail: i32) {
        self.detail = detail;
        self.config.detail = detail;
    }

    pub fn set_2cx(&mut self, x: f64) {
        self.two_c_x = x;
        self.config.two_c_x = x;
    }

    pub fn set_hard_frac(&mut self, f: f64) {
        self.hard_frac = f;
        self.config.hard_frac = f;
    }

    pub fn set_sigma_hard(&mut self, s: f64) {
        self.sig_h = s;
        self.config.sig_h = s;
    }

    pub fn nucleus_a(&self) -> &TGlauNucleus {
        &self.nucleus_a
    }
    pub fn nucleus_b(&self) -> &TGlauNucleus {
        &self.nucleus_b
    }
    pub fn event(&self) -> &TGlauberEvent {
        &self.event
    }
    pub fn b(&self) -> f64 {
        self.event.b as f64
    }
    pub fn bnn(&self) -> f64 {
        self.event.bnn as f64
    }
    pub fn npart(&self) -> i32 {
        self.event.npart as i32
    }
    pub fn ncoll(&self) -> i32 {
        self.event.ncoll as i32
    }
    pub fn nhard(&self) -> i32 {
        self.event.nhard as i32
    }
    pub fn nmpi(&self) -> i32 {
        self.event.nmpi as i32
    }
    pub fn npart_a(&self) -> i32 {
        self.event.npart_a as i32
    }
    pub fn npart_b(&self) -> i32 {
        self.event.npart_b as i32
    }
    pub fn npart0(&self) -> i32 {
        self.event.npart0 as i32
    }
    pub fn npart_an(&self) -> i32 {
        self.event.npart_an as i32
    }
    pub fn npart_bn(&self) -> i32 {
        self.event.npart_bn as i32
    }
    pub fn npart0n(&self) -> i32 {
        self.event.npart0n as i32
    }
    pub fn spec_a(&self) -> f64 {
        self.event.spec_a as f64
    }
    pub fn spec_b(&self) -> f64 {
        self.event.spec_b as f64
    }
    pub fn weight(&self) -> f64 {
        self.event.weight as f64
    }
    pub fn ecc(&self, n: usize) -> f64 {
        match n {
            1 => self.event.ecc1 as f64,
            2 => self.event.ecc2 as f64,
            3 => self.event.ecc3 as f64,
            4 => self.event.ecc4 as f64,
            5 => self.event.ecc5 as f64,
            _ => 0.0,
        }
    }
    pub fn psi(&self, n: usize) -> f64 {
        match n {
            1 => self.event.psi1 as f64,
            2 => self.event.psi2 as f64,
            3 => self.event.psi3 as f64,
            4 => self.event.psi4 as f64,
            5 => self.event.psi5 as f64,
            _ => 0.0,
        }
    }
    pub fn ncollpp(&self) -> i32 {
        self.event.ncollpp as i32
    }
    pub fn ncollpn(&self) -> i32 {
        self.event.ncollpn as i32
    }
    pub fn ncollnn(&self) -> i32 {
        self.event.ncollnn as i32
    }
    pub fn get_npart_found(&self) -> i32 {
        self.max_npart_found
    }

    pub fn total_xsect(&self) -> f64 {
        if self.total_events == 0.0 {
            return 0.0;
        }
        (self.events / self.total_events) * PI * self.bmax * self.bmax / 100.0
    }

    pub fn total_xsect_err(&self) -> f64 {
        if self.events == 0.0 {
            return 0.0;
        }
        self.total_xsect() / (self.events).sqrt() * (1.0 - self.events / self.total_events).sqrt()
    }

    /// Generate the next event (single-threaded version)
    pub fn next_event(&mut self, rng: &mut ThreadRng, bgen: Option<f64>) -> bool {
        let b = match bgen {
            Some(b) if b >= 0.0 => b,
            _ => {
                let b2_range = self.bmax * self.bmax - self.bmin * self.bmin;
                (b2_range * rng.r#gen::<f64>() + self.bmin * self.bmin).sqrt()
            }
        };

        self.nucleus_a.throw_nucleons(-b / 2.0, rng);
        self.nucleus_b.throw_nucleons(b / 2.0, rng);

        self.calc_event(rng, b)
    }

    /// Calculate event results
    fn calc_event(&mut self, rng: &mut ThreadRng, bgen: f64) -> bool {
        let nucleons_a = self.nucleus_a.nucleons();
        let nucleons_b = self.nucleus_b.nucleons();
        let an = nucleons_a.len();
        let bn = nucleons_b.len();

        self.bc.clear();
        self.bc.resize(an, vec![false; bn]);

        self.event = TGlauberEvent::default();

        let mut nc = 0;
        let mut nh = 0;
        let mut bnn_sum = 0.0;
        let mut x0 = 0.0;
        let mut y0 = 0.0;
        let mut first_collision = true;

        let d2pp = self.xsect * MB_TO_FM2 / PI;
        let d2np = if self.xsect_np > 0.0 {
            self.xsect_np * MB_TO_FM2 / PI
        } else {
            d2pp
        };
        let bh = (d2pp * self.hard_frac).sqrt();

        for i in 0..bn {
            let nucleon_b = &nucleons_b[i];
            for j in 0..an {
                let nucleon_a = &nucleons_a[j];
                let dx = nucleon_b.x() - nucleon_a.x();
                let dy = nucleon_b.y() - nucleon_a.y();
                let dij = dx * dx + dy * dy;

                let is_pp = nucleon_a.is_proton() && nucleon_b.is_proton();
                let is_nn = nucleon_a.is_neutron() && nucleon_b.is_neutron();
                let d2 = if self.xsect_np > 0.0 && !is_pp && !is_nn {
                    d2np
                } else {
                    d2pp
                };

                if dij > d2 {
                    continue;
                }

                let bij = dij.sqrt();
                let mut collides = true;

                if let Some(profile) = &self.nn_profile {
                    let val = profile.eval(bij);
                    if rng.r#gen::<f64>() > val {
                        collides = false;
                    }
                }

                if collides {
                    self.bc[j][i] = true;
                    nc += 1;
                    bnn_sum += bij;

                    if bij < bh {
                        nh += 1;
                    }

                    if is_pp {
                        self.event.ncollpp += 1.0;
                    } else if is_nn {
                        self.event.ncollnn += 1.0;
                    } else {
                        self.event.ncollpn += 1.0;
                    }

                    if first_collision {
                        x0 = (nucleon_a.x() + nucleon_b.x()) / 2.0;
                        y0 = (nucleon_a.y() + nucleon_b.y()) / 2.0;
                        first_collision = false;
                    }
                }
            }
        }

        if nc == 0 {
            self.event.b = bgen as f32;
            return false;
        }

        for i in 0..bn {
            let nucleon_b = &mut self.nucleus_b.nucleons_mut()[i];
            let mut ncoll_b = 0;
            for j in 0..an {
                if self.bc[j][i] {
                    ncoll_b += 1;
                }
            }
            nucleon_b.set_n_coll(ncoll_b);
        }

        for j in 0..an {
            let nucleon_a = &mut self.nucleus_a.nucleons_mut()[j];
            let mut ncoll_a = 0;
            for i in 0..bn {
                if self.bc[j][i] {
                    ncoll_a += 1;
                }
            }
            nucleon_a.set_n_coll(ncoll_a);
        }

        self.event.b = bgen as f32;
        self.event.bnn = (bnn_sum / nc as f64) as f32;
        self.event.ncoll = nc as f32;
        self.event.nhard = nh as f32;
        self.event.x0 = x0 as f32;
        self.event.y0 = y0 as f32;

        self.calc_participants();

        true
    }

    /// Calculate participant quantities
    fn calc_participants(&mut self) {
        let nucleons_a = self.nucleus_a.nucleons();
        let nucleons_b = self.nucleus_b.nucleons();

        let mut sum_w = 0.0;
        let mut sum_w_a = 0.0;
        let mut sum_w_b = 0.0;

        let mut npart_a = 0;
        let mut npart_b = 0;
        let mut npart0 = 0;
        let mut npart_an = 0;
        let mut npart_bn = 0;
        let mut npart0n = 0;
        let mut spec_a = 0;
        let mut spec_b = 0;

        let mut mean_x = 0.0;
        let mut mean_y = 0.0;
        let mut mean_x2 = 0.0;
        let mut mean_y2 = 0.0;
        let mut mean_xy = 0.0;
        let mut mean_x_system = 0.0;
        let mut mean_y_system = 0.0;
        let mut mean_x_a = 0.0;
        let mut mean_y_a = 0.0;
        let mut mean_x_b = 0.0;
        let mut mean_y_b = 0.0;

        for nucleon in nucleons_a {
            let x = nucleon.x();
            let y = nucleon.y();
            mean_x_system += x;
            mean_y_system += y;

            if nucleon.is_wounded() {
                let w = nucleon.get_2c_weight(self.two_c_x);
                let ncoll = nucleon.n_coll();

                npart_a += 1;
                if nucleon.is_neutron() {
                    npart_an += 1;
                }
                if ncoll == 1 {
                    npart0 += 1;
                    if nucleon.is_neutron() {
                        npart0n += 1;
                    }
                }

                sum_w += w;
                sum_w_a += 1.0;
                mean_x += x * w;
                mean_y += y * w;
                mean_x2 += x * x * w;
                mean_y2 += y * y * w;
                mean_xy += x * y * w;
                mean_x_a += x;
                mean_y_a += y;
            } else if nucleon.is_neutron() {
                spec_a += 1;
            }
        }

        for nucleon in nucleons_b {
            let x = nucleon.x();
            let y = nucleon.y();
            mean_x_system += x;
            mean_y_system += y;

            if nucleon.is_wounded() {
                let w = nucleon.get_2c_weight(self.two_c_x);
                let ncoll = nucleon.n_coll();

                npart_b += 1;
                if nucleon.is_neutron() {
                    npart_bn += 1;
                }
                if ncoll == 1 {
                    npart0 += 1;
                    if nucleon.is_neutron() {
                        npart0n += 1;
                    }
                }

                sum_w += w;
                sum_w_b += 1.0;
                mean_x += x * w;
                mean_y += y * w;
                mean_x2 += x * x * w;
                mean_y2 += y * y * w;
                mean_xy += x * y * w;
                mean_x_b += x;
                mean_y_b += y;
            } else if nucleon.is_neutron() {
                spec_b += 1;
            }
        }

        let total_n = (nucleons_a.len() + nucleons_b.len()) as f64;
        mean_x_system /= total_n;
        mean_y_system /= total_n;

        if sum_w > 0.0 {
            mean_x /= sum_w;
            mean_y /= sum_w;
            mean_x2 /= sum_w;
            mean_y2 /= sum_w;
            mean_xy /= sum_w;
        }

        if sum_w_a > 0.0 {
            mean_x_a /= sum_w_a;
            mean_y_a /= sum_w_a;
        }
        if sum_w_b > 0.0 {
            mean_x_b /= sum_w_b;
            mean_y_b /= sum_w_b;
        }

        let var_x = mean_x2 - mean_x * mean_x;
        let var_y = mean_y2 - mean_y * mean_y;
        let var_xy = mean_xy - mean_x * mean_y;
        let area_w = if var_x * var_y - var_xy * var_xy < 0.0 {
            -1.0
        } else {
            (var_x * var_y - var_xy * var_xy).sqrt()
        };

        self.event.npart = (npart_a + npart_b) as f32;
        self.event.npart_a = npart_a as f32;
        self.event.npart_b = npart_b as f32;
        self.event.npart0 = npart0 as f32;
        self.event.npart_an = npart_an as f32;
        self.event.npart_bn = npart_bn as f32;
        self.event.npart0n = npart0n as f32;
        self.event.spec_a = spec_a as f32;
        self.event.spec_b = spec_b as f32;
        self.event.var_x = var_x as f32;
        self.event.var_y = var_y as f32;
        self.event.var_xy = var_xy as f32;
        self.event.area_w = area_w as f32;
        self.event.mean_x = mean_x as f32;
        self.event.mean_y = mean_y as f32;
        self.event.mean_x2 = mean_x2 as f32;
        self.event.mean_y2 = mean_y2 as f32;
        self.event.mean_xy = mean_xy as f32;
        self.event.mean_x_system = mean_x_system as f32;
        self.event.mean_y_system = mean_y_system as f32;
        self.event.mean_x_a = mean_x_a as f32;
        self.event.mean_y_a = mean_y_a as f32;
        self.event.mean_x_b = mean_x_b as f32;
        self.event.mean_y_b = mean_y_b as f32;

        let npart = (npart_a + npart_b) as f64;
        if npart > 0.0 {
            let mean_x = self.event.mean_x as f64;
            let mean_y = self.event.mean_y as f64;
            let mut sinphi = [0.0; 10];
            let mut cosphi = [0.0; 10];
            let mut rn = [0.0; 10];

            for nucleon in nucleons_a {
                if !nucleon.is_wounded() {
                    continue;
                }
                let x = nucleon.x() - mean_x;
                let y = nucleon.y() - mean_y;
                let r = (x * x + y * y).sqrt();
                let phi = y.atan2(x);
                for n in 1..10 {
                    let w = if n == 1 { 3.0 } else { n as f64 };
                    let rw = r.powf(w);
                    cosphi[n] += rw * (n as f64 * phi).cos();
                    sinphi[n] += rw * (n as f64 * phi).sin();
                    rn[n] += rw;
                }
            }

            for nucleon in nucleons_b {
                if !nucleon.is_wounded() {
                    continue;
                }
                let x = nucleon.x() - mean_x;
                let y = nucleon.y() - mean_y;
                let r = (x * x + y * y).sqrt();
                let phi = y.atan2(x);
                for n in 1..10 {
                    let w = if n == 1 { 3.0 } else { n as f64 };
                    let rw = r.powf(w);
                    cosphi[n] += rw * (n as f64 * phi).cos();
                    sinphi[n] += rw * (n as f64 * phi).sin();
                    rn[n] += rw;
                }
            }

            for n in 1..6 {
                if rn[n] > 0.0 {
                    let psi = ((sinphi[n].atan2(cosphi[n])) + PI) / (n as f64);
                    let ecc = (sinphi[n] * sinphi[n] + cosphi[n] * cosphi[n]).sqrt() / rn[n];
                    match n {
                        1 => {
                            self.event.psi1 = psi as f32;
                            self.event.ecc1 = ecc as f32;
                        }
                        2 => {
                            self.event.psi2 = psi as f32;
                            self.event.ecc2 = ecc as f32;
                        }
                        3 => {
                            self.event.psi3 = psi as f32;
                            self.event.ecc3 = ecc as f32;
                        }
                        4 => {
                            self.event.psi4 = psi as f32;
                            self.event.ecc4 = ecc as f32;
                        }
                        5 => {
                            self.event.psi5 = psi as f32;
                            self.event.ecc5 = ecc as f32;
                        }
                        _ => {}
                    }
                }
            }
        }

        let npart_total = npart_a + npart_b;
        if npart_total > self.max_npart_found {
            self.max_npart_found = npart_total;
        }

        let w1 = self.nucleus_a.weight();
        let w2 = self.nucleus_b.weight();
        let w = (if w1 == 0.0 { 1.0 } else { w1 }) * (if w2 == 0.0 { 1.0 } else { w2 });
        self.event.weight = w as f32;
        self.total_events += w;
        self.events += 1.0;
    }

    /// Generate a single event with a given seed (for parallel generation)
    fn generate_event_with_seed(
        config: &GlauberConfig,
        seed: u64,
        bgen: Option<f64>,
    ) -> TGlauberEvent {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut nucleus_a = TGlauNucleus::new(&config.nucleus_a_name);
        let mut nucleus_b = TGlauNucleus::new(&config.nucleus_b_name);

        nucleus_a.set_min_dist(config.min_dist);
        nucleus_a.set_node_dist(config.node_dist);
        nucleus_b.set_min_dist(config.min_dist);
        nucleus_b.set_node_dist(config.node_dist);

        let b = match bgen {
            Some(b) if b >= 0.0 => b,
            _ => {
                let b2_range = config.bmax * config.bmax - config.bmin * config.bmin;
                (b2_range * rng.r#gen::<f64>() + config.bmin * config.bmin).sqrt()
            }
        };

        nucleus_a.throw_nucleons(-b / 2.0, &mut rng);
        nucleus_b.throw_nucleons(b / 2.0, &mut rng);

        let nucleons_a = nucleus_a.nucleons();
        let nucleons_b = nucleus_b.nucleons();
        let an = nucleons_a.len();
        let bn = nucleons_b.len();

        let mut event = TGlauberEvent::default();
        let mut bc = vec![vec![false; bn]; an];

        let mut nc = 0;
        let mut nh = 0;
        let mut bnn_sum = 0.0;
        let mut x0 = 0.0;
        let mut y0 = 0.0;
        let mut first_collision = true;

        let d2pp = config.xsect * MB_TO_FM2 / PI;
        let d2np = if config.xsect_np > 0.0 {
            config.xsect_np * MB_TO_FM2 / PI
        } else {
            d2pp
        };
        let bh = (d2pp * config.hard_frac).sqrt();

        for i in 0..bn {
            let nucleon_b = &nucleons_b[i];
            for j in 0..an {
                let nucleon_a = &nucleons_a[j];
                let dx = nucleon_b.x() - nucleon_a.x();
                let dy = nucleon_b.y() - nucleon_a.y();
                let dij = dx * dx + dy * dy;

                let is_pp = nucleon_a.is_proton() && nucleon_b.is_proton();
                let is_nn = nucleon_a.is_neutron() && nucleon_b.is_neutron();
                let d2 = if config.xsect_np > 0.0 && !is_pp && !is_nn {
                    d2np
                } else {
                    d2pp
                };

                if dij > d2 {
                    continue;
                }

                let bij = dij.sqrt();
                let mut collides = true;

                if let Some(profile) = &config.nn_profile {
                    let val = profile.eval(bij);
                    if rng.r#gen::<f64>() > val {
                        collides = false;
                    }
                }

                if collides {
                    bc[j][i] = true;
                    nc += 1;
                    bnn_sum += bij;

                    if bij < bh {
                        nh += 1;
                    }

                    if is_pp {
                        event.ncollpp += 1.0;
                    } else if is_nn {
                        event.ncollnn += 1.0;
                    } else {
                        event.ncollpn += 1.0;
                    }

                    if first_collision {
                        x0 = (nucleon_a.x() + nucleon_b.x()) / 2.0;
                        y0 = (nucleon_a.y() + nucleon_b.y()) / 2.0;
                        first_collision = false;
                    }
                }
            }
        }

        if nc == 0 {
            event.b = b as f32;
            return event;
        }

        for i in 0..bn {
            let nucleon_b = &mut nucleus_b.nucleons_mut()[i];
            let mut ncoll_b = 0;
            for j in 0..an {
                if bc[j][i] {
                    ncoll_b += 1;
                }
            }
            nucleon_b.set_n_coll(ncoll_b);
        }

        for j in 0..an {
            let nucleon_a = &mut nucleus_a.nucleons_mut()[j];
            let mut ncoll_a = 0;
            for i in 0..bn {
                if bc[j][i] {
                    ncoll_a += 1;
                }
            }
            nucleon_a.set_n_coll(ncoll_a);
        }

        event.b = b as f32;
        event.bnn = (bnn_sum / nc as f64) as f32;
        event.ncoll = nc as f32;
        event.nhard = nh as f32;
        event.x0 = x0 as f32;
        event.y0 = y0 as f32;

        // Calculate participant information
        let nucleons_a = nucleus_a.nucleons();
        let nucleons_b = nucleus_b.nucleons();

        let mut sum_w = 0.0;
        let mut sum_w_a = 0.0;
        let mut sum_w_b = 0.0;

        let mut npart_a = 0;
        let mut npart_b = 0;
        let mut npart0 = 0;
        let mut npart_an = 0;
        let mut npart_bn = 0;
        let mut npart0n = 0;
        let mut spec_a = 0;
        let mut spec_b = 0;

        let mut mean_x = 0.0;
        let mut mean_y = 0.0;
        let mut mean_x2 = 0.0;
        let mut mean_y2 = 0.0;
        let mut mean_xy = 0.0;
        let mut mean_x_system = 0.0;
        let mut mean_y_system = 0.0;
        let mut mean_x_a = 0.0;
        let mut mean_y_a = 0.0;
        let mut mean_x_b = 0.0;
        let mut mean_y_b = 0.0;

        for nucleon in nucleons_a {
            let x = nucleon.x();
            let y = nucleon.y();
            mean_x_system += x;
            mean_y_system += y;

            if nucleon.is_wounded() {
                let w = nucleon.get_2c_weight(config.two_c_x);
                let ncoll = nucleon.n_coll();

                npart_a += 1;
                if nucleon.is_neutron() {
                    npart_an += 1;
                }
                if ncoll == 1 {
                    npart0 += 1;
                    if nucleon.is_neutron() {
                        npart0n += 1;
                    }
                }

                sum_w += w;
                sum_w_a += 1.0;
                mean_x += x * w;
                mean_y += y * w;
                mean_x2 += x * x * w;
                mean_y2 += y * y * w;
                mean_xy += x * y * w;
                mean_x_a += x;
                mean_y_a += y;
            } else if nucleon.is_neutron() {
                spec_a += 1;
            }
        }

        for nucleon in nucleons_b {
            let x = nucleon.x();
            let y = nucleon.y();
            mean_x_system += x;
            mean_y_system += y;

            if nucleon.is_wounded() {
                let w = nucleon.get_2c_weight(config.two_c_x);
                let ncoll = nucleon.n_coll();

                npart_b += 1;
                if nucleon.is_neutron() {
                    npart_bn += 1;
                }
                if ncoll == 1 {
                    npart0 += 1;
                    if nucleon.is_neutron() {
                        npart0n += 1;
                    }
                }

                sum_w += w;
                sum_w_b += 1.0;
                mean_x += x * w;
                mean_y += y * w;
                mean_x2 += x * x * w;
                mean_y2 += y * y * w;
                mean_xy += x * y * w;
                mean_x_b += x;
                mean_y_b += y;
            } else if nucleon.is_neutron() {
                spec_b += 1;
            }
        }

        let total_n = (nucleons_a.len() + nucleons_b.len()) as f64;
        mean_x_system /= total_n;
        mean_y_system /= total_n;

        if sum_w > 0.0 {
            mean_x /= sum_w;
            mean_y /= sum_w;
            mean_x2 /= sum_w;
            mean_y2 /= sum_w;
            mean_xy /= sum_w;
        }

        if sum_w_a > 0.0 {
            mean_x_a /= sum_w_a;
            mean_y_a /= sum_w_a;
        }
        if sum_w_b > 0.0 {
            mean_x_b /= sum_w_b;
            mean_y_b /= sum_w_b;
        }

        let var_x = mean_x2 - mean_x * mean_x;
        let var_y = mean_y2 - mean_y * mean_y;
        let var_xy = mean_xy - mean_x * mean_y;
        let area_w = if var_x * var_y - var_xy * var_xy < 0.0 {
            -1.0
        } else {
            (var_x * var_y - var_xy * var_xy).sqrt()
        };

        event.npart = (npart_a + npart_b) as f32;
        event.npart_a = npart_a as f32;
        event.npart_b = npart_b as f32;
        event.npart0 = npart0 as f32;
        event.npart_an = npart_an as f32;
        event.npart_bn = npart_bn as f32;
        event.npart0n = npart0n as f32;
        event.spec_a = spec_a as f32;
        event.spec_b = spec_b as f32;
        event.var_x = var_x as f32;
        event.var_y = var_y as f32;
        event.var_xy = var_xy as f32;
        event.area_w = area_w as f32;
        event.mean_x = mean_x as f32;
        event.mean_y = mean_y as f32;
        event.mean_x2 = mean_x2 as f32;
        event.mean_y2 = mean_y2 as f32;
        event.mean_xy = mean_xy as f32;
        event.mean_x_system = mean_x_system as f32;
        event.mean_y_system = mean_y_system as f32;
        event.mean_x_a = mean_x_a as f32;
        event.mean_y_a = mean_y_a as f32;
        event.mean_x_b = mean_x_b as f32;
        event.mean_y_b = mean_y_b as f32;

        let npart = (npart_a + npart_b) as f64;
        if npart > 0.0 {
            let mean_x = event.mean_x as f64;
            let mean_y = event.mean_y as f64;
            let mut sinphi = [0.0; 10];
            let mut cosphi = [0.0; 10];
            let mut rn = [0.0; 10];

            for nucleon in nucleons_a {
                if !nucleon.is_wounded() {
                    continue;
                }
                let x = nucleon.x() - mean_x;
                let y = nucleon.y() - mean_y;
                let r = (x * x + y * y).sqrt();
                let phi = y.atan2(x);
                for n in 1..10 {
                    let w = if n == 1 { 3.0 } else { n as f64 };
                    let rw = r.powf(w);
                    cosphi[n] += rw * (n as f64 * phi).cos();
                    sinphi[n] += rw * (n as f64 * phi).sin();
                    rn[n] += rw;
                }
            }

            for nucleon in nucleons_b {
                if !nucleon.is_wounded() {
                    continue;
                }
                let x = nucleon.x() - mean_x;
                let y = nucleon.y() - mean_y;
                let r = (x * x + y * y).sqrt();
                let phi = y.atan2(x);
                for n in 1..10 {
                    let w = if n == 1 { 3.0 } else { n as f64 };
                    let rw = r.powf(w);
                    cosphi[n] += rw * (n as f64 * phi).cos();
                    sinphi[n] += rw * (n as f64 * phi).sin();
                    rn[n] += rw;
                }
            }

            for n in 1..6 {
                if rn[n] > 0.0 {
                    let psi = ((sinphi[n].atan2(cosphi[n])) + PI) / (n as f64);
                    let ecc = (sinphi[n] * sinphi[n] + cosphi[n] * cosphi[n]).sqrt() / rn[n];
                    match n {
                        1 => {
                            event.psi1 = psi as f32;
                            event.ecc1 = ecc as f32;
                        }
                        2 => {
                            event.psi2 = psi as f32;
                            event.ecc2 = ecc as f32;
                        }
                        3 => {
                            event.psi3 = psi as f32;
                            event.ecc3 = ecc as f32;
                        }
                        4 => {
                            event.psi4 = psi as f32;
                            event.ecc4 = ecc as f32;
                        }
                        5 => {
                            event.psi5 = psi as f32;
                            event.ecc5 = ecc as f32;
                        }
                        _ => {}
                    }
                }
            }
        }

        let w1 = nucleus_a.weight();
        let w2 = nucleus_b.weight();
        event.weight =
            ((if w1 == 0.0 { 1.0 } else { w1 }) * (if w2 == 0.0 { 1.0 } else { w2 })) as f32;

        event
    }

    /// Generate events in parallel using all available CPU cores
    pub fn run_parallel(&mut self, nevents: i32, b: Option<f64>) -> Vec<TGlauberEvent> {
        let num_threads = rayon::current_num_threads();
        let nevents_per_thread = (nevents / num_threads as i32).max(1);
        let remaining = nevents - nevents_per_thread * num_threads as i32;

        println!("╔════════════════════════════════════════════════════════════════╗");
        println!("║                    PARALLEL EVENT GENERATION                   ║");
        println!("╠════════════════════════════════════════════════════════════════╣");
        println!(
            "║  Total events:     {:>8}                                    ║",
            nevents
        );
        println!(
            "║  CPU threads:      {:>8}                                    ║",
            num_threads
        );
        println!(
            "║  Events per thread: {:>8}                                   ║",
            nevents_per_thread
        );
        if remaining > 0 {
            println!(
                "║  Extra events:     {:>8} (distributed to first {} threads)  ║",
                remaining, remaining
            );
        }
        println!("╚════════════════════════════════════════════════════════════════╝");
        println!();

        self.config.bmin = self.bmin;
        self.config.bmax = self.bmax;
        self.config.min_dist = self.nucleus_a.min_dist();
        self.config.node_dist = self.nucleus_a.node_dist();
        self.config.omega = self.omega;
        self.config.two_c_x = self.two_c_x;
        self.config.hard_frac = self.hard_frac;
        self.config.nn_profile = self.nn_profile.clone();

        let config = Arc::new(self.config.clone());

        use std::sync::atomic::{AtomicUsize, Ordering};
        let progress = AtomicUsize::new(0);
        let total_events = nevents as usize;

        let start_time = Instant::now();

        let all_events: Vec<TGlauberEvent> = (0..num_threads)
            .into_par_iter()
            .flat_map(|thread_id| {
                let events_this_thread = if thread_id < remaining as usize {
                    nevents_per_thread + 1
                } else {
                    nevents_per_thread
                };

                if events_this_thread <= 0 {
                    return Vec::new();
                }

                let mut thread_events = Vec::with_capacity(events_this_thread as usize);
                let base_seed = thread_id as u64 * 1000000 + 42;

                for i in 0..events_this_thread {
                    let seed = base_seed + i as u64;
                    let mut event = TGlauberMC::generate_event_with_seed(&config, seed, b);

                    let mut attempts = 0;
                    while event.ncoll == 0.0 && attempts < 100 {
                        let new_seed = seed + 1000000 + attempts as u64;
                        event = TGlauberMC::generate_event_with_seed(&config, new_seed, b);
                        attempts += 1;
                    }

                    thread_events.push(event);

                    let done = progress.fetch_add(1, Ordering::Relaxed) + 1;
                    if done % 100 == 0 || done == total_events {
                        let elapsed = start_time.elapsed();
                        let percent = (done as f64 / total_events as f64) * 100.0;
                        let rate = done as f64 / elapsed.as_secs_f64();
                        println!("  Progress: {:6}/{} events ({:5.1}%) | Rate: {:8.1} events/sec | Elapsed: {:?}",
                            done, total_events, percent, rate, elapsed);
                    }
                }

                thread_events
            })
            .collect();

        let elapsed = start_time.elapsed();
        let rate = total_events as f64 / elapsed.as_secs_f64();

        let total_weight: f64 = all_events.iter().map(|e| e.weight as f64).sum();
        self.total_events = total_weight;
        self.events = all_events.len() as f64;

        for event in &all_events {
            if event.npart as i32 > self.max_npart_found {
                self.max_npart_found = event.npart as i32;
            }
        }

        if let Some(last) = all_events.last() {
            self.event = last.clone();
        }

        println!();
        println!("╔════════════════════════════════════════════════════════════════╗");
        println!("║                    GENERATION COMPLETE                         ║");
        println!("╠════════════════════════════════════════════════════════════════╣");
        println!(
            "║  Total events:     {:>8}                                    ║",
            total_events
        );
        println!(
            "║  Time elapsed:     {:>8.2?}                                    ║",
            elapsed
        );
        println!(
            "║  Event rate:       {:>8.1} events/sec                         ║",
            rate
        );
        println!(
            "║  Threads used:     {:>8}                                    ║",
            num_threads
        );
        println!("╚════════════════════════════════════════════════════════════════╝");
        println!();

        all_events
    }

    /// Run events (single-threaded, with automatic parallel for large counts)
    pub fn run(&mut self, nevents: i32, rng: &mut ThreadRng, b: Option<f64>) -> Vec<TGlauberEvent> {
        if nevents > 10000 {
            println!("⚠️  Large event count detected ({})", nevents);
            println!("🔄 Switching to parallel mode using all available CPU cores...");
            println!();
            return self.run_parallel(nevents, b);
        }

        let num_threads = rayon::current_num_threads();
        println!("╔════════════════════════════════════════════════════════════════╗");
        println!("║                  SINGLE-THREADED GENERATION                   ║");
        println!("╠════════════════════════════════════════════════════════════════╣");
        println!(
            "║  Total events:     {:>8}                                    ║",
            nevents
        );
        println!("║  Mode:              Single-threaded                            ║");
        println!("║  (Use >10000 events for automatic parallel mode)               ║");
        println!(
            "║  CPU threads available: {:>8}                                    ║",
            num_threads
        );
        println!("╚════════════════════════════════════════════════════════════════╝");
        println!();

        let mut events = Vec::with_capacity(nevents as usize);
        let start_time = Instant::now();

        for i in 0..nevents {
            while !self.next_event(rng, b) {}
            events.push(self.event.clone());

            if i > 0 && i % 100 == 0 {
                let elapsed = start_time.elapsed();
                let percent = (i as f64 / nevents as f64) * 100.0;
                let rate = i as f64 / elapsed.as_secs_f64();
                println!(
                    "  Progress: {:6}/{} events ({:5.1}%) | Rate: {:8.1} events/sec | Elapsed: {:?}",
                    i, nevents, percent, rate, elapsed
                );
            }
        }

        let elapsed = start_time.elapsed();
        let rate = nevents as f64 / elapsed.as_secs_f64();

        println!();
        println!("╔════════════════════════════════════════════════════════════════╗");
        println!("║                    GENERATION COMPLETE                        ║");
        println!("╠════════════════════════════════════════════════════════════════╣");
        println!(
            "║  Total events:     {:>8}                                    ║",
            nevents
        );
        println!(
            "║  Time elapsed:     {:>8.2?}                                    ║",
            elapsed
        );
        println!(
            "║  Event rate:       {:>8.1} events/sec                         ║",
            rate
        );
        println!("║  Mode:              Single-threaded                            ║");
        println!("╚════════════════════════════════════════════════════════════════╝");
        println!();

        events
    }

    pub fn run_save<F>(
        &mut self,
        nevents: i32,
        rng: &mut ThreadRng,
        b: Option<f64>,
        mut callback: F,
    ) where
        F: FnMut(&TGlauberEvent, i32),
    {
        if nevents > 10000 {
            println!("⚠️  Large event count detected ({})", nevents);
            println!("🔄 Switching to parallel mode using all available CPU cores...");
            println!();
            let events = self.run_parallel(nevents, b);
            for (i, event) in events.iter().enumerate() {
                callback(event, i as i32);
            }
            return;
        }

        let num_threads = rayon::current_num_threads();
        println!("╔════════════════════════════════════════════════════════════════╗");
        println!("║                  SINGLE-THREADED GENERATION                   ║");
        println!("╠════════════════════════════════════════════════════════════════╣");
        println!(
            "║  Total events:     {:>8}                                    ║",
            nevents
        );
        println!("║  Mode:              Single-threaded                            ║");
        println!("║  (Use >10000 events for automatic parallel mode)               ║");
        println!(
            "║  CPU threads available: {:>8}                                    ║",
            num_threads
        );
        println!("╚════════════════════════════════════════════════════════════════╝");
        println!();

        let start_time = Instant::now();

        for i in 0..nevents {
            while !self.next_event(rng, b) {}
            callback(&self.event, i);

            if i > 0 && i % 100 == 0 {
                let elapsed = start_time.elapsed();
                let percent = (i as f64 / nevents as f64) * 100.0;
                let rate = i as f64 / elapsed.as_secs_f64();
                println!(
                    "  Progress: {:6}/{} events ({:5.1}%) | Rate: {:8.1} events/sec | Elapsed: {:?}",
                    i, nevents, percent, rate, elapsed
                );
            }
        }

        let elapsed = start_time.elapsed();
        let rate = nevents as f64 / elapsed.as_secs_f64();

        println!();
        println!("╔════════════════════════════════════════════════════════════════╗");
        println!("║                    GENERATION COMPLETE                        ║");
        println!("╠════════════════════════════════════════════════════════════════╣");
        println!(
            "║  Total events:     {:>8}                                    ║",
            nevents
        );
        println!(
            "║  Time elapsed:     {:>8.2?}                                    ║",
            elapsed
        );
        println!(
            "║  Event rate:       {:>8.1} events/sec                         ║",
            rate
        );
        println!("║  Mode:              Single-threaded                            ║");
        println!("╚════════════════════════════════════════════════════════════════╝");
        println!();
    }

    pub fn str(&self) -> String {
        format!(
            "TGlauberMC_{}_{}_snn{:.1}_md{:.1}_om{:.1}_rc{}_smax{}",
            self.nucleus_a.name(),
            self.nucleus_b.name(),
            self.xsect,
            self.nucleus_a.min_dist(),
            self.omega,
            self.nucleus_a.recenter(),
            self.nucleus_a.smax()
        )
    }
}
