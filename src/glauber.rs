// src/glauber.rs
use crate::constants::{MB_TO_FM2, PI, TWO_PI};
use crate::cross_section::CrossSection;
use crate::nucleon::TGlauNucleon;
use crate::nucleus::{NucleusError, TGlauNucleus};
use crate::profile::{NNProfile, profile_from_omega};
use rand::Rng;
use rand::RngExt;
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

/// Glauber-Gribov fluctuating nucleon-nucleon cross section, matching the `fPTot` TF1
/// built in the C++ constructor when `xsectsigma>0`:
/// f(x) = ((x/lambda)/((x/lambda)+sigma)) * exp(-(((x/lambda)/sigma-1)^2)/omega^2) / lambda
#[derive(Debug, Clone)]
pub struct FluctuatingXSect {
    sigma: f64,
    omega: f64,
    lambda: f64,
    envelope: f64,
}

impl FluctuatingXSect {
    const UPPER: f64 = 300.0;

    fn shape(x: f64, sigma: f64, omega: f64, lambda: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        let t = x / lambda;
        (t / (t + sigma)) * (-((t / sigma - 1.0).powi(2) / (omega * omega))).exp() / lambda
    }

    /// Grid mean of `shape(x, sigma, omega, lambda)` over [0, UPPER] using 1000 bin centers,
    /// matching `TF1::SetNpx(1000)` + `TH1::GetMean()`.
    fn grid_mean(sigma: f64, omega: f64, lambda: f64) -> f64 {
        const NPX: usize = 1000;
        let dx = Self::UPPER / NPX as f64;
        let mut sum_fx = 0.0;
        let mut sum_f = 0.0;
        for i in 0..NPX {
            let x = (i as f64 + 0.5) * dx;
            let f = Self::shape(x, sigma, omega, lambda);
            sum_fx += x * f;
            sum_f += f;
        }
        if sum_f > 0.0 { sum_fx / sum_f } else { 0.0 }
    }

    /// Calibrate lambda so that the distribution's mean equals `sigma`, matching the
    /// C++ constructor's `fXSectLambda = fXSect/fPTot->GetHistogram()->GetMean()` step.
    pub fn new(sigma: f64, omega: f64) -> Self {
        let mean_at_1 = Self::grid_mean(sigma, omega, 1.0);
        let lambda = if mean_at_1 > 0.0 { sigma / mean_at_1 } else { 1.0 };

        const GRID: usize = 1000;
        let mut peak = 0.0f64;
        for i in 0..=GRID {
            let x = Self::UPPER * i as f64 / GRID as f64;
            let v = Self::shape(x, sigma, omega, lambda);
            if v.is_finite() && v > peak {
                peak = v;
            }
        }
        let envelope = if peak > 0.0 { peak * 1.05 } else { 0.0 };

        Self {
            sigma,
            omega,
            lambda,
            envelope,
        }
    }

    /// The achieved mean of the calibrated distribution (should be very close to `sigma`).
    pub fn mean(&self) -> f64 {
        Self::grid_mean(self.sigma, self.omega, self.lambda)
    }

    pub fn eval(&self, x: f64) -> f64 {
        Self::shape(x, self.sigma, self.omega, self.lambda)
    }

    /// Draw a fluctuating cross-section value via grid-envelope rejection sampling,
    /// matching `fPTot->GetRandom()`.
    pub fn sample<R: Rng>(&self, rng: &mut R) -> f64 {
        if self.envelope <= 0.0 {
            return self.sigma;
        }
        for _ in 0..100_000 {
            let x = rng.random::<f64>() * Self::UPPER;
            let u = rng.random::<f64>() * self.envelope;
            if u < self.eval(x) {
                return x;
            }
        }
        self.sigma
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
    do_aagg: bool,
    shadow: bool,
    sig_h: f64,
    nn_profile: Option<NNProfile>,
    xsect_fluct: Option<FluctuatingXSect>,
}

/// Parameters needed to simulate one event, shared between the single-threaded
/// (`TGlauberMC::calc_event`) and parallel (`generate_event_with_seed`) code paths.
struct EventParams<'a> {
    xsect: f64,
    xsect_np: f64,
    hard_frac: f64,
    two_c_x: f64,
    do_core: bool,
    sig_h: f64,
    shadow: bool,
    calc_area: bool,
    calc_length: bool,
    do_aagg: bool,
    nn_profile: Option<&'a NNProfile>,
    xsect_fluct: Option<&'a FluctuatingXSect>,
}

/// Simulate one Glauber event on already-thrown nuclei, mirroring
/// `TGlauberMC::CalcEvent`+`CalcResults`. Returns (event, binary-collision matrix,
/// per-multiplicity MPI counts, event weight `w` - valid whether or not the event
/// succeeded, needed for total-cross-section bookkeeping). `event.ncoll > 0.0` signals
/// success (a `kFALSE`-equivalent CalcEvent returns event.ncoll == 0).
fn simulate_event<R: Rng>(
    params: &EventParams,
    nucleus_a: &mut TGlauNucleus,
    nucleus_b: &mut TGlauNucleus,
    bgen: f64,
    rng: &mut R,
) -> (TGlauberEvent, Vec<Vec<bool>>, [i32; 99], f64) {
    let an = nucleus_a.nucleons().len();
    let bn = nucleus_b.nucleons().len();

    // Per-event (and, if fDoAAGG, per-nucleon) fluctuating cross section draws.
    let xsect_event = match params.xsect_fluct {
        Some(fluct) => fluct.sample(rng),
        None => params.xsect,
    };
    let (xsec_a, xsec_b): (Vec<f64>, Vec<f64>) = match params.xsect_fluct {
        Some(fluct) if params.do_aagg => (
            (0..an).map(|_| fluct.sample(rng)).collect(),
            (0..bn).map(|_| fluct.sample(rng)).collect(),
        ),
        _ => (Vec::new(), Vec::new()),
    };

    let d2pp = xsect_event * MB_TO_FM2 / PI;
    let d2np = if params.xsect_np > 0.0 {
        params.xsect_np * MB_TO_FM2 / PI
    } else {
        d2pp
    };
    let bh = (d2pp * params.hard_frac).sqrt();

    // Baseline ball-diameter cutoff: when an NN profile is active, C++ overrides d2
    // with the profile's own declared range (xmax^2) *before* the per-pair loop, so
    // the geometric pre-filter doesn't reject pairs the profile itself would still
    // accept. This baseline is itself overridden per-pair below when xsect_np>0 or
    // the Glauber-Gribov per-nucleon fluctuation is active (matching CalcEvent).
    let d2_base = match params.nn_profile {
        Some(profile) => {
            let xmax = profile.max_b();
            xmax * xmax
        }
        None => d2pp,
    };

    let sig_s = xsect_event;
    let sig_hs = if params.shadow {
        let rrb = (bgen * bgen / 35.2 / 1.44).min(1.0);
        let aphx = 0.1 * (4.0 / 3.0) * 4.92 * (1.0 - rrb).sqrt();
        params.sig_h - aphx * 103.65
    } else {
        params.sig_h
    };

    let mut event = TGlauberEvent::default();
    let mut bc = vec![vec![false; bn]; an];
    let mut mpi = [0i32; 99];

    let mut nc = 0i32;
    let mut nh = 0i32;
    let mut njet = 0i32;
    let mut bnn_sum = 0.0;
    let mut x0 = 0.0;
    let mut y0 = 0.0;
    let mut first_collision = true;

    {
        let nucleons_a = nucleus_a.nucleons();
        let nucleons_b = nucleus_b.nucleons();

        for i in 0..bn {
            let nucleon_b = &nucleons_b[i];
            let t_b = nucleon_b.is_proton();
            for j in 0..an {
                let nucleon_a = &nucleons_a[j];
                let t_a = nucleon_a.is_proton();
                let dx = nucleon_b.x() - nucleon_a.x();
                let dy = nucleon_b.y() - nucleon_a.y();
                let dij = dx * dx + dy * dy;

                let d2 = if params.xsect_np > 0.0 {
                    if t_a != t_b { d2np } else { d2pp }
                } else if params.do_aagg && !xsec_a.is_empty() {
                    0.5 * (xsec_a[j] + xsec_b[i]) * MB_TO_FM2 / PI
                } else {
                    d2_base
                };

                if dij > d2 {
                    continue;
                }
                let bij = dij.sqrt();

                if let Some(profile) = params.nn_profile {
                    let val = profile.eval(bij);
                    let ran: f64 = rng.random();
                    if ran > val {
                        continue;
                    }
                    // Second-stage MPI veto/sampling, from HIJING.
                    let ts = 2.0 * val;
                    let es = (-ts).exp();
                    let tt = ts * sig_hs / sig_s;
                    let et = (-tt).exp();
                    if ran < et * (1.0 - es) {
                        mpi[0] += 1;
                        continue;
                    }
                    let u: f64 = rng.random();
                    let mut xr = -(et + u * (1.0 - et)).ln();
                    let mut nj = 0i32;
                    // C++ only checks `nj>99` *after* this loop (as an error to abort
                    // the whole process on), so a degenerate `tt` (e.g. NaN from a
                    // profile parameter at a boundary value like TRENTO w=0) would hang
                    // it too. Bound the loop here instead of hanging.
                    loop {
                        nj += 1;
                        let u2: f64 = rng.random();
                        xr -= u2.ln();
                        if xr > tt || nj >= 99 {
                            break;
                        }
                    }
                    let nj_idx = nj.min(98) as usize;
                    mpi[nj_idx] += 1;
                    njet += nj;
                    event.nmpi = njet as f32;
                }

                bc[j][i] = true;
                nc += 1;
                bnn_sum += bij;
                if bij < bh {
                    nh += 1;
                }
                if t_a != t_b {
                    event.ncollpn += 1.0;
                } else if t_a {
                    event.ncollpp += 1.0;
                } else {
                    event.ncollnn += 1.0;
                }
                if first_collision {
                    x0 = (nucleon_a.x() + nucleon_b.x()) / 2.0;
                    y0 = (nucleon_a.y() + nucleon_b.y()) / 2.0;
                    first_collision = false;
                }
            }
        }
    }

    event.b = bgen as f32;

    let w1 = nucleus_a.weight();
    let w2 = nucleus_b.weight();
    let w = (if w1 == 0.0 { 1.0 } else { w1 }) * (if w2 == 0.0 { 1.0 } else { w2 });
    if w.abs() > 1e6 {
        println!(
            "Warning: Weight is too large: {} (w1={}, w2={}, A={}, B={})",
            w,
            w1,
            w2,
            nucleus_a.name(),
            nucleus_b.name()
        );
    }

    if nc == 0 {
        return (event, bc, mpi, w);
    }

    event.weight = w as f32;
    event.ncoll = nc as f32;
    event.nhard = nh as f32;
    event.bnn = (bnn_sum / nc as f64) as f32;
    event.x0 = x0 as f32;
    event.y0 = y0 as f32;

    for i in 0..bn {
        let mut ncoll_b = 0;
        for j in 0..an {
            if bc[j][i] {
                ncoll_b += 1;
            }
        }
        nucleus_b.nucleons_mut()[i].set_n_coll(ncoll_b);
    }
    for j in 0..an {
        let mut ncoll_a = 0;
        for i in 0..bn {
            if bc[j][i] {
                ncoll_a += 1;
            }
        }
        nucleus_a.nucleons_mut()[j].set_n_coll(ncoll_a);
    }

    calc_participants(params, nucleus_a, nucleus_b, &mut event);

    if event.npart > 0.0 {
        if params.calc_area {
            let (area_o, area_a) = calc_area_overlap(
                nucleus_a,
                nucleus_b,
                if params.do_core { 1 } else { 0 },
                xsect_event,
                event.mean_x as f64,
                event.mean_y as f64,
            );
            event.area_o = area_o;
            event.area_a = area_a;
        }
        if params.calc_length {
            calc_length(nucleus_a, nucleus_b, xsect_event, x0, y0, &mut event, rng);
        }
    }

    (event, bc, mpi, w)
}

/// Compute participant/moment/eccentricity quantities for an event that had at least one
/// collision, mirroring the second half of `TGlauberMC::CalcResults`.
fn calc_participants(
    params: &EventParams,
    nucleus_a: &TGlauNucleus,
    nucleus_b: &TGlauNucleus,
    event: &mut TGlauberEvent,
) {
    let nucleons_a = nucleus_a.nucleons();
    let nucleons_b = nucleus_b.nucleons();

    let mut sum_w = 0.0;
    // sum_w_a is intentionally never accumulated: in the C++ reference `sumWA` is
    // declared but never incremented, so MeanXA/MeanYA always finalize to 0 below.
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
    let mut mean_x_b = 0.0;
    let mut mean_y_b = 0.0;

    for nucleon in nucleons_a {
        let x = nucleon.x();
        let y = nucleon.y();
        mean_x_system += x;
        mean_y_system += y;

        if nucleon.is_wounded() {
            let w = nucleon.get_2c_weight(params.two_c_x);
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
            mean_x += x * w;
            mean_y += y * w;
            mean_x2 += x * x * w;
            mean_y2 += y * y * w;
            mean_xy += x * y * w;
        } else if nucleon.is_neutron() {
            spec_a += 1;
        }
    }

    for nucleon in nucleons_b {
        let x = nucleon.x();
        let y = nucleon.y();
        mean_x_system += x;
        mean_y_system += y;
        mean_x_b += x;
        mean_y_b += y;

        if nucleon.is_wounded() {
            let w = nucleon.get_2c_weight(params.two_c_x);
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
            sum_w_b += w;
            mean_x += x * w;
            mean_y += y * w;
            mean_x2 += x * x * w;
            mean_y2 += y * y * w;
            mean_xy += x * y * w;
        } else if nucleon.is_neutron() {
            spec_b += 1;
        }
    }

    let total_n = (nucleons_a.len() + nucleons_b.len()) as f64;
    if total_n > 0.0 {
        mean_x_system /= total_n;
        mean_y_system /= total_n;
    }

    if sum_w > 0.0 {
        mean_x /= sum_w;
        mean_y /= sum_w;
        mean_x2 /= sum_w;
        mean_y2 /= sum_w;
        mean_xy /= sum_w;
    } else {
        mean_x = 0.0;
        mean_y = 0.0;
        mean_x2 = 0.0;
        mean_y2 = 0.0;
        mean_xy = 0.0;
    }

    // Bug-compatible with the C++ reference: MeanXA/MeanYA always finalize to 0 (see
    // note on sum_w_a above). MeanXB/MeanYB use a numerator summed over *all* B
    // nucleons but a denominator (sum_w_b) summed over *wounded* B nucleons only -
    // an odd, mismatched normalization that's nonetheless what the reference does.
    let mean_x_a = 0.0;
    let mean_y_a = 0.0;
    if sum_w_b > 0.0 {
        mean_x_b /= sum_w_b;
        mean_y_b /= sum_w_b;
    } else {
        mean_x_b = 0.0;
        mean_y_b = 0.0;
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

    let k_nc = if params.do_core { 1 } else { 0 };
    let npart = (npart_a + npart_b) as f64;
    if npart > 0.0 {
        let mut sinphi = [0.0; 10];
        let mut cosphi = [0.0; 10];
        let mut rn = [0.0; 10];

        for nucleon in nucleons_a.iter().chain(nucleons_b.iter()) {
            if nucleon.n_coll() <= k_nc {
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
                let psi = (sinphi[n].atan2(cosphi[n]) + PI) / (n as f64);
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
}

/// Overlap area calculation (opt-in via `SetCalcArea`), mirroring the grid-stamping done
/// inline in `TGlauberMC::CalcResults`. Returns (AreaO, AreaA).
fn calc_area_overlap(
    nucleus_a: &TGlauNucleus,
    nucleus_b: &TGlauNucleus,
    k_nc: i32,
    xsect_event: f64,
    mean_x: f64,
    mean_y: f64,
) -> (f32, f32) {
    const NBINS: usize = 200;
    const ELL: f64 = 10.0;
    let bin_width = 2.0 * ELL / NBINS as f64;
    let da = bin_width * bin_width;
    let d2 = xsect_event * MB_TO_FM2 / PI;
    let r2 = d2 / 4.0;

    let stamp = |nucleons: &[TGlauNucleon]| -> Vec<bool> {
        let mut grid = vec![false; NBINS * NBINS];
        for nucleon in nucleons {
            if nucleon.n_coll() <= k_nc {
                continue;
            }
            let x = nucleon.x() - mean_x;
            let y = nucleon.y() - mean_y;
            for xi in 0..NBINS {
                let cx = -ELL + (xi as f64 + 0.5) * bin_width;
                let dx = x - cx;
                if dx * dx > r2 {
                    continue;
                }
                for yi in 0..NBINS {
                    let idx = xi * NBINS + yi;
                    if grid[idx] {
                        continue;
                    }
                    let cy = -ELL + (yi as f64 + 0.5) * bin_width;
                    let dy = y - cy;
                    if dx * dx + dy * dy < r2 {
                        grid[idx] = true;
                    }
                }
            }
        }
        grid
    };

    let area_a = stamp(nucleus_a.nucleons());
    let area_b = stamp(nucleus_b.nucleons());

    let mut overlap1 = 0usize;
    let mut overlap2 = 0usize;
    for i in 0..NBINS * NBINS {
        let (va, vb) = (area_a[i], area_b[i]);
        if va && vb {
            overlap1 += 1;
        }
        if va || vb {
            overlap2 += 1;
        }
    }
    ((overlap1 as f64 * da) as f32, (overlap2 as f64 * da) as f32)
}

/// Radial density used by the path-length calculation, matching the `rad` TF1 in
/// `TGlauberMC::CalcResults` (`2*pi/sigma^2*exp(-r^2/(2*sigma^2))`).
fn rad_density(r: f64, sigma: f64) -> f64 {
    2.0 * PI / (sigma * sigma) * (-(r * r) / (2.0 * sigma * sigma)).exp()
}

/// Sum of `rad_density` over all wounded nucleons within range, matching `TGlauberMC::CalcDens`.
fn calc_dens(
    nucleus_a: &TGlauNucleus,
    nucleus_b: &TGlauNucleus,
    sigma: f64,
    rmax: f64,
    xval: f64,
    yval: f64,
) -> f64 {
    let r2max = rmax * rmax;
    let mut ret = 0.0;
    for nucleon in nucleus_a.nucleons().iter().chain(nucleus_b.nucleons().iter()) {
        if !nucleon.is_wounded() {
            continue;
        }
        let dx = xval - nucleon.x();
        let dy = yval - nucleon.y();
        let r2 = dx * dx + dy * dy;
        if r2 > r2max {
            continue;
        }
        ret += rad_density(r2.sqrt(), sigma);
    }
    ret
}

/// Path-length calculation (opt-in via `SetCalcLength`), mirroring the ray-marching loop
/// in `TGlauberMC::CalcResults`.
fn calc_length<R: Rng>(
    nucleus_a: &TGlauNucleus,
    nucleus_b: &TGlauNucleus,
    xsect_event: f64,
    x0: f64,
    y0: f64,
    event: &mut TGlauberEvent,
    rng: &mut R,
) {
    let krhs = (xsect_event / 40.0 / PI).sqrt();
    let ksg = krhs / (5.0f64).sqrt();
    let kdl = 0.1;
    let rmax = 5.0 * ksg;
    let minval = rad_density(rmax, ksg);

    let phi0 = rng.random::<f64>() * TWO_PI;
    event.phi0 = phi0 as f32;
    let (sphi0, cphi0) = phi0.sin_cos();

    let mut x = x0;
    let mut y = y0;
    let mut i0a = 0.0;
    let mut i1a = 0.0;
    let mut l = 0.0;
    let mut val = calc_dens(nucleus_a, nucleus_b, ksg, rmax, x, y);
    while val > minval {
        x += kdl * cphi0;
        y += kdl * sphi0;
        i0a += val;
        i1a += l * val;
        l += kdl;
        val = calc_dens(nucleus_a, nucleus_b, ksg, rmax, x, y);
    }
    event.length = if i0a > 0.0 { (2.0 * i1a / i0a) as f32 } else { 0.0 };
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
    do_aagg: bool,
    shadow: bool,
    sig_h: f64,
    omega: f64,
    max_npart_found: i32,
    two_c_x: f64,
    events: f64,
    total_events: f64,
    nn_profile: Option<NNProfile>,
    xsect_fluct: Option<FluctuatingXSect>,
    event: TGlauberEvent,
    bc: Vec<Vec<bool>>,
    mpi: [i32; 99],
    config: GlauberConfig,
}

impl TGlauberMC {
    /// Fails if `na` or `nb` is not a nucleus defined in `data/nuclei.ron`.
    pub fn new(
        na: &str,
        nb: &str,
        xsect: f64,
        xsect_sigma: f64,
        xsect_np: f64,
    ) -> Result<Self, NucleusError> {
        let nucleus_a = TGlauNucleus::new(na)?;
        let nucleus_b = TGlauNucleus::new(nb)?;
        let mut xsect_use = xsect;
        let mut xsect_np_use = xsect_np;
        // Matches the C++ member-initializer-list default (fSigH(129.4)); only
        // overridden below when operating in energy mode (xsect<0).
        let mut sig_h = 129.4;

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

        let xsect_fluct = if xsect_sigma > 0.0 {
            let fluct = FluctuatingXSect::new(xsect_use, xsect_sigma);
            println!(
                "Using fluctuating cross section with <sigma>={:.3}mb, using fXSectOmega={} and lambda={}",
                fluct.mean(),
                xsect_sigma,
                fluct.lambda
            );
            Some(fluct)
        } else {
            None
        };

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
            do_aagg: true,
            shadow: false,
            sig_h,
            nn_profile: None,
            xsect_fluct: xsect_fluct.clone(),
        };

        Ok(Self {
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
            do_aagg: true,
            shadow: false,
            sig_h,
            omega: -1.0,
            max_npart_found: 0,
            two_c_x: 0.0,
            events: 0.0,
            total_events: 0.0,
            nn_profile: None,
            xsect_fluct,
            event: TGlauberEvent::default(),
            bc: Vec::new(),
            mpi: [0; 99],
            config,
        })
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

    pub fn set_calc_aagg(&mut self, calc: bool) {
        self.do_aagg = calc;
        self.config.do_aagg = calc;
    }

    pub fn set_shadowing(&mut self, shadow: bool) {
        self.shadow = shadow;
        self.config.shadow = shadow;
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

    /// Check if there was a binary collision between nucleon `j` of A and `i` of B.
    pub fn is_bc(&self, j: usize, i: usize) -> bool {
        self.bc.get(j).and_then(|row| row.get(i)).copied().unwrap_or(false)
    }

    /// Get the number of sub-collisions with `n` MPI for the current event.
    pub fn get_mpi(&self, n: usize) -> i32 {
        self.mpi.get(n).copied().unwrap_or(0)
    }

    /// Total (geometric) cross section in mb, from the fraction of thrown impact
    /// parameters that produced at least one collision. Same statistics/formula as
    /// `TGlauberMC::GetTotXSect` (`(fEvents/fTotalEvents)*Pi()*fBmax*fBmax/100`), which
    /// returns barn - converted to mb here (matching every other cross section quantity
    /// in this codebase, e.g. `xsect`/`xsect_np`) the same way the C++ reference itself
    /// does at its own "mb"-labeled call sites (`GetTotXSect()*1e3`, see `Run()`).
    pub fn total_xsect(&self) -> f64 {
        if self.total_events == 0.0 {
            return 0.0;
        }
        (self.events / self.total_events) * PI * self.bmax * self.bmax / 100.0 * 1000.0
    }

    /// Statistical error on `total_xsect`, in mb. Same formula as
    /// `TGlauberMC::GetTotXSectErr`; scales with `total_xsect` so it stays in mb too.
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
                (b2_range * rng.random::<f64>() + self.bmin * self.bmin).sqrt()
            }
        };

        self.nucleus_a.throw_nucleons(-b / 2.0, rng);
        self.nucleus_b.throw_nucleons(b / 2.0, rng);

        self.calc_event(rng, b)
    }

    /// Calculate event results
    fn calc_event(&mut self, rng: &mut ThreadRng, bgen: f64) -> bool {
        let params = EventParams {
            xsect: self.xsect,
            xsect_np: self.xsect_np,
            hard_frac: self.hard_frac,
            two_c_x: self.two_c_x,
            do_core: self.do_core,
            sig_h: self.sig_h,
            shadow: self.shadow,
            calc_area: self.calc_area,
            calc_length: self.calc_length,
            do_aagg: self.do_aagg,
            nn_profile: self.nn_profile.as_ref(),
            xsect_fluct: self.xsect_fluct.as_ref(),
        };

        let (event, bc, mpi, w) =
            simulate_event(&params, &mut self.nucleus_a, &mut self.nucleus_b, bgen, rng);

        self.bc = bc;
        self.mpi = mpi;
        let success = event.ncoll > 0.0;
        self.event = event;

        self.total_events += w;
        if success {
            self.events += w;
            if self.event.npart as i32 > self.max_npart_found {
                self.max_npart_found = self.event.npart as i32;
            }
        }

        success
    }

    /// Generate a single event with a given seed (for parallel generation).
    /// Returns the event and the `w` weight used for total-cross-section bookkeeping
    /// (valid whether or not the event succeeded).
    fn generate_event_with_seed(
        config: &GlauberConfig,
        seed: u64,
        bgen: Option<f64>,
    ) -> (TGlauberEvent, f64) {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        // Names were already validated in `TGlauberMC::new`.
        let mut nucleus_a =
            TGlauNucleus::new(&config.nucleus_a_name).expect("nucleus A validated in new()");
        let mut nucleus_b =
            TGlauNucleus::new(&config.nucleus_b_name).expect("nucleus B validated in new()");

        nucleus_a.set_min_dist(config.min_dist);
        nucleus_a.set_node_dist(config.node_dist);
        nucleus_b.set_min_dist(config.min_dist);
        nucleus_b.set_node_dist(config.node_dist);

        let b = match bgen {
            Some(b) if b >= 0.0 => b,
            _ => {
                let b2_range = config.bmax * config.bmax - config.bmin * config.bmin;
                (b2_range * rng.random::<f64>() + config.bmin * config.bmin).sqrt()
            }
        };

        nucleus_a.throw_nucleons(-b / 2.0, &mut rng);
        nucleus_b.throw_nucleons(b / 2.0, &mut rng);

        let params = EventParams {
            xsect: config.xsect,
            xsect_np: config.xsect_np,
            hard_frac: config.hard_frac,
            two_c_x: config.two_c_x,
            do_core: config.do_core,
            sig_h: config.sig_h,
            shadow: config.shadow,
            calc_area: config.calc_area,
            calc_length: config.calc_length,
            do_aagg: config.do_aagg,
            nn_profile: config.nn_profile.as_ref(),
            xsect_fluct: config.xsect_fluct.as_ref(),
        };

        let (event, _bc, _mpi, w) =
            simulate_event(&params, &mut nucleus_a, &mut nucleus_b, b, &mut rng);
        (event, w)
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
        self.config.do_core = self.do_core;
        self.config.do_aagg = self.do_aagg;
        self.config.shadow = self.shadow;
        self.config.calc_area = self.calc_area;
        self.config.calc_length = self.calc_length;
        self.config.sig_h = self.sig_h;
        self.config.nn_profile = self.nn_profile.clone();
        self.config.xsect_fluct = self.xsect_fluct.clone();

        let config = Arc::new(self.config.clone());

        use std::sync::atomic::{AtomicUsize, Ordering};
        let progress = AtomicUsize::new(0);
        let total_events = nevents as usize;

        let start_time = Instant::now();

        let results: Vec<(Vec<TGlauberEvent>, f64, f64)> = (0..num_threads)
            .into_par_iter()
            .map(|thread_id| {
                let events_this_thread = if thread_id < remaining as usize {
                    nevents_per_thread + 1
                } else {
                    nevents_per_thread
                };

                if events_this_thread <= 0 {
                    return (Vec::new(), 0.0, 0.0);
                }

                let mut thread_events = Vec::with_capacity(events_this_thread as usize);
                let mut thread_total_weight = 0.0;
                let mut thread_accepted_weight = 0.0;
                let base_seed = thread_id as u64 * 1000000 + 42;

                for i in 0..events_this_thread {
                    let seed = base_seed + i as u64;
                    let (mut event, mut w) = TGlauberMC::generate_event_with_seed(&config, seed, b);
                    thread_total_weight += w;

                    let mut attempts = 0;
                    while event.ncoll == 0.0 && attempts < 100 {
                        let new_seed = seed + 1000000 + attempts as u64;
                        let (e2, w2) =
                            TGlauberMC::generate_event_with_seed(&config, new_seed, b);
                        event = e2;
                        w = w2;
                        thread_total_weight += w;
                        attempts += 1;
                    }

                    if event.ncoll > 0.0 {
                        thread_accepted_weight += w;
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

                (thread_events, thread_total_weight, thread_accepted_weight)
            })
            .collect();

        let elapsed = start_time.elapsed();
        let rate = total_events as f64 / elapsed.as_secs_f64();

        let mut all_events = Vec::with_capacity(nevents as usize);
        let mut total_weight_attempted = 0.0;
        let mut total_weight_accepted = 0.0;
        for (events, tw, aw) in results {
            all_events.extend(events);
            total_weight_attempted += tw;
            total_weight_accepted += aw;
        }
        self.total_events = total_weight_attempted;
        self.events = total_weight_accepted;

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
