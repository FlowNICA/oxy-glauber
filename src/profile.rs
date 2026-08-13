// src/profile.rs
use crate::constants::{MB_TO_FM2, PI};

/// Nucleon-nucleon profile types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NNProfileType {
    HardSphere,
    Gamma,
    Hijing,
    Pythia,
    Trento,
}

/// Nucleon-nucleon interaction profile
#[derive(Debug, Clone)]
pub struct NNProfile {
    pub profile_type: NNProfileType,
    pub sigma_nn: f64, // in mb
    pub omega: f64,    // Gamma parameter (0-2)
    pub w: f64,        // Trento W parameter
    pub mu: f64,       // HIJING mu parameter
    pub m: f64,        // PYTHIA m parameter
    pub rp: f64,       // PYTHIA rp parameter
    pub g: f64,        // Gamma G parameter
    /// Normalization constant (McGlauber's "A"/"k"), calibrated so the profile's radial
    /// integral reproduces sigma_nn. Unused (1.0) for HardSphere/Gamma, which aren't
    /// calibrated in the C++ reference either.
    pub calibration: f64,
}

/// Composite Simpson's rule integrator, matching TF1::Integral closely enough for the
/// loose (+/-0.01 mb) calibration tolerance used below.
fn integrate<F: Fn(f64) -> f64>(f: F, a: f64, b: f64, n: usize) -> f64 {
    let n = if n % 2 == 1 { n + 1 } else { n };
    let h = (b - a) / n as f64;
    let mut sum = f(a) + f(b);
    for i in 1..n {
        let x = a + i as f64 * h;
        sum += if i % 2 == 0 { 2.0 * f(x) } else { 4.0 * f(x) };
    }
    sum * h / 3.0
}

/// Calibrate the normalization constant `k` so that `integrate(|b| shape(k,b), 0, upper)`
/// reproduces `signn*0.1`, matching the `k*=1.01`/`k*=0.99` loop used by
/// GetNNHijingDist/GetNNPythiaDist/GetNNTrentoDist in the C++ reference.
fn calibrate<F: Fn(f64, f64) -> f64>(signn: f64, upper: f64, shape: F) -> f64 {
    let target = signn * 0.1;
    let mut k = 1.0;
    for _ in 0..100_000 {
        let val = integrate(|b| shape(k, b), 0.0, upper, 2000);
        let diff = val - target;
        if diff < -0.01 {
            k *= 1.01;
        } else if diff > 0.01 {
            k *= 0.99;
        } else {
            break;
        }
    }
    k
}

// --- Special functions (hand-rolled, no external crate needed) ---

/// ln(Gamma(x)), Lanczos approximation (Numerical Recipes coefficients).
fn gamma_ln(x: f64) -> f64 {
    const COF: [f64; 6] = [
        76.18009172947146,
        -86.50532032941677,
        24.01409824083091,
        -1.231739572450155,
        0.1208650973866179e-2,
        -0.5395239384953e-5,
    ];
    let mut y = x;
    let tmp0 = x + 5.5;
    let tmp = tmp0 - (x + 0.5) * tmp0.ln();
    let mut ser = 1.000000000190015;
    for c in COF.iter() {
        y += 1.0;
        ser += c / y;
    }
    -tmp + (2.5066282746310005 * ser / x).ln()
}

/// Series representation of the regularized lower incomplete gamma function P(a,x),
/// valid for x < a+1.
fn gamma_p_series(a: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    let gln = gamma_ln(a);
    let mut ap = a;
    let mut sum = 1.0 / a;
    let mut del = sum;
    for _ in 0..200 {
        ap += 1.0;
        del *= x / ap;
        sum += del;
        if del.abs() < sum.abs() * 3.0e-12 {
            break;
        }
    }
    sum * (-x + a * x.ln() - gln).exp()
}

/// Continued-fraction representation of the regularized upper incomplete gamma
/// function Q(a,x) = 1-P(a,x), valid for x >= a+1.
fn gamma_q_cf(a: f64, x: f64) -> f64 {
    const FPMIN: f64 = 1.0e-300;
    let gln = gamma_ln(a);
    let mut b = x + 1.0 - a;
    let mut c = 1.0 / FPMIN;
    let mut d = 1.0 / b;
    let mut h = d;
    for i in 1..=200 {
        let an = -(i as f64) * (i as f64 - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = b + an / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < 3.0e-12 {
            break;
        }
    }
    (-x + a * x.ln() - gln).exp() * h
}

/// Regularized lower incomplete gamma function P(a,x), matching ROOT's `TMath::Gamma(a,x)`.
fn regularized_gamma_p(a: f64, x: f64) -> f64 {
    if x < 0.0 || a <= 0.0 {
        return 0.0;
    }
    if x == 0.0 {
        return 0.0;
    }
    if x < a + 1.0 {
        gamma_p_series(a, x)
    } else {
        1.0 - gamma_q_cf(a, x)
    }
}

/// Modified Bessel function I0(x), Abramowitz & Stegun 9.8.1/9.8.2 polynomial approximation.
fn bessel_i0(x: f64) -> f64 {
    let ax = x.abs();
    if ax < 3.75 {
        let y = (x / 3.75).powi(2);
        1.0 + y
            * (3.5156229
                + y * (3.0899424
                    + y * (1.2067492 + y * (0.2659732 + y * (0.0360768 + y * 0.0045813)))))
    } else {
        let y = 3.75 / ax;
        (ax.exp() / ax.sqrt())
            * (0.39894228
                + y * (0.01328592
                    + y * (0.00225319
                        + y * (-0.00157565
                            + y * (0.00916281
                                + y * (-0.02057706
                                    + y * (0.02635537 + y * (-0.01647633 + y * 0.00392377))))))))
    }
}

/// Modified Bessel function K0(x), Abramowitz & Stegun 9.8.5/9.8.6.
fn bessel_k0(x: f64) -> f64 {
    if x <= 2.0 {
        let y = x * x / 4.0;
        (-(x / 2.0).ln()) * bessel_i0(x)
            + (-0.57721566
                + y * (0.42278420
                    + y * (0.23069756
                        + y * (0.03488590 + y * (0.00262698 + y * (0.00010750 + y * 0.00000740))))))
    } else {
        let y = 2.0 / x;
        ((-x).exp() / x.sqrt())
            * (1.25331414
                + y * (-0.07832358
                    + y * (0.02189568
                        + y * (-0.01062446
                            + y * (0.00587872 + y * (-0.00251540 + y * 0.00053208))))))
    }
}

/// Modified Bessel function I1(x), Abramowitz & Stegun 9.8.3/9.8.4.
fn bessel_i1(x: f64) -> f64 {
    let ax = x.abs();
    let result = if ax < 3.75 {
        let y = (x / 3.75).powi(2);
        ax * (0.5
            + y * (0.87890594
                + y * (0.51498869
                    + y * (0.15084934 + y * (0.02658733 + y * (0.00301532 + y * 0.00032411))))))
    } else {
        let y = 3.75 / ax;
        let ans = 0.02282967 + y * (-0.02895312 + y * (0.01787654 - y * 0.00420059));
        let ans = 0.39894228
            + y * (-0.03988024
                + y * (-0.00362018 + y * (0.00163801 + y * (-0.01031555 + y * ans))));
        ans * (ax.exp() / ax.sqrt())
    };
    if x < 0.0 { -result } else { result }
}

/// Modified Bessel function K1(x), Abramowitz & Stegun 9.8.7/9.8.8.
fn bessel_k1(x: f64) -> f64 {
    if x <= 2.0 {
        let y = x * x / 4.0;
        (x / 2.0).ln() * bessel_i1(x)
            + (1.0 / x)
                * (1.0
                    + y * (0.15443144
                        + y * (-0.67278579
                            + y * (-0.18156897
                                + y * (-0.01919402
                                    + y * (-0.00110404 + y * (-0.00004686)))))))
    } else {
        let y = 2.0 / x;
        ((-x).exp() / x.sqrt())
            * (1.25331414
                + y * (0.23498619
                    + y * (-0.03655620
                        + y * (0.01504268
                            + y * (-0.00780353 + y * (0.00325614 + y * (-0.00068245)))))))
    }
}

/// Modified Bessel function K3(x), via the standard recurrence
/// K_{n+1}(x) = K_{n-1}(x) + (2n/x)*K_n(x) starting from K0/K1.
fn bessel_k3(x: f64) -> f64 {
    let k0 = bessel_k0(x);
    let k1 = bessel_k1(x);
    let k2 = k0 + (2.0 / x) * k1;
    k1 + (4.0 / x) * k2
}

/// x^3 * K3(x), with the analytic small-x limit (-> 8) used near x=0 where K3 itself
/// diverges but the product stays finite.
fn x3_bessel_k3(x: f64) -> f64 {
    if x < 1.0e-4 { 8.0 } else { x.powi(3) * bessel_k3(x) }
}

impl NNProfile {
    pub fn new_hard_sphere(sigma_nn: f64) -> Self {
        Self {
            profile_type: NNProfileType::HardSphere,
            sigma_nn,
            omega: 0.0,
            w: 0.5,
            mu: 3.9,
            m: 1.85,
            rp: 1.0,
            g: 1.0,
            calibration: 1.0,
        }
    }

    pub fn new_gamma(sigma_nn: f64, omega: f64, g: f64) -> Self {
        Self {
            profile_type: NNProfileType::Gamma,
            sigma_nn,
            omega,
            w: 0.5,
            mu: 3.9,
            m: 1.85,
            rp: 1.0,
            g,
            calibration: 1.0,
        }
    }

    pub fn new_hijing(sigma_nn: f64, mu: f64) -> Self {
        let b0 = (0.5 * sigma_nn * MB_TO_FM2 / PI).sqrt();
        let a = mu / b0;
        // C++ GetNNHijingDist integrates to 5, not the function's own declared range of 10.
        let calibration = calibrate(sigma_nn, 5.0, |k, b| {
            let exponent = k * a * a / 96.0 * x3_bessel_k3(a * b);
            2.0 * PI * b * (1.0 - (-exponent).exp())
        });
        Self {
            profile_type: NNProfileType::Hijing,
            sigma_nn,
            omega: 7.0,
            w: 0.5,
            mu,
            m: 1.85,
            rp: 1.0,
            g: 1.0,
            calibration,
        }
    }

    pub fn new_pythia(sigma_nn: f64, m: f64, rp: f64) -> Self {
        // C++ GetNNPythiaDist integrates to 5, not the function's own declared range of 10.
        let calibration = calibrate(sigma_nn, 5.0, |k, b| {
            let arg = (b / rp).powf(m);
            2.0 * PI * b * (1.0 - (-k * (-arg).exp()).exp())
        });
        Self {
            profile_type: NNProfileType::Pythia,
            sigma_nn,
            omega: 8.0,
            w: 0.5,
            mu: 3.9,
            m,
            rp,
            g: 1.0,
            calibration,
        }
    }

    pub fn new_trento(sigma_nn: f64, w: f64) -> Self {
        const MAXB: f64 = 6.0;
        let calibration = calibrate(sigma_nn, MAXB, |k, b| {
            let arg = b * b / (4.0 * w * w);
            2.0 * PI * b * (1.0 - (-k * (-arg).exp()).exp())
        });
        Self {
            profile_type: NNProfileType::Trento,
            sigma_nn,
            omega: 9.0 + w,
            w,
            mu: 3.9,
            m: 1.85,
            rp: 1.0,
            g: 1.0,
            calibration,
        }
    }

    /// Evaluate the profile at impact parameter b (in fm)
    pub fn eval(&self, b: f64) -> f64 {
        if b < 0.0 {
            return 0.0;
        }
        let b2 = b * b;

        match self.profile_type {
            NNProfileType::HardSphere => {
                let r2 = self.sigma_nn * MB_TO_FM2 / PI;
                if b2 < r2 { 1.0 } else { 0.0 }
            }
            NNProfileType::Gamma => {
                // P(b) = G * Q(1/omega, (G/(omega*R^2))*b^2), from arXiv:1307.0636
                if self.omega <= 0.0 || self.omega > 2.0 {
                    return 0.0;
                }
                let r2 = self.sigma_nn * MB_TO_FM2 / PI;
                let a = 1.0 / self.omega;
                let beta = self.g / (self.omega * r2);
                let x = beta * b2;
                self.g * (1.0 - regularized_gamma_p(a, x))
            }
            NNProfileType::Hijing => {
                // P(b) = 1 - exp(-A*a^2/96*(a*b)^3*K3(a*b)), a = mu/b0
                let b0 = (0.5 * self.sigma_nn * MB_TO_FM2 / PI).sqrt();
                let a = self.mu / b0;
                let exponent = self.calibration * a * a / 96.0 * x3_bessel_k3(a * b);
                1.0 - (-exponent).exp()
            }
            NNProfileType::Pythia => {
                // P(b) = 1 - exp(-A*exp(-(b/rp)^m))
                let arg = (b / self.rp).powf(self.m);
                1.0 - (-self.calibration * (-arg).exp()).exp()
            }
            NNProfileType::Trento => {
                // P(b) = 1 - exp(-A*exp(-b^2/(4*w^2)))
                let arg = b2 / (4.0 * self.w * self.w);
                1.0 - (-self.calibration * (-arg).exp()).exp()
            }
        }
    }

    /// Get the maximum impact parameter where profile is non-zero (matches each profile's
    /// declared TF1 range in the C++ reference; used to pre-filter nucleon pairs).
    pub fn max_b(&self) -> f64 {
        match self.profile_type {
            NNProfileType::HardSphere => (self.sigma_nn * MB_TO_FM2 / PI).sqrt(),
            NNProfileType::Gamma => 5.0,
            NNProfileType::Hijing => 10.0,
            NNProfileType::Pythia => 10.0,
            NNProfileType::Trento => 6.0,
        }
    }
}

/// Create a profile from omega parameter, matching the dispatch in
/// `TGlauberMC::CalcEvent` (`if (fOmega>0 && !fNNProf) { if (fOmega<2) ... }`):
/// omega<=0 means no profile at all (hard-sphere, ball-diameter cutoff only) - note
/// this is *not* the same code path as `omega<0` in `TGlauberMC::SetOmega`, but has
/// the same net effect since `CalcEvent` only ever builds a profile when `fOmega>0`.
pub fn profile_from_omega(sigma_nn: f64, omega: f64) -> Option<NNProfile> {
    if omega <= 0.0 {
        None
    } else if omega < 2.0 {
        Some(NNProfile::new_gamma(sigma_nn, omega, 1.0))
    } else if omega == 7.0 {
        Some(NNProfile::new_hijing(sigma_nn, 3.9))
    } else if omega == 8.0 {
        Some(NNProfile::new_pythia(sigma_nn, 1.85, 1.0))
    } else if (9.0..11.0).contains(&omega) {
        let w = omega - 9.0;
        Some(NNProfile::new_trento(sigma_nn, w))
    } else {
        None
    }
}
