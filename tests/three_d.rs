// tests/three_d.rs
use oxy_glauber::{Glauber3DParams, TATB_LABELS, TGlauberMC};

const M_PROTON: f64 = 0.938;

fn au_au_3d(params: Glauber3DParams) -> TGlauberMC {
    let mut glauber = TGlauberMC::new("Au", "Au", 42.0, 0.0, 0.0).unwrap();
    glauber.set_three_d(params).unwrap();
    glauber
}

#[test]
fn standard_mode_has_no_3d_results() {
    let mut glauber = TGlauberMC::new("Au", "Au", 42.0, 0.0, 0.0).unwrap();
    let mut rng = rand::rng();
    while !glauber.next_event(&mut rng, Some(5.0)) {}
    assert!(glauber.event().three_d.is_none());
}

#[test]
fn cascade_counts_and_energy_conservation() {
    let mut glauber = au_au_3d(Glauber3DParams::default());
    let ybeam = glauber.three_d().unwrap().ybeam();
    let deta = glauber.three_d().unwrap().params().grid_deta;
    let mut rng = rand::rng();
    for b in [0.0, 4.0, 8.0, 11.0] {
        while !glauber.next_event(&mut rng, Some(b)) {}
        let event = glauber.event();
        let e3 = event.three_d.as_ref().unwrap();

        // every participant is connected to at least one string, and each executed
        // collision corresponds to one of the colliding pairs
        assert!(
            e3.nstrings >= event.npart / 2.0,
            "b={b}: Nstrings={} Npart={}",
            e3.nstrings,
            event.npart
        );
        assert!(
            e3.ncoll_exec >= 1.0 && e3.ncoll_exec <= event.ncoll,
            "b={b}: {} vs {}",
            e3.ncoll_exec,
            event.ncoll
        );

        // the strings and remnants carry exactly the participants' beam energy
        let deposited: f64 = e3.de_deta.iter().map(|&v| v as f64 * deta).sum();
        let available = event.npart as f64 * M_PROTON * ybeam.cosh();
        assert!(
            (deposited / available - 1.0).abs() < 1e-3,
            "b={b}: deposited {deposited:.1} GeV of {available:.1} GeV"
        );

        for (label, (ecc, psi)) in TATB_LABELS.iter().zip(e3.ecc_tatb.iter().zip(&e3.psi_tatb)) {
            assert_eq!(ecc.len(), 3);
            for (&e, &p) in ecc.iter().zip(psi) {
                assert!((0.0..=1.0).contains(&e), "{label}: eps = {e}");
                assert!(p.is_finite());
            }
        }
        assert_eq!(e3.ecc_eta.len(), 3);
        assert!(e3.ecc_eta.iter().flatten().all(|e| (0.0..=1.0).contains(e)));
    }
}

#[test]
fn eccentricity_eta_can_be_disabled() {
    let params = Glauber3DParams {
        eccentricity_eta: false,
        ..Default::default()
    };
    let mut glauber = au_au_3d(params);
    let mut rng = rand::rng();
    while !glauber.next_event(&mut rng, Some(6.0)) {}
    let e3 = glauber.event().three_d.as_ref().unwrap();
    assert!(e3.de_deta.is_empty() && e3.ecc_eta.is_empty());
    assert_eq!(e3.ecc_tatb.len(), TATB_LABELS.len());
}

#[test]
fn elliptic_overlap_in_mid_central_collisions() {
    // <eps_2> of the participant density grows from central to mid-central events
    let params = Glauber3DParams {
        eccentricity_eta: false,
        ..Default::default()
    };
    let mut glauber = au_au_3d(params);
    let mut rng = rand::rng();
    let mut mean = |b: f64| {
        let mut sum = 0.0;
        for _ in 0..200 {
            while !glauber.next_event(&mut rng, Some(b)) {}
            sum += glauber.event().three_d.as_ref().unwrap().ecc_tatb[0][1] as f64;
        }
        sum / 200.0
    };
    let (central, mid) = (mean(0.0), mean(8.0));
    assert!(
        mid > central + 0.1,
        "<eps_2>: b=0 {central:.3}, b=8 {mid:.3}"
    );
}

#[test]
fn energy_runs_set_roots() {
    // a run created from sqrt(s_NN) (negative xsect) collides at that energy
    let mut glauber = TGlauberMC::new("Au", "Au", -2.4, 0.0, 0.0).unwrap();
    glauber.set_three_d(Glauber3DParams { roots: 200.0, ..Default::default() }).unwrap();
    let g3 = glauber.three_d().unwrap();
    assert_eq!(g3.params().roots, 2.4);
    assert!((g3.ybeam() - (2.4 / (2.0 * M_PROTON)).acosh()).abs() < 1e-12);

    // a run created from a cross section keeps the configured roots
    let glauber = au_au_3d(Glauber3DParams { roots: 62.4, ..Default::default() });
    assert_eq!(glauber.three_d().unwrap().params().roots, 62.4);
}
