// examples/run_save_ntuple.rs
use oxiroot::Compression;
use oxiroot::tree::{Branch, Tree};
use oxy_glauber::{TGlauberEvent, TGlauberMC};
use std::env;

/// Compression applied to every basket of the output TTree.
///
/// Same goal as `ZSTD_LEVEL` in `run_save_parquet.rs`: the smallest file that is still
/// fast to read. oxiroot's encoders are pure Rust, and its Zstd encoder has a single
/// (fastest) level - the requested level is only recorded, not used - so Zstd can't be
/// tuned for ratio here the way it is for Parquet. Measured on a 200k-event Pb+Pb run
/// (all 32 branches, one basket each):
/// - None: 25.64 MB
/// - LZ4: 18.69 MB
/// - Zstd: 16.92 MB, full read-back 0.05 s
/// - Zlib level 9: 14.55 MB, full read-back 0.06 s
/// - LZMA level 9: 10.44 MB, full read-back 0.42 s - the winner, kept below.
///
/// LZMA's extra write time (~5 s) is negligible next to event generation, and reading
/// is still well under a second, so the smaller file is the better trade.
const COMPRESSION: Compression = Compression::Lzma(9);

fn print_usage() {
    println!("Usage: run_save_ntuple [options]");
    println!("Options:");
    println!("  --nevents N       Number of events to generate (default: 10000)");
    println!("  --sysA NAME       Name of nucleus A (default: Pbpnrw)");
    println!("  --sysB NAME       Name of nucleus B (default: Pbpnrw)");
    println!("  --signn VAL       Nucleon-nucleon cross section in mb (default: 68.0)");
    println!("  --sigwidth VAL    Standard deviation of NN cross section (default: -1)");
    println!("  --mind VAL        Minimum distance between nucleons in fm (default: 0.4)");
    println!("  --omega VAL       Omega parameter for NN profile (default: 0.0)");
    println!(
        "  --noded VAL       Node distance for lattice placement (≤0 for continuous) (default: -1)"
    );
    println!("  --seed VAL        Random seed (default: 42)");
    println!("  --output FILE     Output file name (default: auto-generated)");
    println!("  --help            Print this help message");
    println!();
    println!("Note: If signn is negative, it is interpreted as beam energy in GeV");
}

fn parse_args() -> Result<
    (
        i32,
        String,
        String,
        f64,
        f64,
        f64,
        f64,
        f64,
        u64,
        Option<String>,
    ),
    String,
> {
    let args: Vec<String> = env::args().collect();

    let mut nevents = 10000;
    let mut sys_a = "Pbpnrw".to_string();
    let mut sys_b = "Pbpnrw".to_string();
    let mut signn = 68.0;
    let mut sigwidth = -1.0;
    let mut mind = 0.4;
    // Matches C++ runAndSaveNtuple's default (omega=0.0 means no NN profile - pure
    // hard-sphere ball-diameter cutoff, since TGlauberMC::CalcEvent only builds a
    // profile when fOmega>0).
    let mut omega = 0.0;
    let mut noded = -1.0;
    let mut seed = 42;
    let mut output = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--nevents" => {
                i += 1;
                if i >= args.len() {
                    return Err("Missing value for --nevents".to_string());
                }
                nevents = args[i]
                    .parse()
                    .map_err(|_| "Invalid value for --nevents".to_string())?;
            }
            "--sysA" => {
                i += 1;
                if i >= args.len() {
                    return Err("Missing value for --sysA".to_string());
                }
                sys_a = args[i].clone();
            }
            "--sysB" => {
                i += 1;
                if i >= args.len() {
                    return Err("Missing value for --sysB".to_string());
                }
                sys_b = args[i].clone();
            }
            "--signn" => {
                i += 1;
                if i >= args.len() {
                    return Err("Missing value for --signn".to_string());
                }
                signn = args[i]
                    .parse()
                    .map_err(|_| "Invalid value for --signn".to_string())?;
            }
            "--sigwidth" => {
                i += 1;
                if i >= args.len() {
                    return Err("Missing value for --sigwidth".to_string());
                }
                sigwidth = args[i]
                    .parse()
                    .map_err(|_| "Invalid value for --sigwidth".to_string())?;
            }
            "--mind" => {
                i += 1;
                if i >= args.len() {
                    return Err("Missing value for --mind".to_string());
                }
                mind = args[i]
                    .parse()
                    .map_err(|_| "Invalid value for --mind".to_string())?;
            }
            "--omega" => {
                i += 1;
                if i >= args.len() {
                    return Err("Missing value for --omega".to_string());
                }
                omega = args[i]
                    .parse()
                    .map_err(|_| "Invalid value for --omega".to_string())?;
            }
            "--noded" => {
                i += 1;
                if i >= args.len() {
                    return Err("Missing value for --noded".to_string());
                }
                noded = args[i]
                    .parse()
                    .map_err(|_| "Invalid value for --noded".to_string())?;
            }
            "--seed" => {
                i += 1;
                if i >= args.len() {
                    return Err("Missing value for --seed".to_string());
                }
                seed = args[i]
                    .parse()
                    .map_err(|_| "Invalid value for --seed".to_string())?;
            }
            "--output" => {
                i += 1;
                if i >= args.len() {
                    return Err("Missing value for --output".to_string());
                }
                output = Some(args[i].clone());
            }
            "--help" => {
                print_usage();
                std::process::exit(0);
            }
            _ => {
                return Err(format!("Unknown argument: {}", args[i]));
            }
        }
        i += 1;
    }

    Ok((
        nevents, sys_a, sys_b, signn, sigwidth, mind, omega, noded, seed, output,
    ))
}

/// Equivalent of runAndSaveNtuple from C++ code
fn run_and_save_ntuple(
    nevents: i32,
    sys_a: &str,
    sys_b: &str,
    signn: f64,
    sigwidth: f64,
    mind: f64,
    omega: f64,
    noded: f64,
    _seed: u64,
    output_file: Option<&str>,
) -> Result<Vec<TGlauberEvent>, Box<dyn std::error::Error>> {
    let mut rng = rand::rng();

    let mut glauber = TGlauberMC::new(sys_a, sys_b, signn, sigwidth, 0.0);
    glauber.set_min_distance(mind);
    glauber.set_node_distance(noded);
    glauber.set_calc_area(false);
    glauber.set_calc_length(false);
    glauber.set_calc_core(false);
    glauber.set_detail(99);

    if (0.0..=11.0).contains(&omega) {
        glauber.set_omega(omega);
    }

    let name = format!("{}.root", glauber.str());
    let filename = output_file.unwrap_or(&name);

    println!("Running Glauber MC for {} + {}", sys_a, sys_b);
    println!("Generating {} events...", nevents);

    let events = glauber.run(nevents, &mut rng, None);

    // --- Prepare data for writing with oxiroot ---
    // Collect each field into a separate Vec<f32>
    let npart: Vec<f32> = events.iter().map(|e| e.npart).collect();
    let ncoll: Vec<f32> = events.iter().map(|e| e.ncoll).collect();
    let nhard: Vec<f32> = events.iter().map(|e| e.nhard).collect();
    let nmpi: Vec<f32> = events.iter().map(|e| e.nmpi).collect();
    let b: Vec<f32> = events.iter().map(|e| e.b).collect();
    let bnn: Vec<f32> = events.iter().map(|e| e.bnn).collect();
    let ncollpp: Vec<f32> = events.iter().map(|e| e.ncollpp).collect();
    let ncollpn: Vec<f32> = events.iter().map(|e| e.ncollpn).collect();
    let ncollnn: Vec<f32> = events.iter().map(|e| e.ncollnn).collect();
    let var_x: Vec<f32> = events.iter().map(|e| e.var_x).collect();
    let var_y: Vec<f32> = events.iter().map(|e| e.var_y).collect();
    let var_xy: Vec<f32> = events.iter().map(|e| e.var_xy).collect();
    let npart_a: Vec<f32> = events.iter().map(|e| e.npart_a).collect();
    let npart_b: Vec<f32> = events.iter().map(|e| e.npart_b).collect();
    let npart0: Vec<f32> = events.iter().map(|e| e.npart0).collect();
    let npart_an: Vec<f32> = events.iter().map(|e| e.npart_an).collect();
    let npart_bn: Vec<f32> = events.iter().map(|e| e.npart_bn).collect();
    let npart0n: Vec<f32> = events.iter().map(|e| e.npart0n).collect();
    let area_w: Vec<f32> = events.iter().map(|e| e.area_w).collect();
    let spec_a: Vec<f32> = events.iter().map(|e| e.spec_a).collect();
    let spec_b: Vec<f32> = events.iter().map(|e| e.spec_b).collect();
    let weight: Vec<f32> = events.iter().map(|e| e.weight).collect();
    let psi1: Vec<f32> = events.iter().map(|e| e.psi1).collect();
    let ecc1: Vec<f32> = events.iter().map(|e| e.ecc1).collect();
    let psi2: Vec<f32> = events.iter().map(|e| e.psi2).collect();
    let ecc2: Vec<f32> = events.iter().map(|e| e.ecc2).collect();
    let psi3: Vec<f32> = events.iter().map(|e| e.psi3).collect();
    let ecc3: Vec<f32> = events.iter().map(|e| e.ecc3).collect();
    let psi4: Vec<f32> = events.iter().map(|e| e.psi4).collect();
    let ecc4: Vec<f32> = events.iter().map(|e| e.ecc4).collect();
    let psi5: Vec<f32> = events.iter().map(|e| e.psi5).collect();
    let ecc5: Vec<f32> = events.iter().map(|e| e.ecc5).collect();

    // --- Write to ROOT file as a TTree ---
    // One basket per branch: like the single Parquet row group in run_save_parquet.rs,
    // this lets the compressor see each whole column at once.
    let tree = Tree::new(
        "glauber",
        vec![
            Branch::f32("Npart", npart),
            Branch::f32("Ncoll", ncoll),
            Branch::f32("Nhard", nhard),
            Branch::f32("Nmpi", nmpi),
            Branch::f32("B", b),
            Branch::f32("BNN", bnn),
            Branch::f32("Ncollpp", ncollpp),
            Branch::f32("Ncollpn", ncollpn),
            Branch::f32("Ncollnn", ncollnn),
            Branch::f32("VarX", var_x),
            Branch::f32("VarY", var_y),
            Branch::f32("VarXY", var_xy),
            Branch::f32("NpartA", npart_a),
            Branch::f32("NpartB", npart_b),
            Branch::f32("Npart0", npart0),
            Branch::f32("NpartAn", npart_an),
            Branch::f32("NpartBn", npart_bn),
            Branch::f32("Npart0n", npart0n),
            Branch::f32("AreaW", area_w),
            Branch::f32("SpecA", spec_a),
            Branch::f32("SpecB", spec_b),
            Branch::f32("Weight", weight),
            Branch::f32("Psi1", psi1),
            Branch::f32("Ecc1", ecc1),
            Branch::f32("Psi2", psi2),
            Branch::f32("Ecc2", ecc2),
            Branch::f32("Psi3", psi3),
            Branch::f32("Ecc3", ecc3),
            Branch::f32("Psi4", psi4),
            Branch::f32("Ecc4", ecc4),
            Branch::f32("Psi5", psi5),
            Branch::f32("Ecc5", ecc5),
        ],
    );
    tree.write_root(filename, COMPRESSION)?;

    println!();
    println!(
        "Total cross section: {:.3} +/- {:.3} mb",
        glauber.total_xsect(),
        glauber.total_xsect_err()
    );
    println!("Results saved to {}", filename);

    Ok(events)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();

    // Check if no arguments or help requested
    if args.len() == 1 {
        print_usage();
        return Ok(());
    }

    let (nevents, sys_a, sys_b, signn, sigwidth, mind, omega, noded, seed, output) = parse_args()?;

    println!("=== Oxy-Glauber v{} ===", oxy_glauber::VERSION);
    println!("Simulation parameters:");
    println!("  Events: {}", nevents);
    println!("  Nucleus A: {}", sys_a);
    println!("  Nucleus B: {}", sys_b);
    println!("  σ_NN: {} mb", signn);
    println!("  σ width: {} mb", sigwidth);
    println!("  Min distance: {} fm", mind);
    println!("  Omega: {}", omega);
    println!("  Node distance: {} fm", noded);
    println!("  Random seed: {}", seed);
    println!(
        "  Output: {}",
        output.as_deref().unwrap_or("auto-generated")
    );
    println!();

    let _events = run_and_save_ntuple(
        nevents,
        &sys_a,
        &sys_b,
        signn,
        sigwidth,
        mind,
        omega,
        noded,
        seed,
        output.as_deref(),
    )?;

    Ok(())
}
