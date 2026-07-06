// examples/run_save_ntuple.rs
use oxy_glauber::{TGlauberEvent, TGlauberMC};
use oxyroot::{RootFile, WriterTree};
use std::env;

fn print_usage() {
    println!("Usage: run_save_ntuple [options]");
    println!("Options:");
    println!("  --nevents N       Number of events to generate (default: 10000)");
    println!("  --sysA NAME       Name of nucleus A (default: Pbpnrw)");
    println!("  --sysB NAME       Name of nucleus B (default: Pbpnrw)");
    println!("  --signn VAL       Nucleon-nucleon cross section in mb (default: 68.0)");
    println!("  --sigwidth VAL    Standard deviation of NN cross section (default: -1)");
    println!("  --mind VAL        Minimum distance between nucleons in fm (default: 0.4)");
    println!("  --omega VAL       Omega parameter for NN profile (default: 0.3)");
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
    let mut omega = 0.3;
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
    let mut rng = rand::thread_rng();

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

    // --- Prepare data for writing with oxyroot ---
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

    // --- Write to ROOT file using WriterTree ---
    let mut file = RootFile::create(&filename)?;
    let mut tree = WriterTree::new("glauber");

    // Pass iterators using .into_iter()
    tree.new_branch("Npart", npart.into_iter());
    tree.new_branch("Ncoll", ncoll.into_iter());
    tree.new_branch("Nhard", nhard.into_iter());
    tree.new_branch("Nmpi", nmpi.into_iter());
    tree.new_branch("B", b.into_iter());
    tree.new_branch("BNN", bnn.into_iter());
    tree.new_branch("Ncollpp", ncollpp.into_iter());
    tree.new_branch("Ncollpn", ncollpn.into_iter());
    tree.new_branch("Ncollnn", ncollnn.into_iter());
    tree.new_branch("VarX", var_x.into_iter());
    tree.new_branch("VarY", var_y.into_iter());
    tree.new_branch("VarXY", var_xy.into_iter());
    tree.new_branch("NpartA", npart_a.into_iter());
    tree.new_branch("NpartB", npart_b.into_iter());
    tree.new_branch("Npart0", npart0.into_iter());
    tree.new_branch("NpartAn", npart_an.into_iter());
    tree.new_branch("NpartBn", npart_bn.into_iter());
    tree.new_branch("Npart0n", npart0n.into_iter());
    tree.new_branch("AreaW", area_w.into_iter());
    tree.new_branch("SpecA", spec_a.into_iter());
    tree.new_branch("SpecB", spec_b.into_iter());
    tree.new_branch("Weight", weight.into_iter());
    tree.new_branch("Psi1", psi1.into_iter());
    tree.new_branch("Ecc1", ecc1.into_iter());
    tree.new_branch("Psi2", psi2.into_iter());
    tree.new_branch("Ecc2", ecc2.into_iter());
    tree.new_branch("Psi3", psi3.into_iter());
    tree.new_branch("Ecc3", ecc3.into_iter());
    tree.new_branch("Psi4", psi4.into_iter());
    tree.new_branch("Ecc4", ecc4.into_iter());
    tree.new_branch("Psi5", psi5.into_iter());
    tree.new_branch("Ecc5", ecc5.into_iter());

    // Write the tree to the file and close it
    tree.write(&mut file)?;
    file.close()?;

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
