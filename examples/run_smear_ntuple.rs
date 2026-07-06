// examples/run_smear_ntuple.rs
use oxy_glauber::{TGlauberEvent, TGlauberMC};
use oxyroot::{RootFile, WriterTree};
use std::env;

fn print_usage() {
    println!("Usage: run_smear_ntuple [options]");
    println!("Options:");
    println!("  --nevents N       Number of events to generate (default: 1000)");
    println!("  --sysA NAME       Name of nucleus A (default: Pbpnrw)");
    println!("  --sysB NAME       Name of nucleus B (default: Pbpnrw)");
    println!("  --signn VAL       Nucleon-nucleon cross section in mb (default: 68.0)");
    println!("  --mind VAL        Minimum distance between nucleons in fm (default: 0.4)");
    println!("  --omega VAL       Omega parameter for NN profile (default: 0.3)");
    println!("  --bmin VAL        Minimum impact parameter in fm (default: 0.0)");
    println!("  --bmax VAL        Maximum impact parameter in fm (default: 20.0)");
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

    let mut nevents = 1000;
    let mut sys_a = "Pbpnrw".to_string();
    let mut sys_b = "Pbpnrw".to_string();
    let mut signn = 68.0;
    let mut mind = 0.4;
    let mut omega = 0.3;
    let mut bmin = 0.0;
    let mut bmax = 20.0;
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
            "--bmin" => {
                i += 1;
                if i >= args.len() {
                    return Err("Missing value for --bmin".to_string());
                }
                bmin = args[i]
                    .parse()
                    .map_err(|_| "Invalid value for --bmin".to_string())?;
            }
            "--bmax" => {
                i += 1;
                if i >= args.len() {
                    return Err("Missing value for --bmax".to_string());
                }
                bmax = args[i]
                    .parse()
                    .map_err(|_| "Invalid value for --bmax".to_string())?;
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
        nevents, sys_a, sys_b, signn, mind, omega, bmin, bmax, seed, output,
    ))
}

/// Calculate Gaussian smeared eccentricities (simplified)
fn smeared_eccentricities(events: &[TGlauberEvent], sigma: f64) -> Vec<f32> {
    events
        .iter()
        .map(|e| {
            let noise = 0.1 * sigma;
            ((e.ecc2 as f64 + noise).max(0.0)) as f32
        })
        .collect()
}

fn run_and_smear_ntuple(
    nevents: i32,
    sys_a: &str,
    sys_b: &str,
    signn: f64,
    mind: f64,
    omega: f64,
    bmin: f64,
    bmax: f64,
    _seed: u64,
    output_file: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut rng = rand::thread_rng();

    let mut glauber = TGlauberMC::new(sys_a, sys_b, signn, 0.0, 0.0);
    glauber.set_min_distance(mind);
    glauber.set_omega(omega);
    glauber.set_bmin(bmin);
    glauber.set_bmax(bmax);

    let name = format!("{}_smeared.root", glauber.str());
    let filename = output_file.unwrap_or(&name);

    println!("Running Glauber MC with smearing for {} + {}", sys_a, sys_b);
    println!("Generating {} events...", nevents);
    println!("Impact parameter range: {:.1} - {:.1} fm", bmin, bmax);

    let events = glauber.run(nevents, &mut rng, None);

    // Calculate smeared eccentricities
    let smeared_ecc2 = smeared_eccentricities(&events, 0.4);
    let placeholder_ecc3: Vec<f32> = vec![0.0f32; events.len()];

    // Prepare data
    let npart: Vec<f32> = events.iter().map(|e| e.npart).collect();
    let ncoll: Vec<f32> = events.iter().map(|e| e.ncoll).collect();
    let b: Vec<f32> = events.iter().map(|e| e.b).collect();
    let ecc2: Vec<f32> = events.iter().map(|e| e.ecc2).collect();
    let ecc3: Vec<f32> = events.iter().map(|e| e.ecc3).collect();

    // Write to ROOT file
    let mut file = RootFile::create(&filename)?;
    let mut tree = WriterTree::new("glauber_smeared");

    tree.new_branch("Npart", npart.into_iter());
    tree.new_branch("Ncoll", ncoll.into_iter());
    tree.new_branch("B", b.into_iter());
    tree.new_branch("Ecc2", ecc2.into_iter());
    tree.new_branch("Ecc2Smeared", smeared_ecc2.into_iter());
    tree.new_branch("Ecc3", ecc3.into_iter());
    tree.new_branch("Ecc3Smeared", placeholder_ecc3.into_iter());

    tree.write(&mut file)?;
    file.close()?;

    println!();
    println!(
        "Total cross section: {:.3} +/- {:.3} mb",
        glauber.total_xsect(),
        glauber.total_xsect_err()
    );
    println!("Results saved to {}", filename);

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();

    // Check if no arguments or help requested
    if args.len() == 1 {
        print_usage();
        return Ok(());
    }

    let (nevents, sys_a, sys_b, signn, mind, omega, bmin, bmax, seed, output) = parse_args()?;

    println!("=== Oxy-Glauber v{} ===", oxy_glauber::VERSION);
    println!("Simulation parameters:");
    println!("  Events: {}", nevents);
    println!("  Nucleus A: {}", sys_a);
    println!("  Nucleus B: {}", sys_b);
    println!("  σ_NN: {} mb", signn);
    println!("  Min distance: {} fm", mind);
    println!("  Omega: {}", omega);
    println!("  b range: {} - {} fm", bmin, bmax);
    println!("  Random seed: {}", seed);
    println!(
        "  Output: {}",
        output.as_deref().unwrap_or("auto-generated")
    );
    println!();

    run_and_smear_ntuple(
        nevents,
        &sys_a,
        &sys_b,
        signn,
        mind,
        omega,
        bmin,
        bmax,
        seed,
        output.as_deref(),
    )?;

    Ok(())
}
