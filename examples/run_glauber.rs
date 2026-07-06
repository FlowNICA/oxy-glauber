// examples/run_glauber.rs
use oxy_glauber::TGlauberMC;
use oxyroot::{RootFile, WriterTree};
use std::env;

fn print_usage() {
    println!("Usage: run_glauber [options]");
    println!("Options:");
    println!("  --nevents N       Number of events to generate (default: 1000)");
    println!("  --sysA NAME       Name of nucleus A (default: Pb)");
    println!("  --sysB NAME       Name of nucleus B (default: Pb)");
    println!("  --signn VAL       Nucleon-nucleon cross section in mb (default: 68.0)");
    println!("  --mind VAL        Minimum distance between nucleons in fm (default: 0.4)");
    println!("  --omega VAL       Omega parameter for NN profile (default: 0.3)");
    println!("  --bmin VAL        Minimum impact parameter in fm (default: 0.0)");
    println!("  --bmax VAL        Maximum impact parameter in fm (default: 20.0)");
    println!("  --seed VAL        Random seed (default: 42)");
    println!("  --output FILE     Output file name (default: glauber_output.root)");
    println!("  --help            Print this help message");
    println!();
    println!("Note: If signn is negative, it is interpreted as beam energy in GeV");
}

fn parse_args() -> Result<(i32, String, String, f64, f64, f64, f64, f64, u64, String), String> {
    let args: Vec<String> = env::args().collect();

    let mut nevents = 1000;
    let mut sys_a = "Pb".to_string();
    let mut sys_b = "Pb".to_string();
    let mut signn = 68.0;
    let mut mind = 0.4;
    let mut omega = 0.3;
    let mut bmin = 0.0;
    let mut bmax = 20.0;
    let mut seed = 42;
    let mut output = "glauber_output.root".to_string();

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
                output = args[i].clone();
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();

    // Check if no arguments or help requested
    if args.len() == 1 {
        print_usage();
        return Ok(());
    }

    let (nevents, sys_a, sys_b, signn, mind, omega, bmin, bmax, _seed, output) = parse_args()?;

    println!("=== Oxy-Glauber v{} ===", oxy_glauber::VERSION);
    println!("Simulation parameters:");
    println!("  Events: {}", nevents);
    println!("  Nucleus A: {}", sys_a);
    println!("  Nucleus B: {}", sys_b);
    println!("  σ_NN: {} mb", signn);
    println!("  Min distance: {} fm", mind);
    println!("  Omega: {}", omega);
    println!("  b range: {} - {} fm", bmin, bmax);
    println!("  Output: {}", output);
    println!();

    let mut rng = rand::thread_rng();

    let mut glauber = TGlauberMC::new(&sys_a, &sys_b, signn, 0.0, 0.0);
    glauber.set_min_distance(mind);
    glauber.set_omega(omega);
    glauber.set_bmin(bmin);
    glauber.set_bmax(bmax);

    println!("Running Glauber Monte Carlo for {}+{}...", sys_a, sys_b);
    println!("Generating {} events...", nevents);

    let events = glauber.run(nevents, &mut rng, None);

    println!();
    println!(
        "Total cross section: {:.3} +/- {:.3} mb",
        glauber.total_xsect(),
        glauber.total_xsect_err()
    );

    // Print summary statistics
    let npart_avg: f64 = events.iter().map(|e| e.npart as f64).sum::<f64>() / events.len() as f64;
    let ncoll_avg: f64 = events.iter().map(|e| e.ncoll as f64).sum::<f64>() / events.len() as f64;
    let ecc2_avg: f64 = events.iter().map(|e| e.ecc2 as f64).sum::<f64>() / events.len() as f64;
    let ecc3_avg: f64 = events.iter().map(|e| e.ecc3 as f64).sum::<f64>() / events.len() as f64;

    println!();
    println!("Average Npart: {:.2}", npart_avg);
    println!("Average Ncoll: {:.2}", ncoll_avg);
    println!("Average Ecc2: {:.4}", ecc2_avg);
    println!("Average Ecc3: {:.4}", ecc3_avg);

    // Save to ROOT file using oxyroot
    println!("\nSaving results to {}", output);

    // Prepare data
    let npart: Vec<f32> = events.iter().map(|e| e.npart).collect();
    let ncoll: Vec<f32> = events.iter().map(|e| e.ncoll).collect();
    let b: Vec<f32> = events.iter().map(|e| e.b).collect();
    let ecc2: Vec<f32> = events.iter().map(|e| e.ecc2).collect();
    let ecc3: Vec<f32> = events.iter().map(|e| e.ecc3).collect();

    let mut file = RootFile::create(&output)?;
    let mut tree = WriterTree::new("glauber");

    tree.new_branch("Npart", npart.into_iter());
    tree.new_branch("Ncoll", ncoll.into_iter());
    tree.new_branch("B", b.into_iter());
    tree.new_branch("Ecc2", ecc2.into_iter());
    tree.new_branch("Ecc3", ecc3.into_iter());

    tree.write(&mut file)?;
    file.close()?;

    println!("Done!");

    Ok(())
}
