// src/bin/plot_radius_distribution.rs
use oxy_glauber::TGlauNucleus;
use oxy_glauber::constants::{PI, TWO_PI};
use plotters::coord::Shift;
use plotters::prelude::*;
use rand::Rng;
use rand::RngExt;
use rand::rngs::ThreadRng;
use std::env;

fn print_usage() {
    println!("Usage: plot_radius_distribution [options]");
    println!("Options:");
    println!("  --nucleus NAME   Name of the nucleus to plot (default: Pb)");
    println!("  --samples N      Number of random samples (default: 100000)");
    println!("  --bins N         Number of histogram bins (default: 100)");
    println!("  --output FILE    Output image file name (default: radius_distribution.png)");
    println!("  --help           Print this help message");
    println!();
    println!("Plots the nucleon distribution function (3pF, 3pG, deformed profiles, etc.)");
    println!("of the given nucleus as a function of radius r, azimuthal angle phi, and");
    println!("polar angle theta.");
    println!();
    println!("Supported nuclei: Pb, Au, Cu, O, Ne, Al, U, p, d, and many more.");
    println!("See the TGlauNucleus::lookup function for the complete list.");
}

fn parse_args() -> Result<(String, usize, usize, String), String> {
    let args: Vec<String> = env::args().collect();

    let mut nucleus = "Pb".to_string();
    let mut samples = 100000;
    let mut bins = 100;
    let mut output = "radius_distribution.png".to_string();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--nucleus" => {
                i += 1;
                if i >= args.len() {
                    return Err("Missing value for --nucleus".to_string());
                }
                nucleus = args[i].clone();
            }
            "--samples" => {
                i += 1;
                if i >= args.len() {
                    return Err("Missing value for --samples".to_string());
                }
                samples = args[i]
                    .parse()
                    .map_err(|_| "Invalid value for --samples".to_string())?;
            }
            "--bins" => {
                i += 1;
                if i >= args.len() {
                    return Err("Missing value for --bins".to_string());
                }
                bins = args[i]
                    .parse()
                    .map_err(|_| "Invalid value for --bins".to_string())?;
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

    Ok((nucleus, samples, bins, output))
}

/// Calculate the root-mean-square radius from the sampled radii
fn calculate_rms_radius(radii: &[f64]) -> f64 {
    if radii.is_empty() {
        return 0.0;
    }
    let sum_sq: f64 = radii.iter().map(|&r| r * r).sum();
    (sum_sq / radii.len() as f64).sqrt()
}

/// Sample (r, phi, theta) triples directly from the nuclear density function.
///
/// This deliberately does *not* go through `throw_nucleons`: that function
/// recenters the whole nucleus on its center of mass and applies a random
/// per-event overall orientation, which distorts the radial shape (or, for a
/// single nucleon, collapses it to r=0) and - crucially for phi/theta - washes
/// out any deformation-driven angular dependence into an isotropic lab-frame
/// distribution. Sampling directly from the density function shows the true
/// rho(r, theta) shape in the nucleus's own intrinsic frame, matching
/// McGlauber's `TGlauNucleus::fF` (3pF, 3pG, deformed profiles, etc.) via
/// rejection sampling, the same way `throw_nucleons` itself does per nucleon.
fn sample_distributions<R: Rng>(
    nucleus: &mut TGlauNucleus,
    rng: &mut R,
    n_samples: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    if nucleus.n() == 0 {
        return (Vec::new(), Vec::new(), Vec::new());
    }

    // Sample protons/neutrons in proportion to the nucleus composition, since some
    // profiles (e.g. ProtonNeutron3PF) use a different density for each species.
    let proton_fraction = nucleus.z() as f64 / nucleus.n() as f64;

    let mut radii = Vec::with_capacity(n_samples);
    let mut phis = Vec::with_capacity(n_samples);
    let mut thetas = Vec::with_capacity(n_samples);
    for _ in 0..n_samples {
        let is_proton = rng.random::<f64>() < proton_fraction;
        let (r, phi, theta) = nucleus.sample_nucleon_direction(rng, is_proton);
        radii.push(r);
        phis.push(phi);
        thetas.push(theta);
    }

    (radii, phis, thetas)
}

/// Generate a single nucleus and return the nucleon positions
fn generate_nucleus_positions<R: Rng>(
    nucleus: &mut TGlauNucleus,
    rng: &mut R,
) -> Vec<(f64, f64, f64, bool)> {
    // Throw nucleons for this nucleus
    nucleus.throw_nucleons(0.0, rng);

    // Collect positions and types from all nucleons
    let mut positions = Vec::with_capacity(nucleus.n() as usize);
    for nucleon in nucleus.nucleons() {
        let x = nucleon.x();
        let y = nucleon.y();
        let z = nucleon.z();
        let is_proton = nucleon.is_proton();
        positions.push((x, y, z, is_proton));
    }
    positions
}

/// Build a fixed-range histogram of `values` over `[min_v, max_v]`.
fn histogram(values: &[f64], min_v: f64, max_v: f64, bins: usize) -> Vec<usize> {
    let bin_width = (max_v - min_v) / bins as f64;
    let mut hist = vec![0usize; bins];
    for &v in values {
        let bin = ((v - min_v) / bin_width) as usize;
        let bin = bin.min(bins - 1);
        hist[bin] += 1;
    }
    hist
}

/// Draw one histogram panel (bar chart) into a plotters drawing area.
fn draw_histogram_panel<DB: DrawingBackend>(
    area: &DrawingArea<DB, Shift>,
    title: &str,
    x_desc: &str,
    hist: &[usize],
    min_v: f64,
    max_v: f64,
) -> Result<(), Box<dyn std::error::Error>>
where
    DB::ErrorType: 'static,
{
    let bins = hist.len();
    let bin_width = (max_v - min_v) / bins as f64;
    let max_count = *hist.iter().max().unwrap_or(&1);

    let mut chart = ChartBuilder::on(area)
        .caption(title, ("sans-serif", 16).into_font())
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(50)
        .build_cartesian_2d(min_v..max_v, 0..max_count.max(1))?;

    chart
        .configure_mesh()
        .x_desc(x_desc)
        .y_desc("Count")
        .axis_desc_style(("sans-serif", 12))
        .light_line_style(WHITE.mix(0.0))
        .draw()?;

    let bars = hist.iter().enumerate().map(|(i, &count)| {
        let x0 = min_v + i as f64 * bin_width;
        let x1 = x0 + bin_width;
        Rectangle::new(
            [(x0, 0), (x1, count)],
            ShapeStyle {
                color: BLUE.mix(0.6).into(),
                filled: true,
                stroke_width: 1,
            },
        )
    });
    chart.draw_series(bars)?;

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();

    if args.len() == 1 {
        print_usage();
        return Ok(());
    }

    let (nucleus_name, samples, bins, output) = parse_args()?;

    println!("=== Oxy-Glauber v{} ===", oxy_glauber::VERSION);
    println!(
        "Plotting nucleon distribution function for nucleus: {}",
        nucleus_name
    );
    println!("Samples: {}", samples);
    println!("Bins: {}", bins);
    println!("Output: {}", output);
    println!();

    // Create the nucleus - uses the same lookup table as the main simulation
    let mut nucleus = TGlauNucleus::new(&nucleus_name)?;
    let mut rng = ThreadRng::default();

    // Check if nucleus is valid
    if nucleus.n() == 0 {
        eprintln!("Error: Nucleus '{}' has no nucleons defined.", nucleus_name);
        std::process::exit(1);
    }

    println!(
        "Nucleus: {} (A={}, Z={})",
        nucleus_name,
        nucleus.n(),
        nucleus.z()
    );
    println!("Sampling (r, phi, theta) directly from the nuclear density function...");

    let (radii, phis, thetas) = sample_distributions(&mut nucleus, &mut rng, samples);

    if radii.is_empty() {
        eprintln!("Error: No samples collected.");
        std::process::exit(1);
    }

    println!("Collected {} samples", radii.len());

    // Generate one nucleus to show the positions
    let positions = generate_nucleus_positions(&mut nucleus, &mut rng);
    println!("Generated one nucleus with {} nucleons", positions.len());

    // Find the maximum radius for the histogram
    let max_r = radii
        .iter()
        .fold(0.0_f64, |acc, &r| if r > acc { r } else { acc });
    let min_r = radii
        .iter()
        .fold(f64::INFINITY, |acc, &r| if r < acc { r } else { acc });
    println!("Radius range: {:.3} - {:.3} fm", min_r, max_r);

    // Guard against a degenerate (zero-width) range, e.g. too few samples.
    let max_r = if max_r > min_r { max_r } else { min_r + 1.0 };

    let rms_radius = calculate_rms_radius(&radii);
    println!("RMS radius: {:.3} fm", rms_radius);

    let r_hist = histogram(&radii, min_r, max_r, bins);
    let phi_hist = histogram(&phis, 0.0, TWO_PI, bins);
    let theta_hist = histogram(&thetas, 0.0, PI, bins);

    // --- Layout: 2x2 grid (radius, phi, theta, nucleon positions) ---
    let root = BitMapBackend::new(&output, (1400, 1000)).into_drawing_area();
    root.fill(&WHITE)?;

    root.draw_text(
        &format!(
            "Nucleon Distribution Function of {} (A={}, Z={}, RMS r={:.3} fm, samples={})",
            nucleus_name,
            nucleus.n(),
            nucleus.z(),
            rms_radius,
            radii.len()
        ),
        &TextStyle::from(("sans-serif", 16).into_font()).color(&BLACK),
        (20, 15),
    )?;
    let (_, root) = root.split_vertically(40);

    let (top, bottom) = root.split_vertically(480);
    let (top_left, top_right) = top.split_horizontally(700);
    let (bottom_left, bottom_right) = bottom.split_horizontally(700);

    draw_histogram_panel(
        &top_left,
        &format!("Radial distribution rho(r)*r^2 of {}", nucleus_name),
        "Radius r (fm)",
        &r_hist,
        min_r,
        max_r,
    )?;

    draw_histogram_panel(
        &top_right,
        "Azimuthal distribution (phi)",
        "phi (rad)",
        &phi_hist,
        0.0,
        TWO_PI,
    )?;

    draw_histogram_panel(
        &bottom_left,
        "Polar angle distribution (theta)",
        "theta (rad)",
        &theta_hist,
        0.0,
        PI,
    )?;

    // --- Bottom-right: nucleon positions in a single generated nucleus ---
    let mut chart_right = ChartBuilder::on(&bottom_right)
        .caption(
            format!("Nucleon Positions in a Single {} Nucleus", nucleus_name),
            ("sans-serif", 16).into_font(),
        )
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(50)
        .build_cartesian_2d(-max_r * 1.1..max_r * 1.1, -max_r * 1.1..max_r * 1.1)?;

    chart_right
        .configure_mesh()
        .x_desc("X (fm)")
        .y_desc("Y (fm)")
        .axis_desc_style(("sans-serif", 12))
        .draw()?;

    let mut protons = Vec::new();
    let mut neutrons = Vec::new();
    for (x, y, _z, is_proton) in &positions {
        if *is_proton {
            protons.push((*x, *y));
        } else {
            neutrons.push((*x, *y));
        }
    }

    let neutron_style = ShapeStyle {
        color: RED.mix(0.7).into(),
        filled: true,
        stroke_width: 1,
    };
    chart_right.draw_series(
        neutrons
            .iter()
            .map(|&(x, y)| Circle::new((x, y), 5, neutron_style)),
    )?;

    let proton_style = ShapeStyle {
        color: BLUE.mix(0.7).into(),
        filled: true,
        stroke_width: 1,
    };
    chart_right.draw_series(
        protons
            .iter()
            .map(|&(x, y)| Circle::new((x, y), 5, proton_style)),
    )?;

    // Manually drawn legend (background box + colored circles + labels) directly on the
    // drawing area, in screen-pixel coordinates - avoids plotting spurious data points.
    let legend_x = 20;
    let legend_y = 20;
    let legend_bg = Rectangle::new(
        [(legend_x, legend_y), (legend_x + 110, legend_y + 55)],
        ShapeStyle {
            color: WHITE.mix(0.85).into(),
            filled: true,
            stroke_width: 1,
        },
    );
    bottom_right.draw(&legend_bg)?;
    let legend_border = Rectangle::new(
        [(legend_x, legend_y), (legend_x + 110, legend_y + 55)],
        ShapeStyle {
            color: BLACK.into(),
            filled: false,
            stroke_width: 1,
        },
    );
    bottom_right.draw(&legend_border)?;

    let label_style = TextStyle::from(("sans-serif", 12).into_font()).color(&BLACK);
    bottom_right.draw(&Circle::new((legend_x + 15, legend_y + 15), 5, proton_style))?;
    bottom_right.draw_text("Protons (p)", &label_style, (legend_x + 25, legend_y + 8))?;
    bottom_right.draw(&Circle::new((legend_x + 15, legend_y + 40), 5, neutron_style))?;
    bottom_right.draw_text("Neutrons (n)", &label_style, (legend_x + 25, legend_y + 33))?;

    root.present()?;

    println!("Plot saved to {}", output);

    Ok(())
}
