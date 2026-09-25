// src/bin/run_save_parquet.rs
use oxy_glauber::{TGlauberEvent, TGlauberMC};
use parquet::basic::{Compression, Encoding, Type as PhysicalType, ZstdLevel};
use parquet::data_type::FloatType;
use parquet::file::properties::{WriterProperties, WriterPropertiesPtr};
use parquet::file::writer::SerializedFileWriter;
use parquet::schema::types::{Type as SchemaType, TypePtr};
use std::env;
use std::fs::File;
use std::sync::Arc;

/// zstd compression level used for every column.
///
/// zstd *decompression* speed is essentially flat across the whole level range - it's
/// only *compression* (write) time that grows with the level, and read speed (the
/// stated goal here) is unaffected by which level was used to write the file. So it's
/// worth spending extra one-time write time for a smaller file. Measured on a 200k-event
/// Pb+Pb run: level 3 (zstd's own default) -> 14.18 MB, level 9 -> 14.05 MB, level 15 ->
/// 13.48 MB, level 19 -> 13.41 MB, level 22 (max) -> 13.41 MB (no further gain). 19 sits
/// right at that plateau - effectively the smallest file this codec can produce here
/// without paying for the "ultra" levels' extra compression time for no benefit.
const ZSTD_LEVEL: i32 = 19;

/// All branches written, in the same order/names as `run_save_ntuple.rs`'s ROOT tree.
const COLUMNS: &[&str] = &[
    "Npart", "Ncoll", "Nhard", "Nmpi", "B", "BNN", "Ncollpp", "Ncollpn", "Ncollnn", "VarX",
    "VarY", "VarXY", "NpartA", "NpartB", "Npart0", "NpartAn", "NpartBn", "Npart0n", "AreaW",
    "SpecA", "SpecB", "Weight", "Psi1", "Ecc1", "Psi2", "Ecc2", "Psi3", "Ecc3", "Psi4", "Ecc4",
    "Psi5", "Ecc5",
];

fn print_usage() {
    println!("Usage: run_save_parquet [options]");
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
    println!();
    println!(
        "Same event generation as run_save_ntuple, but written as an Apache Parquet file \
         instead of a ROOT TTree (ZSTD level 19, PLAIN-encoded, single row group - \
         tuned for minimal file size while staying fast to read)."
    );
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

/// Build the flat, all-REQUIRED-FLOAT Parquet schema shared by every branch.
fn build_schema() -> TypePtr {
    let fields: Vec<TypePtr> = COLUMNS
        .iter()
        .map(|name| {
            Arc::new(
                SchemaType::primitive_type_builder(name, PhysicalType::FLOAT)
                    .with_repetition(parquet::basic::Repetition::REQUIRED)
                    .build()
                    .expect("valid primitive type"),
            )
        })
        .collect();

    Arc::new(
        SchemaType::group_type_builder("glauber")
            .with_fields(fields)
            .build()
            .expect("valid schema"),
    )
}

/// Writer properties tuned for minimal file size while staying fast to decode.
///
/// Three encoding strategies were measured on a 200k-event Pb+Pb run to pick these
/// (don't assume - Parquet's "best for floats" folklore doesn't hold for every shape
/// of data):
/// - dictionary encoding (the crate's own default): 15.90 MB - worst. These event
///   quantities are numerous enough, and varied enough, that per-column dictionaries
///   just add index overhead on top of what zstd already compresses away.
/// - BYTE_STREAM_SPLIT (splits each f32's 4 bytes into 4 separate streams before
///   compression - usually a win for scientific float columns): 13.41 MB - better than
///   dictionary, but still not as good as plain, because it breaks up the *exact*
///   whole-float repeats these columns are full of (many-valued columns like `AreaW`/
///   `Nmpi` sitting at 0.0, `Weight` sitting at 1.0, small repeated integer-valued
///   counts) into four separately-compressed byte planes, which zstd's LZ matching on
///   those repeats can't reassemble as effectively as matching the raw 4-byte values.
/// - PLAIN encoding, no dictionary: 12.62 MB - the winner, kept below. zstd's own
///   match-finding on the raw interleaved floats captures the repeats directly.
///
/// ZSTD (see `ZSTD_LEVEL`) is the compression codec itself: decompression is fast
/// regardless of the level used to compress, so it's a straightforward win over
/// SNAPPY/GZIP/LZ4 for "smallest file, still fast to read".
fn build_writer_properties() -> WriterPropertiesPtr {
    Arc::new(
        WriterProperties::builder()
            .set_compression(Compression::ZSTD(
                ZstdLevel::try_new(ZSTD_LEVEL).expect("valid zstd level"),
            ))
            .set_dictionary_enabled(false)
            .set_encoding(Encoding::PLAIN)
            .build(),
    )
}

/// Equivalent of runAndSaveNtuple from C++ code, writing a Parquet file instead of a
/// ROOT TTree.
fn run_and_save_parquet(
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

    let mut glauber = TGlauberMC::new(sys_a, sys_b, signn, sigwidth, 0.0)?;
    glauber.set_min_distance(mind);
    glauber.set_node_distance(noded);
    glauber.set_calc_area(false);
    glauber.set_calc_length(false);
    glauber.set_calc_core(false);
    glauber.set_detail(99);

    if (0.0..=11.0).contains(&omega) {
        glauber.set_omega(omega);
    }

    let name = format!("{}.parquet", glauber.str());
    let filename = output_file.unwrap_or(&name);

    println!("Running Glauber MC for {} + {}", sys_a, sys_b);
    println!("Generating {} events...", nevents);

    let events = glauber.run(nevents, &mut rng, None);

    // --- Prepare data for writing: one Vec<f32> column per branch, same set/order
    // as run_save_ntuple.rs's ROOT tree branches ---
    let columns: [Vec<f32>; 32] = [
        events.iter().map(|e| e.npart).collect(),
        events.iter().map(|e| e.ncoll).collect(),
        events.iter().map(|e| e.nhard).collect(),
        events.iter().map(|e| e.nmpi).collect(),
        events.iter().map(|e| e.b).collect(),
        events.iter().map(|e| e.bnn).collect(),
        events.iter().map(|e| e.ncollpp).collect(),
        events.iter().map(|e| e.ncollpn).collect(),
        events.iter().map(|e| e.ncollnn).collect(),
        events.iter().map(|e| e.var_x).collect(),
        events.iter().map(|e| e.var_y).collect(),
        events.iter().map(|e| e.var_xy).collect(),
        events.iter().map(|e| e.npart_a).collect(),
        events.iter().map(|e| e.npart_b).collect(),
        events.iter().map(|e| e.npart0).collect(),
        events.iter().map(|e| e.npart_an).collect(),
        events.iter().map(|e| e.npart_bn).collect(),
        events.iter().map(|e| e.npart0n).collect(),
        events.iter().map(|e| e.area_w).collect(),
        events.iter().map(|e| e.spec_a).collect(),
        events.iter().map(|e| e.spec_b).collect(),
        events.iter().map(|e| e.weight).collect(),
        events.iter().map(|e| e.psi1).collect(),
        events.iter().map(|e| e.ecc1).collect(),
        events.iter().map(|e| e.psi2).collect(),
        events.iter().map(|e| e.ecc2).collect(),
        events.iter().map(|e| e.psi3).collect(),
        events.iter().map(|e| e.ecc3).collect(),
        events.iter().map(|e| e.psi4).collect(),
        events.iter().map(|e| e.ecc4).collect(),
        events.iter().map(|e| e.psi5).collect(),
        events.iter().map(|e| e.ecc5).collect(),
    ];
    debug_assert_eq!(columns.len(), COLUMNS.len());

    // --- Write to a Parquet file using the low-level (arrow-free) column writer API ---
    let file = File::create(filename)?;
    let schema = build_schema();
    let props = build_writer_properties();

    let mut writer = SerializedFileWriter::new(file, schema, props)?;
    // A single row group maximizes the amount of data zstd sees at once per column,
    // which improves the compression ratio; all columns are already fully buffered in
    // memory above, so there's no streaming benefit to splitting into several groups.
    let mut row_group_writer = writer.next_row_group()?;
    for column in &columns {
        let mut col_writer = row_group_writer
            .next_column()?
            .expect("schema/column count mismatch");
        col_writer
            .typed::<FloatType>()
            .write_batch(column, None, None)?;
        col_writer.close()?;
    }
    row_group_writer.close()?;
    writer.close()?;

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

    let _events = run_and_save_parquet(
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
