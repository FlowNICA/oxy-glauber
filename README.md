# Oxy-Glauber

[![Crates.io](https://img.shields.io/crates/v/oxy-glauber.svg)](https://crates.io/crates/oxy-glauber)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

A Rust implementation of the Monte Carlo Glauber Model for Heavy-Ion Collisions, based on the [TGlauberMC v3.3.2](https://tglaubermc.hepforge.org) C++ code.

**NOTE**: This is a very early version, developed with AI-assisted tools. It may not yet be ready for use in a formal analysis. While the general cross-section, impact parameter, Npart, and Ncoll appear to be generated correctly, bugs are still very much expected.

## Overview

Oxy-Glauber simulates the geometry of nuclear collisions and calculates quantities like:

- Number of participating nucleons (`Npart`)
- Number of binary collisions (`Ncoll`)
- Impact parameter (`B`)
- Eccentricities (`ε₂`, `ε₃`, `ε₄`, `ε₅`)
- Participant plane angles (`Ψ₂`, `Ψ₃`, `Ψ₄`, `Ψ₅`)
- And many more event-by-event observables

The code supports a wide range of nuclei, deformation parameters, and nucleon-nucleon interaction profiles.

## Features

- **Multiple nucleus types**: Pb, Au, Cu, O, Ne, U, p, d, and many more
- **Deformed nuclei**: Support for β₂, β₃, β₄ deformation parameters
- **NN interaction profiles**: 
  - Hard sphere
  - Gamma distribution (ω parameter)
  - HIJING-based
  - PYTHIA-based
  - TRENTO-based
- **Energy-dependent cross sections**: Automatic calculation from beam energy
- **Multi-threaded parallel execution**: Automatically uses all available CPU cores for large event counts
- **ROOT output**: Write results directly to LZMA-compressed ROOT TTrees using the [`oxiroot`](https://github.com/mathieuouillon/oxiroot) crate
- **Parquet output**: Write results to size-optimized Apache Parquet files using the `parquet` crate
- **Command-line interface**: Ready-to-run binaries with argument parsing

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
oxy-glauber = "0.3.6"
```

Or clone the repository and build:
```bash
git clone https://github.com/FlowNICA/oxy-glauber
cd oxy-glauber
cargo build --release
```

# Quick Start

## Running the Examples

The simplest way to get started is to run the provided binaries:

### Run with default parameters (Pbpnrw+Pbpnrw, 68 mb, 10000 events)
```bash
cargo run --release --bin run_save_ntuple
```

### Run with custom parameters
```bash
cargo run --release --bin run_save_ntuple -- \
    --nevents 5000 \
    --sysA Pb \
    --sysB Pb \
    --signn 68.0 \
    --mind 0.4 \
    --omega 0.3 \
    --output my_output.root
```

### Run with energy instead of cross section (signn negative = energy in GeV)
```bash
cargo run --release --bin run_save_ntuple -- \
    --nevents 1000 \
    --signn -5360  # 5.36 TeV
```

### Run with default parameters, saving to Apache Parquet instead of ROOT
`run_save_parquet` generates the exact same events as `run_save_ntuple` (same options,
same defaults, same branches) but writes them to a size-optimized `.parquet` file
instead of a ROOT TTree - see [Parquet Output](#parquet-output) below.
```bash
cargo run --release --bin run_save_parquet
```

### Run with custom parameters, saving to Parquet
```bash
cargo run --release --bin run_save_parquet -- \
    --nevents 5000 \
    --sysA Pb \
    --sysB Pb \
    --signn 68.0 \
    --mind 0.4 \
    --omega 0.3 \
    --output my_output.parquet
```

### Run the smearing binary
```bash
cargo run --release --bin run_smear_ntuple -- \
    --nevents 1000 \
    --bmax 15.0
```

### Run the basic binary with fewer branches
```bash
cargo run --release --bin run_glauber -- \
    --nevents 1000 \
    --sysA Pb \
    --sysB Pb
```

## Command-line Options

All binaries (`run_save_ntuple`, `run_save_parquet`, `run_smear_ntuple`, `run_glauber`)
accept the same option names, though a few defaults differ per binary:

| Option | Description | Default |
|--------|-------------|---------|
| `--nevents N`   | Number of events to generate | 10000 (1000 for `run_glauber`/`run_smear_ntuple`) |
| `--sysA NAME`   | Name of nucleus A | Pbpnrw (Pb for `run_glauber`) |
| `--sysB NAME`   | Name of nucleus B | Pbpnrw (Pb for `run_glauber`) |
| `--signn VAL`   | σ_NN in mb (negative = beam energy in GeV) | 68.0 |
| `--mind VAL`    | Minimum nucleon distance in fm | 0.4 |
| `--omega VAL`   | Omega parameter for NN profile | 0.0/hard sphere for `run_save_ntuple`/`run_save_parquet`, 0.3/Gamma for `run_glauber`/`run_smear_ntuple` |
| `--seed VAL`    | Random seed | 42 |
| `--output FILE` | Output file name | Auto-generated from run parameters (fixed `glauber_output.root` for `run_glauber`) |
| `--help`        | Print help message | - |

Additional options for specific binaries:
- `run_save_ntuple`: `--sigwidth`, `--noded`
- `run_save_parquet`: `--sigwidth`, `--noded`
- `run_glauber`: `--bmin`, `--bmax`
- `run_smear_ntuple`: `--bmin`, `--bmax`

 # Multi-Threading Support

 Oxy-Glauber features automatic multi-threading support using the Rayon crate. This provides significant performance improvements for large event generation.

 ## How It Works
- **Automatic detection**: When generating more than 10,000 events, the code automatically switches to parallel mode
- **Maximum thread usage**: Uses all available CPU cores
- **Even distribution**: Events are evenly distributed across threads
- **Deterministic results**: Each thread uses a deterministic seed based on thread ID and event index
- **Progress reporting**: Shows real-time progress with generation rate and elapsed time

## Performance Example
```bash
╔════════════════════════════════════════════════════════════════╗
║                    PARALLEL EVENT GENERATION                   ║
╠════════════════════════════════════════════════════════════════╣
║  Total events:      100000                                     ║
║  CPU threads:           8                                      ║
║  Events per thread:   12500                                    ║
╚════════════════════════════════════════════════════════════════╝

  Progress:    100/100000 events (  0.1%) | Rate:    124.3 events/sec | Elapsed: 804.4ms
  Progress:    200/100000 events (  0.2%) | Rate:    248.5 events/sec | Elapsed: 804.8ms
  ...
  Progress: 100000/100000 events (100.0%) | Rate:   1245.6 events/sec | Elapsed: 80.3s

╔════════════════════════════════════════════════════════════════╗
║                    GENERATION COMPLETE                         ║
╠════════════════════════════════════════════════════════════════╣
║  Total events:     100000                                      ║
║  Time elapsed:      80.28s                                     ║
║  Event rate:       1245.6 events/sec                           ║
║  Threads used:          8                                      ║
╚════════════════════════════════════════════════════════════════╝
```

## Manual Thread Control
To manually control threading behavior:
```rust
use oxy_glauber::TGlauberMC;
use rayon::ThreadPoolBuilder;

// Limit to 4 threads
let pool = ThreadPoolBuilder::new()
    .num_threads(4)
    .build()
    .unwrap();

pool.install(|| {
    let mut glauber = TGlauberMC::new("Pb", "Pb", 68.0, 0.0, 0.0).unwrap();
    let events = glauber.run_parallel(100000, None);
});
```

## Single-Threaded Mode
For smaller event counts or debugging, the code runs in single-threaded mode:
```bash
╔════════════════════════════════════════════════════════════════╗
║                  SINGLE-THREADED GENERATION                    ║
╠════════════════════════════════════════════════════════════════╣
║  Total events:       1000                                      ║
║  Mode:              Single-threaded                            ║
║  (Use >10000 events for automatic parallel mode)               ║
║  CPU threads available:           8                            ║
╚════════════════════════════════════════════════════════════════╝
```

# Using as a Library
```rust
use oxy_glauber::TGlauberMC;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut rng = rand::rng();
    
    // Create Glauber model for Pb+Pb at 5.02 TeV (energy -> cross section)
    let mut glauber = TGlauberMC::new("Pb", "Pb", -5020.0, 0.0, 0.0)?;
    
    // Configure parameters
    glauber.set_min_distance(0.4);
    glauber.set_omega(0.3);  // Gamma distribution for NN profile
    glauber.set_bmax(20.0);
    
    // Generate events (automatically uses parallel mode for >10000 events)
    let nevents = 100000;
    let events = glauber.run(nevents, &mut rng, None);
    
    // Or force parallel mode
    // let events = glauber.run_parallel(nevents, None);
    
    // Access results
    for event in &events[0..10] {
        println!("B={:.3}, Npart={:.0}, Ncoll={:.0}, ε₂={:.4}",
            event.b, event.npart, event.ncoll, event.ecc2);
    }
    
    println!("Total cross section: {:.3} +/- {:.3} mb",
        glauber.total_xsect(), glauber.total_xsect_err());
    
    Ok(())
}
```

# Supported Nuclei
See `data/nuclei.ron` for the complete list and the parameters of each nucleus. The file is embedded in the library at compile time; to add a nucleus, add an entry there and rebuild. Requesting a name that is not in the file makes `TGlauNucleus::new` / `TGlauberMC::new` return a `NucleusError::Unknown` error.

Nucleons can also be taken from precomputed configurations (e.g. from ab-initio calculations) with `profile: FromFile` and a `file` parameter:
```ron
"MyO16": (n: 16, z: 8, profile: FromFile, file: "o16_configurations.dat"),
```
The file contains one configuration per line: `x y z` (fm) for each nucleon, optionally followed by an isospin column (`x y z isospin`, 1 = proton, 0 = neutron). Blank lines and `#` comments are ignored, relative paths are resolved against the working directory, and each event uses a randomly chosen configuration in a random orientation. The `He3`, `H3`, `He4`, `C` and `O` entries use this profile but have no file set yet, so they can't be used until one is added.

# NN Profile Types
The `omega` parameter controls the nucleon-nucleon interaction profile:

| Omega value | Profile Type | Description |
|-------------|--------------|-------------|
| ≤ 0 |	Hard sphere |	Default approximation (no NN profile) |
| 0 < ω < 2 |	Gamma distribution | Parameterized by ω |
| 7	| HIJING | Based on HIJING model |
| 8 |	PYTHIA | Based on PYTHIA model |
| 9 ≤ ω < 11 |	TRENTO | w = omega - 9 |

# Output Format
Results are written to ROOT files as TTrees with the following branches:
- `Npart`, `Ncoll`, `Nhard`, `Nmpi`: Multiplicity and collision counters
- `B`, `BNN`: Impact parameters
- `Ncollpp`, `Ncollpn`, `Ncollnn`: Collision type counters
- `VarX`, `VarY`, `VarXY`: Participant distribution variances
- `NpartA`, `NpartB`, `Npart0`: Participant counts
- `SpecA`, `SpecB`: Spectator counts
- `Ecc1`-`Ecc5`: Eccentricities
- `Psi1`-`Psi5`: Participant plane angles

# Parquet Output
`run_save_parquet` writes the identical set of branches listed above to an
[Apache Parquet](https://parquet.apache.org) file instead of a ROOT TTree, using the
`parquet` crate directly (no Arrow dependency required). This is a convenient,
language-agnostic alternative when you want to read the results with `pandas`,
`polars`, `DuckDB`, Spark, etc. instead of ROOT.

The writer is tuned to produce the smallest file that's still fast to read:
- **ZSTD compression** at level 19 - the level only affects write time, not read
  (decompression) speed, so it's set high for a smaller file at no read-time cost.
- **PLAIN encoding, no dictionary** - measured against the alternatives (dictionary
  encoding, `BYTE_STREAM_SPLIT`) on real event data; PLAIN+ZSTD won because these
  columns are full of exact repeated values (many counters at `0`, `Weight` at `1.0`,
  small repeated integer-valued counts) that ZSTD's own match-finding compresses
  better on raw interleaved floats than either alternative encoding does.
- **A single row group** - all columns are already fully buffered in memory before
  writing, so one row group per file maximizes how much repetition ZSTD can see per
  column, with no downside since the file is meant to be read as a whole.

On a 200k-event Pb+Pb run this produces a file roughly **half the size** of the
equivalent ROOT TTree (12.0 MB vs 24.5 MB), with no loss of precision (all columns stay
`f32`, matching the ROOT branches exactly).

# Performance
Oxy-Glauber leverages Rust's zero-cost abstractions and Rayon's work-stealing thread pool for excellent performance:
| Events | Threads | Time (s) | Rate (events/sec) |
|--------|---------|----------|-------------------|
| 1,000 | 1 | 0.28 | ~3,605 |
| 10,000 | 1 | 2.6 | ~3,896 |
| 100,000 | 16 | 2.8 | ~35,719 |
| 1,000,000 | 16 | 27.9 | ~35,838 |

*Performance measured on an 16-core CPU with Au+Au collisions at the energy 2.4 GeV.*

# References
This implementation is based on the TGlauberMC C++ code:
- TGlauberMC v3.3: https://tglaubermc.hepforge.org
- "Glauber predictions for oxygen and neon collisions at the LHC", https://arxiv.org/abs/2507.05853
- "Improved Monte Carlo Glauber predictions at present and future nuclear colliders", https://arxiv.org/abs/1710.07098
- "Improved version of the PHOBOS Glauber Monte Carlo", https://arxiv.org/abs/1408.2549

# License
This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

See the LICENSE file for details.

# Contributing
Contributions are welcome! Please submit issues and pull requests on the GitHub repository.

# Version
Current version: 0.3.6
Based on TGlauberMC v3.3.2
