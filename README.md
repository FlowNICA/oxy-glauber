# Oxy-Glauber

[![Crates.io](https://img.shields.io/crates/v/oxy-glauber.svg)](https://crates.io/crates/oxy-glauber)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

A Rust implementation of the Monte Carlo Glauber Model for Heavy-Ion Collisions, based on the [TGlauberMC v3.3.2](https://tglaubermc.hepforge.org) C++ code.

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
- **ROOT output**: Write results directly to ROOT TTrees using the `oxyroot` crate
- **Command-line interface**: Easy-to-use examples with argument parsing

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
oxy-glauber = "0.3.3"
```

Or clone the repository and build:
```bash
git clone https://github.com/FlowNICA/oxy-glauber
cd oxy-glauber
cargo build --release
```

# Quick Start

## Running the Examples

The simplest way to get started is to run the provided examples:

### Run with default parameters (Pb+Pb, 68 mb, 10000 events)
```bash
cargo run --example run_save_ntuple
```

### Run with custom parameters
```bash
cargo run --example run_save_ntuple -- \
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
cargo run --example run_save_ntuple -- \
    --nevents 1000 \
    --signn -5360  # 5.36 TeV
```

### Run the smearing example
```bash
cargo run --example run_smear_ntuple -- \
    --nevents 1000 \
    --bmax 15.0
```

### Run the basic example with fewer branches
```bash
cargo run --example run_glauber -- \
    --nevents 1000 \
    --sysA Pb \
    --sysB Pb
```

## Command-line Options

All examples support the following common options:

| Option | Description | Default |
|--------|-------------|---------|
| `--nevents N`   |	Number of events to generate |	10000 |
| `--sysA NAME`   |	Name of nucleus A |	Pbpnrw |
| `--sysB NAME`   |	Name of nucleus B |	Pbpnrw |
| `--signn VAL`   |	σ_NN in mb (negative = beam energy in GeV) |	68.0 |
| `--mind VAL`    | Minimum nucleon distance in fm |	0.4 |
| `--omega VAL`	  | Omega parameter for NN profile |	0.3 |
| `--seed VAL`    | Random seed |	42 |
| `--output FILE` |	Output file name |	Auto-generated |
| `--help	Print` | help message |	- |

Additional options for specific examples:
- `run_save_ntuple`: `--sigwidth`, `--noded`
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
    let mut glauber = TGlauberMC::new("Pb", "Pb", 68.0, 0.0, 0.0);
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
    let mut rng = rand::thread_rng();
    
    // Create Glauber model for Pb+Pb at 5.02 TeV (energy -> cross section)
    let mut glauber = TGlauberMC::new("Pb", "Pb", -5020.0, 0.0, 0.0);
    
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
See the `TGlauNucleus::lookup` function in the `src/nucleus.rs` for the complete list.

# NN Profile Types
The `omega` parameter controls the nucleon-nucleon interaction profile:

| Omega value | Profile Type | Description |
|-------------|--------------|-------------|
| < 0 |	Hard sphere |	Default  approximation |
| 0.0 - 2.0 |	Gamma distribution | Parameterized by ω |
| 7	| HIJING | Based on HIJING model |
| 8 |	PYTHIA | Based on PYTHIA model |
| 9 - 11 |	TRENTO | w = omega - 9 |

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
Current version: 0.3.3
Based on TGlauberMC v3.3.2
