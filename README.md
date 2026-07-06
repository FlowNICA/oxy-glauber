# Oxy-Glauber

<!--[![Crates.io](https://img.shields.io/crates/v/oxy-glauber.svg)](https://crates.io/crates/oxy-glauber)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)-->

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
- **ROOT output**: Write results directly to ROOT TTrees using the `oxyroot` crate
- **Command-line interface**: Easy-to-use examples with argument parsing

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
oxy-glauber = "0.3.3"
