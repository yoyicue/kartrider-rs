//! Parity bridge for scripts/verify_kart_physics_state_parity.py.
//!
//! Reads a JSON fixture produced by that script (Python `dataclasses.asdict`
//! output), deserializes every vehicle state / parameter set / rail context /
//! topology case, validates, recomputes topology decisions with
//! `advance_topology`, and prints the re-serialized JSON to stdout. The Python
//! script then asserts the round-trip is structurally identical (floats via
//! f64 bit patterns) and that Python-invalid fixtures are rejected here.

use kart_physics::params::VehicleParameters;
use kart_physics::rail::RailRuntimeContext;
use kart_physics::state::VehicleState;
use kart_physics::topology::{
    advance_topology, TopologyDecision, TopologyObservation, TopologyState,
};
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct TopologyCase {
    state: TopologyState,
    observation: TopologyObservation,
}

#[derive(Deserialize)]
struct InvalidCase<T> {
    value: T,
    expect_error_contains: String,
}

#[derive(Deserialize)]
struct ParityInput {
    states: Vec<VehicleState>,
    params: Vec<VehicleParameters>,
    rail_contexts: Vec<RailRuntimeContext>,
    topology_cases: Vec<TopologyCase>,
    invalid_states: Vec<InvalidCase<VehicleState>>,
    invalid_params: Vec<InvalidCase<VehicleParameters>>,
    invalid_rail_contexts: Vec<InvalidCase<RailRuntimeContext>>,
}

#[derive(Serialize)]
struct InvalidResult {
    expect_error_contains: String,
    error: Option<String>,
}

#[derive(Serialize)]
struct ParityOutput {
    states: Vec<VehicleState>,
    params: Vec<VehicleParameters>,
    rail_contexts: Vec<RailRuntimeContext>,
    topology_decisions: Vec<TopologyDecision>,
    invalid_states: Vec<InvalidResult>,
    invalid_params: Vec<InvalidResult>,
    invalid_rail_contexts: Vec<InvalidResult>,
}

fn reject<T, F: Fn(&T) -> Result<(), String>>(cases: &[InvalidCase<T>], validate: F) -> Vec<InvalidResult> {
    cases
        .iter()
        .map(|case| InvalidResult {
            expect_error_contains: case.expect_error_contains.clone(),
            error: validate(&case.value).err(),
        })
        .collect()
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: state_trace <parity-input.json>");
    let text = std::fs::read_to_string(&path).expect("failed to read input file");
    let input: ParityInput = serde_json::from_str(&text).expect("failed to parse input JSON");

    for (index, state) in input.states.iter().enumerate() {
        if let Err(error) = state.validate() {
            eprintln!("states[{index}] unexpectedly rejected: {error}");
            std::process::exit(1);
        }
    }
    for (index, params) in input.params.iter().enumerate() {
        if let Err(error) = params.validate() {
            eprintln!("params[{index}] unexpectedly rejected: {error}");
            std::process::exit(1);
        }
    }
    for (index, context) in input.rail_contexts.iter().enumerate() {
        if let Err(error) = context.validate() {
            eprintln!("rail_contexts[{index}] unexpectedly rejected: {error}");
            std::process::exit(1);
        }
    }

    let mut topology_decisions = Vec::with_capacity(input.topology_cases.len());
    for (index, case) in input.topology_cases.iter().enumerate() {
        match advance_topology(&case.state, &case.observation) {
            Ok(decision) => topology_decisions.push(decision),
            Err(error) => {
                eprintln!("topology_cases[{index}] unexpectedly rejected: {error}");
                std::process::exit(1);
            }
        }
    }

    let output = ParityOutput {
        states: input.states,
        params: input.params,
        rail_contexts: input.rail_contexts,
        topology_decisions,
        invalid_states: reject(&input.invalid_states, VehicleState::validate),
        invalid_params: reject(&input.invalid_params, VehicleParameters::validate),
        invalid_rail_contexts: reject(
            &input.invalid_rail_contexts,
            RailRuntimeContext::validate,
        ),
    };
    println!(
        "{}",
        serde_json::to_string_pretty(&output).expect("failed to serialize output")
    );
}
