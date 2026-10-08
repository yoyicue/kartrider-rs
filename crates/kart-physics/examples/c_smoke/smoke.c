/*
 * kart-physics C ABI smoke test (milestone M4 + the 2026-07-29 capability
 * completion, the 2026-07-30 CTRL drift chain, and the 2026-08-01
 * current-family ordinary-N2O chain, ABI v4).
 *
 * C99. Four scenarios, all printed as f64 IEEE-754 bit patterns so
 * scripts/verify_kart_physics_ffi_smoke.py can compare bit-exactly against
 * the Python reference engine:
 *
 * 1. Flat ground plane via the world vtable (the M4 scenario): clock-init
 *    frame at tick 1000, then 20 frames of 16 ms with keys=UP. Unprefixed
 *    output keys.
 * 2. RAIL capture cycle through the ABI v2 open-mapping world slots:
 *    query_rail_contact / query_rail_entry_boundary / rail_query_contexts /
 *    rail_entry_contexts / sweep_point callbacks serve the payloads
 *    recorded from the Python reference run (functional schedule: state-1
 *    selector with alignment gate inputs while aligning, current-state
 *    selector on sustain, negative once the kart passes the path end at
 *    x >= 1.35). Clock-init + 40 frames of 16 ms UP. The engine must enter
 *    AND exit the RAIL domain; per-frame state is printed with "rail.f<N>."
 *    prefixed keys, the final state with "rail." prefixed keys.
 * 3. CTRL drift activation over four 2 ms frames: press LEFT+CTRL, hold,
 *    release CTRL, then let the short B08 window expire. Per-frame state
 *    and drift selector fields use the "drift.f<N>." prefix.
 * 4. ITEM_USE current-family N2O activation over four 2 ms frames from one
 *    item6 in slot0: press, hold, release, press again on the now-empty
 *    inventory. The v4 state/parameter tail uses the "boost." prefix.
 *
 * The binary also verifies the ABI version handshake before creating any
 * engine: hosts MUST create engines with kart_physics_create_v2(...,
 * KP_ABI_VERSION), which the library refuses (NULL + KP_ERR_ABI_VERSION
 * detail) on a version mismatch. Plain kart_physics_create is the v1
 * compatibility shim whose engines read only the v1 vtable prefix and
 * ignore the v2-only slots; the v1-shim degradation path is probed
 * separately by scripts/verify_kart_physics_ffi_smoke.py. The cheap
 * kart_physics_abi_version() self-check below is kept as a loud pre-flight.
 *
 * Build/run (macOS; see README.md for details):
 *   cd rust && CARGO_INCREMENTAL=0 CARGO_TARGET_DIR=/tmp/kart-rust-target \
 *     cargo build -p kart-physics
 *   cc -std=c99 -Wall -Wextra -I rust/crates/kart-physics/include \
 *     rust/crates/kart-physics/examples/c_smoke/smoke.c \
 *     -L /tmp/kart-rust-target/debug -lkart_physics -o /tmp/kart_ffi_smoke
 *   DYLD_LIBRARY_PATH=/tmp/kart-rust-target/debug /tmp/kart_ffi_smoke
 */

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "kart_physics.h"

static uint64_t f64_bits(double value) {
    uint64_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static void set_string(KpString64 *slot, const char *text) {
    size_t len = strlen(text);
    if (len > KP_STRING_CAPACITY) {
        fprintf(stderr, "smoke: string too long: %s\n", text);
        exit(2);
    }
    slot->len = (uint8_t)len;
    memcpy(slot->data, text, len);
}

static void set_json(KpJsonBlob *blob, const char *text) {
    size_t len = strlen(text);
    if (len > KP_JSON_CAPACITY) {
        fprintf(stderr, "smoke: JSON payload too long (%zu bytes)\n", len);
        exit(2);
    }
    blob->len = (uint32_t)len;
    memcpy(blob->data, text, len);
}

static void check_status(int32_t status, const char *what) {
    if (status != KP_OK) {
        char message[256];
        kart_physics_last_error_message(message, sizeof(message));
        fprintf(stderr, "smoke: %s failed: status=%d: %s\n", what, (int)status,
                message);
        exit(1);
    }
}

static void print_vec3_bits(const char *prefix, const char *name,
                            const double value[3]) {
    int i;
    for (i = 0; i < 3; i++) {
        printf("%s%s.%d=%016" PRIx64 "\n", prefix, name, i, f64_bits(value[i]));
    }
}

static void print_state_bits(const char *prefix, const KpVehicleState *state) {
    int i;
    print_vec3_bits(prefix, "position", state->position);
    print_vec3_bits(prefix, "velocity", state->velocity);
    print_vec3_bits(prefix, "angular_velocity", state->angular_velocity);
    for (i = 0; i < 9; i++) {
        printf("%smatrix.%d=%016" PRIx64 "\n", prefix, i,
               f64_bits(state->matrix[i]));
    }
    printf("%stime_ms=%" PRIu64 "\n", prefix, state->time_ms);
    printf("%sstep_index=%" PRIu64 "\n", prefix, state->step_index);
    printf("%sdomain=%u\n", prefix, (unsigned)state->topology.domain);
    printf("%srail_state=%" PRId64 "\n", prefix, state->topology.rail_state);
    printf("%sscheduler_previous_tick_ms=%" PRIu32 "\n", prefix,
           state->scheduler_previous_tick_ms);
}

static void print_drift_state_bits(const char *prefix,
                                   const KpVehicleState *state) {
    print_state_bits(prefix, state);
    printf("%sselector_b08=%u\n", prefix, (unsigned)state->selector_b08);
    printf("%sselector_b5c=%u\n", prefix, (unsigned)state->selector_b5c);
    printf("%sselector_ff8=%" PRId64 "\n", prefix, state->selector_ff8);
    printf("%sdrift_timer_b0c=%016" PRIx64 "\n", prefix,
           f64_bits(state->drift_timer_b0c));
    printf("%srecovery_timer=%016" PRIx64 "\n", prefix,
           f64_bits(state->recovery_timer));
    printf("%sdrift_key_latched=%u\n", prefix,
           (unsigned)state->drift_key_latched);
}

static void print_boost_state_bits(const char *prefix,
                                   const KpVehicleState *state) {
    int i;
    printf("%sboost_state=%" PRId64 "\n", prefix, state->boost_state);
    printf("%sboost_remaining_ms=%" PRId64 "\n", prefix,
           state->boost_remaining_ms);
    for (i = 0; i < 2; i++) {
        printf("%sbooster_inventory_item_ids.%d=%" PRId64 "\n", prefix, i,
               state->booster_inventory_item_ids[i]);
        printf("%sbooster_inventory_quantities.%d=%" PRId64 "\n", prefix, i,
               state->booster_inventory_quantities[i]);
    }
    printf("%sitem_use_key_latched=%u\n", prefix,
           (unsigned)state->item_use_key_latched);
    printf("%sboost_gauge_committed_b10=%016" PRIx64 "\n", prefix,
           f64_bits(state->boost_gauge_committed_b10));
    printf("%sboost_gauge_progress_b20=%016" PRIx64 "\n", prefix,
           f64_bits(state->boost_gauge_progress_b20));
    printf("%sboost_gauge_pending_b2c=%016" PRIx64 "\n", prefix,
           f64_bits(state->boost_gauge_pending_b2c));
    printf("%sboost_gauge_history_b38=%016" PRIx64 "\n", prefix,
           f64_bits(state->boost_gauge_history_b38));
}

/* ------------------------------------------------------------------------
 * Scenario 1 (M4): flat ground plane
 * ------------------------------------------------------------------------ */

/* The canonical parity mock plane: z=0 ground, +Z normal (identical row
 * values to the Python side's plane_row() defaults). */
static void flat_query_ground(void *user_data, const double *position,
                              const double *up, double min_dot,
                              KpWorldGroundResult *out) {
    (void)user_data;
    (void)position;
    (void)up;
    (void)min_dot;
    memset(out, 0, sizeof(*out));
    set_string(&out->track_id, "mock");
    set_string(&out->node, "mock");
    out->tri_index = 7;
    out->surface_id = 3;
    out->point[0] = 0.0;
    out->point[1] = 0.0;
    out->point[2] = 0.0;
    out->normal[0] = 0.0;
    out->normal[1] = 0.0;
    out->normal[2] = 1.0;
}

/* sweep_obb_rotating_all capability present, never any candidates. */
static size_t empty_sweep(void *user_data, const double *start,
                          const double *end, const double *start_matrix,
                          const double *end_matrix, const double *half_extents,
                          KpWorldObbOverlap *out, size_t out_capacity) {
    (void)user_data;
    (void)start;
    (void)end;
    (void)start_matrix;
    (void)end_matrix;
    (void)half_extents;
    (void)out;
    (void)out_capacity;
    return 0;
}

static void run_flat_scenario(void) {
    KpVehicleParameters parameters;
    KpVehicleState state;
    KpWorldVtable world;
    KpInputs inputs;
    KartPhysicsEngine *engine;
    uint32_t tick_ms = 1000;
    int frame;

    check_status(kart_physics_parameters_default_v2(&parameters, KP_ABI_VERSION),
                 "kart_physics_parameters_default_v2");
    check_status(kart_physics_state_default_v2(&state, KP_ABI_VERSION),
                 "kart_physics_state_default_v2");

    engine = kart_physics_create_v2(&state, KP_NUMERIC_MODE_FLOAT32, KP_ABI_VERSION);
    if (engine == NULL) {
        char message[256];
        kart_physics_last_error_message(message, sizeof(message));
        fprintf(stderr, "smoke: kart_physics_create_v2 failed: %s\n", message);
        exit(1);
    }

    memset(&world, 0, sizeof(world));
    world.query_ground = flat_query_ground;
    world.sweep_obb_rotating_all = empty_sweep;

    /* Clock-init frame (zero substeps), then 20 frames of 16 ms UP. */
    memset(&inputs, 0, sizeof(inputs));
    inputs.tick_ms = tick_ms;
    check_status(kart_physics_step(engine, &inputs, &parameters, &world),
                 "kart_physics_step(clock_init)");
    for (frame = 0; frame < 20; frame++) {
        tick_ms += 16;
        inputs.tick_ms = tick_ms;
        inputs.keys = KP_KEY_UP;
        check_status(kart_physics_step(engine, &inputs, &parameters, &world),
                     "kart_physics_step");
    }

    check_status(kart_physics_get_state(engine, &state),
                 "kart_physics_get_state");
    print_state_bits("", &state);

    kart_physics_destroy(engine);
}

/* ------------------------------------------------------------------------
 * Scenario 2 (ABI v2): RAIL capture cycle through the open-mapping slots
 *
 * World payloads recorded from the Python reference run (the functional
 * schedule of the recording mock world):
 * - query_rail_contact: negative (return 0) once the kart's x passes the
 *   path end (1.35 m); a current-state selector on sustain
 *   (previous_rail_state >= 2 — the world must not publish transitions);
 *   otherwise the state-1 selector with the alignment gate inputs.
 * - sweep_point: the constant A5B840 candidate {"hit":true,"road_rail":""}.
 * - rail_query_contexts: the single straight +X context embedded in the
 *   selectors; rail_entry_contexts: empty; query_rail_entry_boundary: no
 *   match (the Python engine never calls the latter two in this scenario,
 *   but the slots are served to prove the ABI path).
 * ------------------------------------------------------------------------ */

/* The RAIL smoke plane row (identical to the Python reference world):
 * z=0 ground with the authored triangle. */
static void rail_query_ground(void *user_data, const double *position,
                              const double *up, double min_dot,
                              KpWorldGroundResult *out) {
    (void)user_data;
    (void)position;
    (void)up;
    (void)min_dot;
    memset(out, 0, sizeof(*out));
    set_string(&out->track_id, "ffi-smoke");
    set_string(&out->node, "n0");
    out->tri_index = 0;
    out->surface_id = 0;
    out->point[0] = 0.0;
    out->point[1] = 0.0;
    out->point[2] = 0.0;
    out->normal[0] = 0.0;
    out->normal[1] = 0.0;
    out->normal[2] = 1.0;
    out->has_triangle = 1;
    out->triangle[0] = -500.0;
    out->triangle[1] = -500.0;
    out->triangle[2] = 0.0;
    out->triangle[3] = 500.0;
    out->triangle[4] = -500.0;
    out->triangle[5] = 0.0;
    out->triangle[6] = 0.0;
    out->triangle[7] = 500.0;
    out->triangle[8] = 0.0;
}

static int rail_sweep_point(void *user_data, const double *start,
                            const double *end, KpJsonBlob *out) {
    (void)user_data;
    (void)start;
    (void)end;
    set_json(out, "{\"hit\":true,\"road_rail\":\"\"}");
    return 1;
}

static int rail_query_contact(void *user_data, const double *position,
                              int64_t previous_rail_state,
                              const char *owner_id, KpJsonBlob *out) {
    char json[2048];
    long long rail_state;
    (void)user_data;
    (void)owner_id;
    if (position[0] >= 1.35) {
        return 0;
    }
    rail_state = previous_rail_state >= 2 ? (long long)previous_rail_state : 1;
    snprintf(
        json, sizeof(json),
        "{\"contact_active\":true,\"rail_state\":%lld,\"path_id\":7,"
        "\"runtime\":{\"rail_dir\":[1,0,0],\"accel_be0\":4608,\"obf_6e8\":1,"
        "\"grip_720\":1,\"min_velocity\":5,\"max_velocity\":12},"
        "\"registered_query_contexts\":[{\"context_id\":0,\"points\":["
        "{\"position\":[0,0,0],\"secondary\":[1,0,0],\"normal\":[0,0,1]},"
        "{\"position\":[1.4,0,0],\"secondary\":[1,0,0],\"normal\":[0,0,1]}],"
        "\"road\":\"rail-road\",\"total_length\":1.4}]%s}",
        rail_state,
        rail_state == 1
            ? ",\"contact_exists\":true,\"contact_identifier_nonempty\":true,"
              "\"alignment_base_ms\":1"
            : "");
    set_json(out, json);
    return 1;
}

static int rail_no_entry_boundary(void *user_data, const double *position,
                                  const char *road, double plane_tolerance,
                                  KpRailEntryBoundaryMatch *out) {
    (void)user_data;
    (void)position;
    (void)road;
    (void)plane_tolerance;
    (void)out;
    return 0;
}

static size_t rail_query_contexts_cb(void *user_data, KpJsonBlob *out,
                                     size_t out_capacity) {
    (void)user_data;
    if (out_capacity >= 1) {
        set_json(out,
                 "{\"context_id\":0,\"points\":["
                 "{\"position\":[0,0,0],\"secondary\":[1,0,0],"
                 "\"normal\":[0,0,1]},"
                 "{\"position\":[1.4,0,0],\"secondary\":[1,0,0],"
                 "\"normal\":[0,0,1]}],"
                 "\"road\":\"rail-road\",\"total_length\":1.4}");
    }
    return 1;
}

static size_t rail_entry_contexts_cb(void *user_data, KpJsonBlob *out,
                                     size_t out_capacity) {
    (void)user_data;
    (void)out;
    (void)out_capacity;
    return 0;
}

#define RAIL_FRAMES 40

static void run_rail_scenario(void) {
    KpVehicleParameters parameters;
    KpVehicleState state;
    KpWorldVtable world;
    KpInputs inputs;
    KartPhysicsEngine *engine;
    uint32_t tick_ms = 1000;
    int frame;
    int entered_rail = 0;
    int exited_rail = 0;

    check_status(kart_physics_parameters_default_v2(&parameters, KP_ABI_VERSION),
                 "kart_physics_parameters_default_v2");
    check_status(kart_physics_state_default_v2(&state, KP_ABI_VERSION),
                 "kart_physics_state_default_v2");

    engine = kart_physics_create_v2(&state, KP_NUMERIC_MODE_FLOAT32, KP_ABI_VERSION);
    if (engine == NULL) {
        char message[256];
        kart_physics_last_error_message(message, sizeof(message));
        fprintf(stderr, "smoke: kart_physics_create_v2 failed: %s\n", message);
        exit(1);
    }

    memset(&world, 0, sizeof(world));
    world.query_ground = rail_query_ground;
    world.sweep_obb_rotating_all = empty_sweep;
    world.sweep_point = rail_sweep_point;
    world.query_rail_contact = rail_query_contact;
    world.query_rail_entry_boundary = rail_no_entry_boundary;
    world.rail_query_contexts = rail_query_contexts_cb;
    world.rail_entry_contexts = rail_entry_contexts_cb;

    memset(&inputs, 0, sizeof(inputs));
    inputs.tick_ms = tick_ms;
    check_status(kart_physics_step(engine, &inputs, &parameters, &world),
                 "kart_physics_step(clock_init)");
    for (frame = 0; frame < RAIL_FRAMES; frame++) {
        char prefix[24];
        tick_ms += 16;
        inputs.tick_ms = tick_ms;
        inputs.keys = KP_KEY_UP;
        check_status(kart_physics_step(engine, &inputs, &parameters, &world),
                     "kart_physics_step(rail)");
        check_status(kart_physics_get_state(engine, &state),
                     "kart_physics_get_state(rail)");
        snprintf(prefix, sizeof(prefix), "rail.f%d.", frame + 1);
        print_state_bits(prefix, &state);
        if (state.topology.domain == KP_DOMAIN_RAIL) {
            entered_rail = 1;
        } else if (entered_rail) {
            exited_rail = 1;
        }
    }

    printf("rail.entered_rail=%d\n", entered_rail);
    printf("rail.exited_rail=%d\n", exited_rail);
    if (!entered_rail || !exited_rail) {
        fprintf(stderr,
                "smoke: RAIL scenario did not enter AND exit the RAIL domain "
                "(entered=%d exited=%d)\n",
                entered_rail, exited_rail);
        exit(3);
    }

    kart_physics_destroy(engine);
}

/* ------------------------------------------------------------------------
 * Scenario 3 (ABI v3): CTRL input reaches B08 before the same substep's
 * ground producer.
 * ------------------------------------------------------------------------ */

static void run_drift_scenario(void) {
    static const uint32_t frame_keys[4] = {
        KP_KEY_LEFT | KP_KEY_CTRL,
        KP_KEY_LEFT | KP_KEY_CTRL,
        KP_KEY_LEFT,
        0,
    };
    KpVehicleParameters parameters;
    KpVehicleState state;
    KpWorldVtable world;
    KpInputs inputs;
    KartPhysicsEngine *engine;
    uint32_t tick_ms = 1000;
    int frame;

    check_status(kart_physics_parameters_default_v2(&parameters, KP_ABI_VERSION),
                 "kart_physics_parameters_default_v2(drift)");
    check_status(kart_physics_state_default_v2(&state, KP_ABI_VERSION),
                 "kart_physics_state_default_v2(drift)");
    state.velocity[1] = -20.0;
    parameters.drift_window_duration_734 = 0.006;
    engine = kart_physics_create_v2(&state, KP_NUMERIC_MODE_FLOAT32,
                                    KP_ABI_VERSION);
    if (engine == NULL) {
        char message[256];
        kart_physics_last_error_message(message, sizeof(message));
        fprintf(stderr, "smoke: drift engine creation failed: %s\n", message);
        exit(1);
    }

    memset(&world, 0, sizeof(world));
    world.query_ground = flat_query_ground;
    world.sweep_obb_rotating_all = empty_sweep;
    memset(&inputs, 0, sizeof(inputs));
    inputs.tick_ms = tick_ms;
    check_status(kart_physics_step(engine, &inputs, &parameters, &world),
                 "kart_physics_step(drift clock_init)");
    for (frame = 0; frame < 4; frame++) {
        char prefix[24];
        tick_ms += 2;
        inputs.tick_ms = tick_ms;
        inputs.keys = frame_keys[frame];
        check_status(kart_physics_step(engine, &inputs, &parameters, &world),
                     "kart_physics_step(drift)");
        check_status(kart_physics_get_state(engine, &state),
                     "kart_physics_get_state(drift)");
        snprintf(prefix, sizeof(prefix), "drift.f%d.", frame + 1);
        print_drift_state_bits(prefix, &state);
    }
    kart_physics_destroy(engine);
}

/* ------------------------------------------------------------------------
 * Scenario 4 (ABI v4): ITEM_USE reaches the current-family item6 consumer,
 * with fixed-slot inventory and gauge state crossing the public tail.
 * ------------------------------------------------------------------------ */

static void run_boost_scenario(void) {
    static const uint32_t frame_keys[4] = {
        KP_KEY_UP | KP_KEY_ITEM_USE,
        KP_KEY_UP | KP_KEY_ITEM_USE,
        KP_KEY_UP,
        KP_KEY_UP | KP_KEY_ITEM_USE,
    };
    KpVehicleParameters parameters;
    KpVehicleState state;
    KpWorldVtable world;
    KpInputs inputs;
    KartPhysicsEngine *engine;
    uint32_t tick_ms = 1000;
    int frame;

    check_status(kart_physics_parameters_default_v2(&parameters, KP_ABI_VERSION),
                 "kart_physics_parameters_default_v2(boost)");
    check_status(kart_physics_state_default_v2(&state, KP_ABI_VERSION),
                 "kart_physics_state_default_v2(boost)");
    printf("boost.parameters.item6_boost_duration_ms=%" PRId64 "\n",
           parameters.item6_boost_duration_ms);
    printf("boost.parameters.boost_gauge_raw_760=%016" PRIx64 "\n",
           f64_bits(parameters.boost_gauge_raw_760));

    state.booster_inventory_item_ids[0] = 6;
    state.booster_inventory_quantities[0] = 1;
    state.boost_gauge_committed_b10 = 12.5;
    state.boost_gauge_progress_b20 = 34.5;
    state.boost_gauge_pending_b2c = 56.5;
    state.boost_gauge_history_b38 = 78.5;
    engine = kart_physics_create_v2(&state, KP_NUMERIC_MODE_FLOAT32,
                                    KP_ABI_VERSION);
    if (engine == NULL) {
        char message[256];
        kart_physics_last_error_message(message, sizeof(message));
        fprintf(stderr, "smoke: boost engine creation failed: %s\n", message);
        exit(1);
    }

    check_status(kart_physics_get_state(engine, &state),
                 "kart_physics_get_state(boost initial)");
    print_boost_state_bits("boost.initial.", &state);

    memset(&world, 0, sizeof(world));
    world.query_ground = flat_query_ground;
    world.sweep_obb_rotating_all = empty_sweep;
    memset(&inputs, 0, sizeof(inputs));
    inputs.tick_ms = tick_ms;
    check_status(kart_physics_step(engine, &inputs, &parameters, &world),
                 "kart_physics_step(boost clock_init)");
    for (frame = 0; frame < 4; frame++) {
        char prefix[24];
        tick_ms += 2;
        inputs.tick_ms = tick_ms;
        inputs.keys = frame_keys[frame];
        check_status(kart_physics_step(engine, &inputs, &parameters, &world),
                     "kart_physics_step(boost)");
        check_status(kart_physics_get_state(engine, &state),
                     "kart_physics_get_state(boost)");
        snprintf(prefix, sizeof(prefix), "boost.f%d.", frame + 1);
        print_boost_state_bits(prefix, &state);
    }
    kart_physics_destroy(engine);
}

int main(void) {
    if (kart_physics_abi_version() != KP_ABI_VERSION) {
        fprintf(stderr,
                "smoke: ABI version mismatch: library reports %u, header "
                "expects %u; refusing to create an engine\n",
                (unsigned)kart_physics_abi_version(), (unsigned)KP_ABI_VERSION);
        return 2;
    }
    printf("abi_version=%u\n", (unsigned)kart_physics_abi_version());
    run_flat_scenario();
    run_rail_scenario();
    run_drift_scenario();
    run_boost_scenario();
    return 0;
}
