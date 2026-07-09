// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// Concepts 1996-2026 Miroslav Sotek. All rights reserved.
// Code 2020-2026 Miroslav Sotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Control - Native fixed-boundary SOR solver.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <new>
#include <vector>

namespace {

struct ScpnSolver {
    int nr;
    int nz;
    double r_min;
    double r_max;
    double z_min;
    double z_max;
    double dr;
    double dz;
    double boundary;
    std::vector<double> psi;
};

bool valid_dimensions(int nr, int nz, double r_min, double r_max, double z_min, double z_max) {
    return nr >= 3 && nz >= 3 && std::isfinite(r_min) && std::isfinite(r_max) &&
           std::isfinite(z_min) && std::isfinite(z_max) && r_max > r_min && z_max > z_min;
}

std::size_t offset(const ScpnSolver& solver, int iz, int ir) {
    return static_cast<std::size_t>(iz) * static_cast<std::size_t>(solver.nr) +
           static_cast<std::size_t>(ir);
}

double radius_at(const ScpnSolver& solver, int ir) {
    return solver.r_min + solver.dr * static_cast<double>(ir);
}

void apply_boundary(ScpnSolver& solver) {
    for (int ir = 0; ir < solver.nr; ++ir) {
        solver.psi[offset(solver, 0, ir)] = solver.boundary;
        solver.psi[offset(solver, solver.nz - 1, ir)] = solver.boundary;
    }
    for (int iz = 0; iz < solver.nz; ++iz) {
        solver.psi[offset(solver, iz, 0)] = solver.boundary;
        solver.psi[offset(solver, iz, solver.nr - 1)] = solver.boundary;
    }
}

bool finite_input(const double* values, int size) {
    if (values == nullptr || size <= 0) {
        return false;
    }
    for (int index = 0; index < size; ++index) {
        if (!std::isfinite(values[index])) {
            return false;
        }
    }
    return true;
}

double update_point(
    ScpnSolver& solver,
    const double* source,
    int iz,
    int ir,
    double omega
) {
    const double r = std::max(radius_at(solver, ir), std::numeric_limits<double>::epsilon());
    const double dr_sq = solver.dr * solver.dr;
    const double dz_sq = solver.dz * solver.dz;
    const double c_r_plus = 1.0 / dr_sq - 1.0 / (2.0 * r * solver.dr);
    const double c_r_minus = 1.0 / dr_sq + 1.0 / (2.0 * r * solver.dr);
    const double c_z = 1.0 / dz_sq;
    const double center = 2.0 / dr_sq + 2.0 / dz_sq;
    const std::size_t idx = offset(solver, iz, ir);
    const double old_value = solver.psi[idx];

    const double prediction =
        (source[idx] +
         c_z * (solver.psi[offset(solver, iz + 1, ir)] + solver.psi[offset(solver, iz - 1, ir)]) +
         c_r_plus * solver.psi[offset(solver, iz, ir + 1)] +
         c_r_minus * solver.psi[offset(solver, iz, ir - 1)]) /
        center;
    solver.psi[idx] = (1.0 - omega) * old_value + omega * prediction;
    return std::abs(solver.psi[idx] - old_value);
}

double sor_sweep(ScpnSolver& solver, const double* source, double omega) {
    double max_delta = 0.0;
    for (int parity = 0; parity < 2; ++parity) {
        for (int iz = 1; iz < solver.nz - 1; ++iz) {
            for (int ir = 1; ir < solver.nr - 1; ++ir) {
                if (((iz + ir) & 1) == parity) {
                    max_delta = std::max(max_delta, update_point(solver, source, iz, ir, omega));
                }
            }
        }
    }
    apply_boundary(solver);
    return max_delta;
}

int safe_iterations(int iterations) {
    return std::max(0, iterations);
}

void copy_solution(const ScpnSolver& solver, double* psi_out, int size) {
    if (psi_out == nullptr || size != solver.nr * solver.nz) {
        return;
    }
    std::copy(solver.psi.begin(), solver.psi.end(), psi_out);
}

}  // namespace

extern "C" void* create_solver(
    int nr,
    int nz,
    double r_min,
    double r_max,
    double z_min,
    double z_max
) {
    if (!valid_dimensions(nr, nz, r_min, r_max, z_min, z_max)) {
        return nullptr;
    }
    try {
        auto* solver = new ScpnSolver{
            nr,
            nz,
            r_min,
            r_max,
            z_min,
            z_max,
            (r_max - r_min) / static_cast<double>(nr - 1),
            (z_max - z_min) / static_cast<double>(nz - 1),
            0.0,
            std::vector<double>(static_cast<std::size_t>(nr * nz), 0.0),
        };
        apply_boundary(*solver);
        return solver;
    } catch (const std::bad_alloc&) {
        return nullptr;
    }
}

extern "C" void set_boundary_dirichlet(void* solver_ptr, double boundary_value) {
    auto* solver = static_cast<ScpnSolver*>(solver_ptr);
    if (solver == nullptr || !std::isfinite(boundary_value)) {
        return;
    }
    solver->boundary = boundary_value;
    apply_boundary(*solver);
}

extern "C" void run_step(void* solver_ptr, double* source, double* psi_out, int size, int iterations) {
    auto* solver = static_cast<ScpnSolver*>(solver_ptr);
    if (solver == nullptr || !finite_input(source, size) || size != solver->nr * solver->nz) {
        return;
    }
    for (int step = 0; step < safe_iterations(iterations); ++step) {
        sor_sweep(*solver, source, 1.5);
    }
    copy_solution(*solver, psi_out, size);
}

extern "C" int run_step_converged(
    void* solver_ptr,
    double* source,
    double* psi_out,
    int size,
    int max_iterations,
    double omega,
    double tolerance,
    double* final_delta
) {
    auto* solver = static_cast<ScpnSolver*>(solver_ptr);
    if (final_delta != nullptr) {
        *final_delta = std::numeric_limits<double>::quiet_NaN();
    }
    if (solver == nullptr || !finite_input(source, size) || size != solver->nr * solver->nz ||
        !std::isfinite(omega) || omega <= 0.0 || omega >= 2.0 || !std::isfinite(tolerance) ||
        tolerance < 0.0) {
        return 0;
    }

    int used = 0;
    double delta = std::numeric_limits<double>::infinity();
    for (; used < safe_iterations(max_iterations); ++used) {
        delta = sor_sweep(*solver, source, omega);
        if (delta <= tolerance) {
            ++used;
            break;
        }
    }
    if (final_delta != nullptr) {
        *final_delta = delta;
    }
    copy_solution(*solver, psi_out, size);
    return used;
}

extern "C" void destroy_solver(void* solver_ptr) {
    delete static_cast<ScpnSolver*>(solver_ptr);
}
