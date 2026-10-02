"""Post-hoc finite-box algebra check; no model calls or replacement grading."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .campaign_client import atomic_json


def spectrum(size, single_energy):
    axis = 1 - np.cos(2 * np.pi * np.arange(size) / size)
    hopping = single_energy / (2 * axis[1])
    free = (4 * hopping * (axis[:, None, None] + axis[None, :, None] + axis[None, None, :])).ravel()
    bound = -17.8
    inverse_g = np.sum(1 / (bound - free))
    g = 1 / inverse_g
    lower, upper = 1e-8, 2 * single_energy - 1e-8
    for _ in range(70):
        midpoint = (lower + upper) / 2
        value = np.sum(1 / (midpoint - free)) - inverse_g
        if value > 0:
            lower = midpoint
        else:
            upper = midpoint
    first = (lower + upper) / 2
    result = {'L': size, 't_MeV': float(hopping), 'C': float(g * size**3 / hopping),
              'first_excited_MeV': float(first), 'first_excited_over_single': float(first / single_energy),
              'first_free_shell_MeV': 2 * single_energy,
              'secular_residual': float(np.sum(1 / (first - free)) - inverse_g)}
    if size == 4:
        values = np.linalg.eigvalsh(np.diag(free) + g * np.ones((size**3, size**3)))
        assert abs(values[0] - bound) < 1e-9
        assert abs(values[1] - first) < 1e-9
        assert np.count_nonzero(np.isclose(values, 2 * single_energy, atol=1e-8)) == 5
        result['dense_eigenvalues_first_eight'] = values[:8].tolist()
    return result


def check():
    # Physical box stays fixed as lattice sites increase: this is NOT infinite volume.
    single = (197.3269804 * 2 * np.pi / 3.4)**2 / (2 * 1634)
    values = [spectrum(size, single) for size in (4, 12, 24, 48, 96)]
    assert all(abs(row['secular_residual']) < 1e-10 for row in values)
    # Six opposite-spin momentum states in the first free shell. A contact
    # interaction has rank one there. Even spatial parity reduces to three
    # combinations; two even and three odd combinations remain unaffected.
    contact = np.ones((6, 6))
    even = np.zeros((6, 3))
    for index in range(3):
        even[2 * index:2 * index + 2, index] = 1 / np.sqrt(2)
    assert np.linalg.matrix_rank(contact) == 1
    assert np.linalg.matrix_rank(even.T @ contact @ even) == 1
    return {'recorded_at': time.time(), 'selection': 'Post hoc, after answers and rubric were inspected. Not blinded, not a replacement grade.',
            'scope': 'Zero-total-momentum opposite-spin contact Hamiltonian at fixed physical box. No additional spin/parity projection in the full spectrum calculation.',
            'single_particle_energy_MeV': float(single), 'binding_energy_MeV': -17.8,
            'physical_box_fm': 3.4, 'nucleon_mass_MeV': 1634,
            'matrix': 'diag(4*t*sum_alpha(1-cos(2*pi*n_alpha/L)))) + (C*t/L^3)*ones',
            'coupling_condition': '1/g = sum_n 1/(-17.8-D_n), g=C*t/L^3',
            'finite_lattice_results': values,
            'sector_caveat': 'Five unaffected states refer to the full six-state opposite-spin shell. Restricting to even spatial parity leaves two unaffected states; three full-shell states have odd parity. Sector specification is needed before assigning a strict spin-singlet degeneracy.',
            'interpretation': 'Discrete levels survive L→infinity at fixed physical box. An infinite-volume interpretation changes the limit. This check does not settle all task wording or rubric issues.',
            'primary_source': 'https://doi.org/10.1103/PhysRevC.110.024002',
            'source_pdf': 'https://par.nsf.gov/servlets/purl/10596563',
            'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


if __name__ == '__main__':
    result = check()
    atomic_json(Path('runs/k10-ablation-3h-20261001/physics-check.json'), result)
    print(json.dumps(result, indent=2))
