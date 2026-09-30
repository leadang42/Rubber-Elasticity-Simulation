#!/usr/bin/env bash
# Run the LAMMPS stretching simulation (lammps/lmp.input) for a series of
# end-to-end lengths L in Å, using the official LAMMPS Docker image.
#
# Usage: ./run_lammps.sh [L ...]    (default: 140 145 ... 200)
#
# Writes lmp<L>.log, lmp<L>.forces and lmp<L>.xyz to lammps/.
# Environment: LAMMPS_IMAGE (default lammps/lammps:latest), OMP_NUM_THREADS (default 4).
set -euo pipefail

LAMMPS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/lammps" && pwd)"

if [ "$#" -gt 0 ]; then
    lengths=("$@")
else
    lengths=(140 145 150 155 160 165 170 175 180 185 190 195 200)
fi

for Lval in "${lengths[@]}"; do
    echo "==> L = ${Lval} Å"
    docker run --rm \
        --platform linux/amd64 \
        -v "${LAMMPS_DIR}":/data \
        -w /data \
        -e OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" \
        "${LAMMPS_IMAGE:-lammps/lammps:latest}" \
        lmp_mpi -in lmp.input -var Lval "${Lval}"
done
