for Lval in 140 145 150 155 160 165 170 175 180 185 190 195 200; do
    docker run --rm \
        --platform linux/amd64 \
        -v "$(pwd)/lammps":/data \
        -w /data \
        -e OMP_NUM_THREADS=4 \
        lammps/lammps:latest \
        lmp_mpi -in lmp.input -var Lval $Lval
done