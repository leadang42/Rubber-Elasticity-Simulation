for Lval in 140 150 160 170 180 190 200; do
    docker run --rm \
        --platform linux/amd64 \
        -v "$(pwd)/lammps":/data \
        -w /data \
        -e OMP_NUM_THREADS=4 \
        lammps/lammps:latest \
        lmp_mpi -in lmp.input -var Lval $Lval
done