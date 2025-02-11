FROM python:3.9-slim

# Install system dependencies first
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    git \
    openmpi-bin \
    libopenmpi-dev \
    && rm -rf /var/lib/apt/lists/*

# Now install Python packages
RUN pip install --upgrade pip
RUN pip install git+https://gitlab.com/ase/ase

RUN pip install --no-cache-dir \
    imolecule \
    spglib \
    torch \
    torchani \
    nglview \
    matplotlib \
    scipy \
    dscribe \
    statsmodels

# Fix ENV syntax as per warnings
ENV LAMMPS_PATH=/opt/lammps
ENV PATH=${LAMMPS_PATH}/src/:${PATH}

# Build LAMMPS from github
RUN mkdir /opt/lammps && \
    cd /opt/lammps && \
    git clone --branch stable --depth 1 https://github.com/lammps/lammps.git .

# Build LAMMPS
RUN cd /opt/lammps/src \
    && make yes-all \
    && make no-lib \
    && make no-intel \
    && make yes-python \
    && make -j4 mpi mode=shlib \
    && ln -s Obj_shared_mpi Obj_mpi \
    && make -j4 mpi \
    && make clean-all