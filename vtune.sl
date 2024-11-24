#!/bin/bash
#SBATCH -p qcpu_exp
#SBATCH -A DD-24-108
#SBATCH -n 1 
#SBATCH --comment "use:vtune=2022.2.0"
#SBATCH -t 0:10:00
#SBATCH --mail-type END
#SBATCH -J AVS-vtune

cd $SLURM_SUBMIT_DIR

ml VTune CMake/3.27.6-GCCcore-13.2.0 intel-compilers/2024.2.0 

export OMP_PROC_BIND=close 
export OMP_PLACES=cores

[ -d build_vtune ] && rm -rf build_vtune
[ -d build_vtune ] || mkdir build_vtune
cd build_vtune

CC=icx CXX=icpx cmake ..
make

for threads in 18 36; do
    for builder in "ref" "loop" "tree"; do
        rm -rf vtune-${builder}-${threads}
        vtune -collect threading -r vtune-${builder}-${threads} -app-working-dir . -- ./PMC --builder ${builder} -t ${threads} --grid 128 ../data/bun_zipper_res3.pts
    done
done
