#!/bin/bash
#SBATCH -p qcpu_exp
#SBATCH -A DD-24-108
#SBATCH -n 1 
#SBATCH -t 0:45:00
#SBATCH --mail-type END
#SBATCH -J AVS-evaluate

cd $SLURM_SUBMIT_DIR

ml purge 
ml matplotlib/3.5.2-foss-2022a
ml CMake/3.27.6-GCCcore-13.2.0 intel-compilers/2024.2.0 


export OMP_PROC_BIND=close 
export OMP_PLACES=cores


[ -d build_evaluate ] && rm -rf build_evaluate
[ -d build_evaluate ] || mkdir build_evaluate

cd build_evaluate
rm tmp_*

CC=icx CXX=icpx cmake ..
make

#ml matplotlib/3.5.2-foss-2022a
bash ../scripts/generate_data_and_plots.sh
