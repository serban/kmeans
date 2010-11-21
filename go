# sequential K-means -------------------------------------------------------
seq_main -o -b -n 4 -i Image_data/color17695.bin
seq_main -o -b -n 4 -i Image_data/edge17695.bin
seq_main -o -b -n 4 -i Image_data/texture17695.bin

seq_main -o    -n 4 -i Image_data/color100.txt
seq_main -o    -n 4 -i Image_data/edge100.txt
seq_main -o    -n 4 -i Image_data/texture100.txt

# OpenMP K-means using pragma atomic -----------------------------------------
omp_main -a -o -b -n 4 -i Image_data/color17695.bin
omp_main -a -o -b -n 4 -i Image_data/edge17695.bin
omp_main -a -o -b -n 4 -i Image_data/texture17695.bin

omp_main -a -o    -n 4 -i Image_data/color100.txt
omp_main -a -o    -n 4 -i Image_data/edge100.txt
omp_main -a -o    -n 4 -i Image_data/texture100.txt

# MPI K-means -------------------------------------------------------
mpiexec -n 4 mpi_main -o -b -n 4 -i Image_data/color17695.bin
mpiexec -n 4 mpi_main -o -b -n 4 -i Image_data/edge17695.bin
mpiexec -n 4 mpi_main -o -b -n 4 -i Image_data/texture17695.bin

mpiexec -n 4 mpi_main -o    -n 4 -i Image_data/color100.txt
mpiexec -n 4 mpi_main -o    -n 4 -i Image_data/edge100.txt
mpiexec -n 4 mpi_main -o    -n 4 -i Image_data/texture100.txt

