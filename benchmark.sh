#!/bin/bash
# vim:set ts=8 sw=4 sts=4 et:

set -e

input='color17695.bin'
#input='texture17695.bin'

mkdir -p profiles

make seq omp cuda

echo "--------------------------------------------------------------------------------"
uptime
echo "--------------------------------------------------------------------------------"

# TODO: Add quotes around ${input} so that spaces in the filename don't break things

for k in 2 4 8 16 32 64 128; do
    seqTime=$(./seq_main -o -n $k -b -i Image_data/${input} | grep 'Computation' | awk '{print $4}')
    gprof ./seq_main > profiles/seq-profile-$k.txt
    mv Image_data/${input}.cluster_centres Image_data/${input}-$k.cluster_centres
    mv Image_data/${input}.membership Image_data/${input}-$k.membership

    ompTime=$(./omp_main -o -n $k -b -i Image_data/${input} | grep 'Computation' | awk '{print $4}')
    gprof ./omp_main > profiles/omp-profile-$k.txt

    cudaTime=$(./cuda_main -o -n $k -b -i Image_data/${input} | grep 'Computation' | awk '{print $4}')
    gprof ./cuda_main > profiles/cuda-profile-$k.txt
    diff -q Image_data/${input}-$k.cluster_centres Image_data/${input}.cluster_centres
    diff -q Image_data/${input}-$k.membership Image_data/${input}.membership

    speedup=$(echo "scale=1; ${seqTime} / ${cudaTime}" | bc)
    echo "k = $(printf "%3d" $k)  seqTime = ${seqTime}s  ompTime = ${ompTime}s  cudaTime = ${cudaTime}s  speedup = ${speedup}x"
done
