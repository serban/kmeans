#!/bin/bash
# vim:set ts=8 sw=4 sts=4 et:

# Copyright (c) 2011 Serban Giuroiu
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

# ------------------------------------------------------------------------------

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
