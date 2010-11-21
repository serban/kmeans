#  * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#    File:         Makefile                                                  */
#    Description:  Makefile for programs running a simple k-means clustering */
#                  algorithm                                                 */
#                                                                            */
#    Author:  Wei-keng Liao                                                  */
#             ECE Department Northwestern University                         */
#             email: wkliao@ece.northwestern.edu                             */
#    Copyright, 2005, Wei-keng Liao                                          */
#                                                                            */
#  * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

.KEEP_STATE:

all: seq omp mpi

DFLAGS      =
OPTFLAGS    = -O -NDEBUG
OPTFLAGS    = -g
INCFLAGS    = -I.
CFLAGS      = $(OPTFLAGS) $(DFLAGS) $(INCFLAGS)
LDFLAGS     = $(OPTFLAGS)
LIBS        =

# please check the compile to the one you use and the openmp flag
# Here, I am using gcc and its openmp compile flag is -fopenmp
# If icc is used, please us -opnemp
#
OMPFLAGS    = -fopenmp

CC          = gcc
MPICC       = mpicc

.c.o:
	$(CC) $(CFLAGS) -c $<

H_FILES     = kmeans.h

#------   OpenMP version -----------------------------------------
OMP_SRC     = omp_main.c \
	      omp_kmeans.c

OMP_OBJ     = $(OMP_SRC:%.c=%.o)

omp_kmeans.o: omp_kmeans.c $(H_FILES)
	$(CC) $(CFLAGS) $(OMPFLAGS) -c omp_kmeans.c

omp: omp_main
omp_main: $(OMP_OBJ) file_io.o
	$(CC) $(LDFLAGS) $(OMPFLAGS) -o omp_main $(OMP_OBJ) file_io.o $(LIBS)

#------   MPI version -----------------------------------------
MPI_SRC     = mpi_main.c   \
              mpi_kmeans.c \
              mpi_io.c     \
	      file_io.c

MPI_OBJ     = $(MPI_SRC:%.c=%.o)

mpi_main.o: mpi_main.c $(H_FILES)
	$(MPICC) $(CFLAGS) -c $*.c

mpi_kmeans.o: mpi_kmeans.c $(H_FILES)
	$(MPICC) $(CFLAGS) -c $*.c

mpi_io.o: mpi_io.c $(H_FILES)
	$(MPICC) $(CFLAGS) -c $*.c

mpi: mpi_main
mpi_main: $(MPI_OBJ) $(H_FILES)
	$(MPICC) $(LDFLAGS) -o mpi_main $(MPI_OBJ) $(LIBS)

#------   sequential version -----------------------------------------
SEQ_SRC     = seq_main.c   \
              seq_kmeans.c \
	      file_io.c    \
	      wtime.c

SEQ_OBJ     = $(SEQ_SRC:%.c=%.o)

$(SEQ_OBJ): $(H_FILES)

seq: seq_main
seq_main: $(SEQ_OBJ) $(H_FILES)
	$(CC) $(LDFLAGS) -o seq_main $(SEQ_OBJ) $(LIBS)

#---------------------------------------------------------------------
clean:
	rm -rf *.o omp_main seq_main mpi_main \
	       core* .make.state              \
               *.cluster_centres *.membership \
               Image_data/*.cluster_centres   \
               Image_data/*.membership
