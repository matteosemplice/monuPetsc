#!/bin/bash
mpirun -np 1 ./monuCart -Nx 10 \
 -snes_monitor -snes_converged_reason \
 -ksp_type fgmres \
 -pc_type fieldsplit \
 -fieldsplit_c_pc_type jacobi \
 -ksp_monitor \
 "$@"
