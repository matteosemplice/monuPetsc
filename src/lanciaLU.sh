#!/bin/bash

./monuCart -Nx 10 \
 -snes_monitor -snes_converged_reason \
 -ksp_type preonly \
 -pc_type lu
"$@"
