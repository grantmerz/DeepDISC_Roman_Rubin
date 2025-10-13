#!/bin/bash
source /opt/lsst/software/stack/loadLSST.bash
setup lsst_distrib
which python

python /pscratch/sd/y/yaswante/MyQuota/roman_lsst/get_multipatch_tiles.py