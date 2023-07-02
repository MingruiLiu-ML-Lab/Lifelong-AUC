#!/bin/bash

MY_PYTHON="python"

cd data/
cd raw/

$MY_PYTHON raw.py

$MY_PYTHON isic_processing.py
#$MY_PYTHON eurosat_processing.py

cd ..
$MY_PYTHON isic_rotations.py
#$MY_PYTHON eurosat_rotations.py
cd ..
