#!/bin/bash

rm -rf hexagon/HTP/RwkvWkvOpPackage/build
make -C hexagon/HTP/RwkvWkvOpPackage/ htp_x86 htp_v75 -j4

make -C hexagon/CPU/RwkvWkvOpPackage/ -j4