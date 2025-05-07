#!/bin/bash

rm -rf hexagon/HTP/RwkvWkvOpPackage/build
make -C hexagon/HTP/RwkvWkvOpPackage/ htp_x86 htp_v68 htp_v69 htp_v73 htp_v75 htp_v79 -j4

make -C hexagon/CPU/RwkvWkvOpPackage/ -j4

rm -rf hexagon/HTP/prebuilt
mkdir -p hexagon/HTP/prebuilt

cp hexagon/HTP/RwkvWkvOpPackage/build/hexagon-v68/libQnnRwkvWkvOpPackage.so hexagon/HTP/prebuilt/libQnnRwkvWkvOpPackageV68.so
cp hexagon/HTP/RwkvWkvOpPackage/build/hexagon-v69/libQnnRwkvWkvOpPackage.so hexagon/HTP/prebuilt/libQnnRwkvWkvOpPackageV69.so
cp hexagon/HTP/RwkvWkvOpPackage/build/hexagon-v73/libQnnRwkvWkvOpPackage.so hexagon/HTP/prebuilt/libQnnRwkvWkvOpPackageV73.so
cp hexagon/HTP/RwkvWkvOpPackage/build/hexagon-v75/libQnnRwkvWkvOpPackage.so hexagon/HTP/prebuilt/libQnnRwkvWkvOpPackageV75.so
cp hexagon/HTP/RwkvWkvOpPackage/build/hexagon-v79/libQnnRwkvWkvOpPackage.so hexagon/HTP/prebuilt/libQnnRwkvWkvOpPackageV79.so