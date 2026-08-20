cd lib/PoseLib/
mkdir -p _build && cd _build
cmake -DCMAKE_INSTALL_PREFIX=../_install ..
cmake --build . --target install -j 8
cd ../../..

cd lib/eccv2026/
mkdir -p _build && cd _build
cmake -DCMAKE_PREFIX_PATH="$(pwd)/../../PoseLib/_install" ..
make -j100
cd ../../..
