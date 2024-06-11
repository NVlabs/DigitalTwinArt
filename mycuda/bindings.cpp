#include <torch/extension.h>
#include "common.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sampleRaysUniformOccupiedVoxels", &sampleRaysUniformOccupiedVoxels);
    m.def("postprocessOctreeRayTracing", &postprocessOctreeRayTracing);
}