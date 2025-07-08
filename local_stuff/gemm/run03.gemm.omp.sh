
PREV_DIR=$(readlink -f .)
#BUILD_DIR="${PREV_DIR}/../../cmake-build-debug"
BUILD_DIR="${PREV_DIR}/../../build"

OUTPUT_DIR="output.$(date +%FT%T)"
mkdir -p ${OUTPUT_DIR}

cd ${OUTPUT_DIR}

dram_gemm_app="${BUILD_DIR}/gemm_dram.omp"
rapid_gemm_app="${BUILD_DIR}/gemm_rapid.omp"

# Run
${dram_gemm_app}
${rapid_gemm_app}

