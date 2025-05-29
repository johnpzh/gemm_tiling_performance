
PREV_DIR=$(readlink -f .)
BUILD_DIR="${PREV_DIR}/../../cmake-build-debug"

OUTPUT_DIR="output.$(date +%FT%T)"
mkdir -p ${OUTPUT_DIR}

cd ${OUTPUT_DIR}

dram_gemm_app="${BUILD_DIR}/gemm_dram"

${dram_gemm_app}
