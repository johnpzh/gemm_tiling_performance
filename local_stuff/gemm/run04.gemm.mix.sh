
PREV_DIR=$(readlink -f .)
#BUILD_DIR="${PREV_DIR}/../../cmake-build-debug"
BUILD_DIR="${PREV_DIR}/../../build"

OUTPUT_DIR="output.$(date +%FT%T)"
mkdir -p ${OUTPUT_DIR}

cd ${OUTPUT_DIR}

# Run

# FAM, Tiling, TileSize, Seq
#"${BUILD_DIR}/gemm_rapid.tile_size.seq"

# FAM, Tiling, TileSize, NumThreads
# Matrix-size-4096
# Matrix-size-8192
#"${BUILD_DIR}/gemm_rapid.tile_size.omp"

# DRAM, Tiling, TileSize, NumThreads
# Matrix-size-4096
# Matrix-size-8192
#"${BUILD_DIR}/gemm_dram.tile_size.omp"

# Extra
# FAM, DB, BufferSize, 1Thread
/home/peng599/pppp/amais_project/soft_cache_gemm/build/gemm_rapid.buffer_size.seq


