# set -euxo pipefail
set -u

# Perf Events
# EVENTS="\
# cache-references,\
# cache-misses,\
# L1-dcache-loads,\
# L1-dcache-load-misses,\
# LLC-loads,\
# LLC-load-misses,\
# LLC-stores,\
# LLC-store-misses,\
# l2_request.all,\
# l2_request.miss,\
# l2_rqsts.all_demand_data_rd,\
# l2_rqsts.demand_data_rd_miss,\
# l2_rqsts.demand_data_rd_hit,\
# l2_rqsts.all_hwpf,\
# l2_rqsts.hwpf_miss,\
# l2_rqsts.swpf_hit,\
# l2_rqsts.swpf_miss\
# "
EVENTS="\
L1-dcache-loads,\
L1-dcache-load-misses,\
l2_request.all,\
l2_request.miss,\
l2_rqsts.all_demand_data_rd,\
l2_rqsts.demand_data_rd_miss,\
LLC-loads,\
LLC-load-misses,\
l2_rqsts.all_hwpf,\
l2_rqsts.hwpf_miss,\
l2_rqsts.swpf_hit,\
l2_rqsts.swpf_miss,\
cycles,\
instructions,\
mem_inst_retired.all_loads\
"

METRICS="\
tma_l1_bound,\
tma_l2_bound,\
tma_l3_bound,\
tma_dram_bound\
"

#
# Test if this machine is running macOS
if [ "$(uname -s)" = "Linux" ]; then
  export EXT="so"
  ulimit -s unlimited  # Set stack size as unlimited
elif [ "$(uname -s)" = "Darwin" ]; then
  # macOS
  export EXT="dylib"
else
  echo "$(uname -s) is unknown."
  exit -1
fi

rapid_app="../../build/gemm_rapid.tile_size.omp.v2.command_option"
dram_app="../../build/gemm_dram.tile_size.omp.v2.command_option"

RAPID_BASE_NAME="gemm.perf.rapid"
DRAM_BASE_NAME="gemm.perf.dram"
output_dir="results.perf_events.$(date +%FT%T)"
mkdir -p "${output_dir}"

TT_TIME_START=$(date +%s.%N)

# FAM
for ((matrix_size = 4096; matrix_size <= 8192; matrix_size *= 2)); do
  for ((tile_size = 8; tile_size <= 2048; tile_size *= 2)); do
    for ((num_threads = 1; num_threads <= 16; num_threads *= 2)); do
#for ((matrix_size = 256; matrix_size <= 512; matrix_size *= 2)); do
#  for ((tile_size = 16; tile_size <= 32; tile_size *= 2)); do
#    for ((num_threads = 8; num_threads <= 16; num_threads *= 2)); do
      rapid_out_file="${output_dir}/${RAPID_BASE_NAME}.matrix-${matrix_size}.tile-${tile_size}.thread-${num_threads}.log"
      COMMAND="${rapid_app} ${matrix_size} ${tile_size} ${num_threads}"
      set -x
      perf stat -e "${EVENTS}" -M "${METRICS}" ${COMMAND} 2>&1 | tee -a ${rapid_out_file}
      set +x
    done

    # one more
    num_threads=28
    rapid_out_file="${output_dir}/${RAPID_BASE_NAME}.matrix-${matrix_size}.tile-${tile_size}.thread-${num_threads}.log"
    COMMAND="${rapid_app} ${matrix_size} ${tile_size} ${num_threads}"
    set -x
    perf stat -e "${EVENTS}" -M "${METRICS}" ${COMMAND} 2>&1 | tee -a ${rapid_out_file}
    set +x
  done
done


# DRAM
for ((matrix_size = 4096; matrix_size <= 8192; matrix_size *= 2)); do
  for ((tile_size = 8; tile_size <= 2048; tile_size *= 2)); do
    for ((num_threads = 1; num_threads <= 16; num_threads *= 2)); do
#for ((matrix_size = 256; matrix_size <= 512; matrix_size *= 2)); do
#  for ((tile_size = 16; tile_size <= 32; tile_size *= 2)); do
#    for ((num_threads = 8; num_threads <= 16; num_threads *= 2)); do
      dram_out_file="${output_dir}/${DRAM_BASE_NAME}.matrix-${matrix_size}.tile-${tile_size}.thread-${num_threads}.log"
      COMMAND="${dram_app} ${matrix_size} ${tile_size} ${num_threads}"
      set -x
      perf stat -e "${EVENTS}" -M "${METRICS}" ${COMMAND} 2>&1 | tee -a ${dram_out_file}
      set +x
    done

    # one more
    num_threads=28
    dram_out_file="${output_dir}/${DRAM_BASE_NAME}.matrix-${matrix_size}.tile-${tile_size}.thread-${num_threads}.log"
    COMMAND="${dram_app} ${matrix_size} ${tile_size} ${num_threads}"
    set -x
    perf stat -e "${EVENTS}" -M "${METRICS}" ${COMMAND} 2>&1 | tee -a ${dram_out_file}
    set +x
  done
done


TT_TIME_END=$(date +%s.%N)
TT_TIME_EXE=$(echo "${TT_TIME_END} - ${TT_TIME_START}" | bc -l)
echo
echo "TT_TIME_EXE(s): ${TT_TIME_EXE}"
echo

