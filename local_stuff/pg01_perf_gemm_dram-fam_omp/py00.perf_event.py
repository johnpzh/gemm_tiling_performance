import sys
import argparse
import time
import pprint
import pandas as pd

# EVENTS=[
#     "cache-references",
#     "cache-misses",
#     "L1-dcache-loads",
#     "L1-dcache-load-misses",
#     "LLC-loads",
#     "LLC-load-misses",
#     "LLC-stores",
#     "LLC-store-misses",
#     "l2_request.all",
#     "l2_request.miss",
#     "l2_rqsts.all_demand_data_rd",
#     "l2_rqsts.demand_data_rd_miss",
#     "l2_rqsts.demand_data_rd_hit",
#     "l2_rqsts.all_hwpf",
#     "l2_rqsts.hwpf_miss",
#     "l2_rqsts.swpf_hit",
#     "l2_rqsts.swpf_miss"
# ]
EVENTS=[
    # "cache-references",
    # "cache-misses",
    "L1-dcache-loads",
    "L1-dcache-load-misses",
    "l2_request.all",
    "l2_request.miss",
    "l2_rqsts.all_demand_data_rd",
    "l2_rqsts.demand_data_rd_miss",
    # "l2_rqsts.demand_data_rd_hit",
    "LLC-loads",
    "LLC-load-misses",
    # "LLC-stores",
    # "LLC-store-misses",
    "l2_rqsts.all_hwpf",
    "l2_rqsts.hwpf_miss",
    "l2_rqsts.swpf_hit",
    "l2_rqsts.swpf_miss",
    "cycles",
    "instructions",
    "mem_inst_retired.all_loads"
]

TMA_MEMORY_BOUND_GROUP=[
    "tma_l1_bound",
    "tma_l2_bound",
    "tma_l3_bound",
    "tma_dram_bound"
]

def one_file(filename):
    print(f"\nfilename: {filename}")
    table = {}
    collected_events = set()
    collected_tma_memory_bound_group = set()
    with open(filename) as fin:
        for line in fin:
            for event in EVENTS:
                if event in line:
                    number = int(line.split()[0].replace(",", ""))
                    table[event] = number
                    collected_events.add(event)
            for mem_bound in TMA_MEMORY_BOUND_GROUP:
                if mem_bound in line:
                    number = float(line.split()[-4])
                    table[mem_bound] = number / 100.0 # perf outputs percentage
                    collected_tma_memory_bound_group.add(mem_bound)

        for event in EVENTS:
            if event not in collected_events:
                # test
                print(f"!!! event {event} not collected")
                # end test
                table[event] = 0.0
        for mem_bound in TMA_MEMORY_BOUND_GROUP:
            if mem_bound not in collected_tma_memory_bound_group:
                print(f"!!! memory bound {mem_bound} not collected")
                table[mem_bound] = 0.0

    return table


if __name__ == "__main__":

    TT_TIME_START = time.perf_counter()
    # parser = argparse.ArgumentParser(f"{sys.argv[0]}")
    # parser.add_argument("input_file", type=str, help="perf stat output file")
    #
    # if len(sys.argv) == 1:
    #     parser.print_help(sys.stderr)
    #     sys.exit(-1)
    # args = parser.parse_args()
    RAPID_BASE_NAME="gemm.perf.rapid"
    DRAM_BASE_NAME="gemm.perf.dram"

    size_list = []

    # dram_l1_dcache_miss_rate_list = []
    # dram_LLC_load_miss_rate_list = []
    # dram_l2_request_miss_rate_list = []
    # dram_l2_data_rd_miss_rate_list = []
    # dram_l2_hwpf_miss_rate_list = []
    # dram_l2_swpf_miss_rate_list = []
    # dram_instr_per_cycle_list = []
    # dram_load_per_instr_list = []
    # dram_tma_dram_bound_list = []
    # dram_tma_l1_bound_list = []
    # dram_tma_l2_bound_list = []
    # dram_tma_l3_bound_list = []
    #
    # rapid_l1_dcache_miss_rate_list = []
    # rapid_LLC_load_miss_rate_list = []
    # rapid_l2_request_miss_rate_list = []
    # rapid_l2_data_rd_miss_rate_list = []
    # rapid_l2_hwpf_miss_rate_list = []
    # rapid_l2_swpf_miss_rate_list = []
    # rapid_instr_per_cycle_list = []
    # rapid_load_per_instr_list = []
    # rapid_tma_dram_bound_list = []
    # rapid_tma_l1_bound_list = []
    # rapid_tma_l2_bound_list = []
    # rapid_tma_l3_bound_list = []

    matrix_size_list = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384] # 4096, 8192
    # matrix_size_list = [32, 64, 128] # 4096, 8192
    tile_size_list = [16384] # 8 - 2048
    num_threads_list = [28] # 1 - 16

    # for _ in range(4, 15):

    dram_results = {}
    fam_results = {}
    for matrix_size in matrix_size_list:
        dram_results[matrix_size] = {}
        fam_results[matrix_size] = {}
        for tile_size in tile_size_list:
            dram_results[matrix_size][tile_size] = {}
            fam_results[matrix_size][tile_size] = {}
            for num_threads in num_threads_list:
                dram_results[matrix_size][tile_size][num_threads] = {}
                fam_results[matrix_size][tile_size][num_threads] = {}
                dram_pool = dram_results[matrix_size][tile_size][num_threads]
                fam_pool = fam_results[matrix_size][tile_size][num_threads]
                rapid_log_file=f"{RAPID_BASE_NAME}.matrix-{matrix_size}.tile-{tile_size}.thread-{num_threads}.log"
                dram_log_file=f"{DRAM_BASE_NAME}.matrix-{matrix_size}.tile-{tile_size}.thread-{num_threads}.log"
                rapid_table=one_file(rapid_log_file)
                dram_table=one_file(dram_log_file)

                # # test
                # print("rapid_table:")
                # pprint.pprint(rapid_table)
                # print("dram_table:")
                # pprint.pprint(dram_table)
                # # end test

                # DRAM
                dram_pool["l1_dcache_miss_rate"] = (dram_table["L1-dcache-load-misses"] /
                                                    dram_table["L1-dcache-loads"])
                dram_pool["l2_request_miss_rate"] = (dram_table["l2_request.miss"] /
                                                     dram_table["l2_request.all"])
                dram_pool["l2_data_rd_miss_rate"] = (dram_table["l2_rqsts.demand_data_rd_miss"] /
                                                     dram_table["l2_rqsts.all_demand_data_rd"])
                dram_pool["LLC_load_miss_rate"] = (dram_table["LLC-load-misses"] /
                                                   dram_table["LLC-loads"])
                dram_pool["l2_hwpf_miss_rate"] = (dram_table["l2_rqsts.hwpf_miss"] /
                                                  dram_table["l2_rqsts.all_hwpf"])
                dram_pool["l2_swpf_miss_rate"] = (dram_table["l2_rqsts.swpf_miss"] /
                                                  (dram_table["l2_rqsts.swpf_miss"] + dram_table["l2_rqsts.swpf_hit"]))
                dram_pool["instr_per_cycle"] = (dram_table["instructions"] /
                                                dram_table["cycles"])
                dram_pool["load_per_instr"] = (dram_table["mem_inst_retired.all_loads"] /
                                               dram_table["instructions"])
                dram_pool["tma_l1_bound"] = dram_table["tma_l1_bound"]
                dram_pool["tma_l2_bound"] = dram_table["tma_l2_bound"]
                dram_pool["tma_l3_bound"] = dram_table["tma_l3_bound"]
                dram_pool["tma_dram_bound"] = dram_table["tma_dram_bound"]

                # FAM
                fam_pool["l1_dcache_miss_rate"] = (rapid_table["L1-dcache-load-misses"] /
                                                    rapid_table["L1-dcache-loads"])
                fam_pool["l2_request_miss_rate"] = (rapid_table["l2_request.miss"] /
                                                     rapid_table["l2_request.all"])
                fam_pool["l2_data_rd_miss_rate"] = (rapid_table["l2_rqsts.demand_data_rd_miss"] /
                                                     rapid_table["l2_rqsts.all_demand_data_rd"])
                fam_pool["LLC_load_miss_rate"] = (rapid_table["LLC-load-misses"] /
                                                   rapid_table["LLC-loads"])
                fam_pool["l2_hwpf_miss_rate"] = (rapid_table["l2_rqsts.hwpf_miss"] /
                                                  rapid_table["l2_rqsts.all_hwpf"])
                fam_pool["l2_swpf_miss_rate"] = (rapid_table["l2_rqsts.swpf_miss"] /
                                                  (rapid_table["l2_rqsts.swpf_miss"] + rapid_table["l2_rqsts.swpf_hit"]))
                fam_pool["instr_per_cycle"] = (rapid_table["instructions"] /
                                                rapid_table["cycles"])
                fam_pool["load_per_instr"] = (rapid_table["mem_inst_retired.all_loads"] /
                                               rapid_table["instructions"])
                fam_pool["tma_l1_bound"] = rapid_table["tma_l1_bound"]
                fam_pool["tma_l2_bound"] = rapid_table["tma_l2_bound"]
                fam_pool["tma_l3_bound"] = rapid_table["tma_l3_bound"]
                fam_pool["tma_dram_bound"] = rapid_table["tma_dram_bound"]

                # # test
                # print(f"dram_pool")
                # pprint.pprint(dram_pool)
                # print(f"dram_results[matrix_size][tile_size][num_threads]")
                # pprint.pprint(dram_results[matrix_size][tile_size][num_threads])
                # # end test

    # Save to csv

    KEYS=[
        "l1_dcache_miss_rate",
        "l2_request_miss_rate",
        "l2_data_rd_miss_rate",
        "LLC_load_miss_rate",
        "l2_hwpf_miss_rate",
        "l2_swpf_miss_rate",
        "instr_per_cycle",
        "load_per_instr",
        "tma_l1_bound",
        "tma_l2_bound",
        "tma_l3_bound",
        "tma_dram_bound",
    ]


    """
    matrix-size,event0,event1,event2,...
    32,xxx,xxx,xxx,...
    64,xxx,xxx,xxx,...
    128,xxx,xxx,xxx,...
    ...
    """


    for tile_size in tile_size_list:
        for num_threads in num_threads_list:

            dram_data = []
            fam_data = []
            for matrix_size in matrix_size_list:
                dram_row = {"matrix_size": matrix_size}
                fam_row = {"matrix_size": matrix_size}
                for key in KEYS:
                    dram_row[key] = dram_results[matrix_size][tile_size][num_threads][key]
                    fam_row[key] = fam_results[matrix_size][tile_size][num_threads][key]
                dram_data.append(dram_row)
                fam_data.append(fam_row)

            # Save to csv
            dram_df = pd.DataFrame(dram_data)
            dram_df.to_csv(f"{DRAM_BASE_NAME}.collect.tile-{tile_size}.thread-{num_threads}.matrix-sizes.csv", index=False)
            fam_df = pd.DataFrame(fam_data)
            fam_df.to_csv(f"{RAPID_BASE_NAME}.collect.tile-{tile_size}.thread-{num_threads}.matrix-sizes.csv", index=False)

    TT_TIME_END = time.perf_counter()
    print(f"\nTT_EXE_TIME(S): {TT_TIME_END - TT_TIME_START}")