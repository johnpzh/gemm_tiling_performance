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

    matrix_size_list = [int(2 ** i) for i in range(12, 14)] # 4096, 8192
    tile_size_list = [int(2 ** i) for i in range(3, 12)] # 8 - 2048
    num_threads_list = [int(2 ** i) for i in range(0, 5)] # 1 - 16
    num_threads_list.append(28)
    # num_threads_list = [32, 56] #
    # matrix_size_list = [int(2 ** i) for i in range(8, 10)]
    # tile_size_list = [int(2 ** i) for i in range(4, 6)]
    # num_threads_list = [int(2 ** i) for i in range(3, 5)]
    # num_threads_list.append(28)

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
    ------,thread-1,thread-2,thread-4
    event0,
    event1,
    event2,
    """
    threads = [f"thread-{x}" for x in num_threads_list]
    for matrix_size in matrix_size_list:
        for tile_size in tile_size_list:
            dram_pool = {}
            fam_pool = {}

            for num_threads in num_threads_list:
                dram_pool[num_threads] = []
                fam_pool[num_threads] = []
                for key in KEYS:
                    dram_pool[num_threads].append(dram_results[matrix_size][tile_size][num_threads][key])
                    fam_pool[num_threads].append(fam_results[matrix_size][tile_size][num_threads][key])

            # DRAM
            data = {
                "Perf_Event": KEYS,
            }
            for num_threads in num_threads_list:
                data[f"thread-{num_threads}"] = dram_pool[num_threads]
            df = pd.DataFrame(data)
            output_file = f"{DRAM_BASE_NAME}.collect.matrix-{matrix_size}.tile-{tile_size}.threads.csv"
            df.to_csv(output_file, index=False)
            print("")
            print(df.to_string())
            print(f"\nSaved to {output_file}")

            # FAM
            data = {
                "Perf_Event": KEYS,
            }
            for num_threads in num_threads_list:
                data[f"thread-{num_threads}"] = fam_pool[num_threads]
            df = pd.DataFrame(data)
            output_file = f"{RAPID_BASE_NAME}.collect.matrix-{matrix_size}.tile-{tile_size}.threads.csv"
            df.to_csv(output_file, index=False)
            print("")
            print(df.to_string())
            print(f"\nSaved to {output_file}")

    """
    ------,tile-8,tile-16,tile-32
    event0,
    event1,
    event2,
    """
    tiles = [f"tile-{x}" for x in tile_size_list]
    for matrix_size in matrix_size_list:
        for num_threads in num_threads_list:
            dram_pool = {}
            fam_pool = {}

            for tile_size in tile_size_list:
                dram_pool[tile_size] = []
                fam_pool[tile_size] = []
                for key in KEYS:
                    dram_pool[tile_size].append(dram_results[matrix_size][tile_size][num_threads][key])
                    fam_pool[tile_size].append(fam_results[matrix_size][tile_size][num_threads][key])

            # # test
            # print("dram_pool")
            # pprint.pprint(dram_pool)
            # # end test

            # DRAM
            data = {
                "Perf_Event": KEYS,
            }
            for tile_size in tile_size_list:
                data[f"tile-dim-size-{tile_size}"] = dram_pool[tile_size]
            df = pd.DataFrame(data)
            output_file = f"{DRAM_BASE_NAME}.collect.matrix-{matrix_size}.thread-{num_threads}.tile-dim-sizes.csv"
            df.to_csv(output_file, index=False)
            print("")
            print(df.to_string())
            print(f"\nSaved to {output_file}")

            # FAM
            data = {
                "Perf_Event": KEYS,
            }
            for tile_size in tile_size_list:
                data[f"tile-dim-size-{tile_size}"] = fam_pool[tile_size]
            df = pd.DataFrame(data)
            output_file = f"{RAPID_BASE_NAME}.collect.matrix-{matrix_size}.thread-{num_threads}.tile-dim-sizes.csv"
            df.to_csv(output_file, index=False)
            print("")
            print(df.to_string())
            print(f"\nSaved to {output_file}")


    TT_TIME_END = time.perf_counter()
    print(f"\nTT_EXE_TIME(S): {TT_TIME_END - TT_TIME_START}")