#ifndef AE_TPCC_FLEXIL_PROFILER_H
#define AE_TPCC_FLEXIL_PROFILER_H

#include "learn.h"
struct Profiler {
    uint64_t tid;
    volatile bool lock_ready;
    // for thrashing prevention techniques.
    uint16_t n_tx_active;

    // for conflict ratio thrashing: (n_locks/n_locks_blocked <= 1.3).
    uint32_t n_locks;
    uint32_t n_locks_blocked;

    // for half-half technique.
    uint32_t n_avg_tx;  // the average length of transactions that half finished.
    xact* locked_xact;

};

#endif//AE_TPCC_FLEXIL_PROFILER_H
