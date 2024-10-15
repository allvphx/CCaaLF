#ifndef FLEXIL_LEARN_H
#define FLEXIL_LEARN_H

#include <set>
#include "policy.h"
#include "cstring"
#include <pthread.h>
#include <chrono>
#include "fstream"
#include "iostream"


#define REP(i, s, t) for(int (i)=(s);(i)<(t);(i)++)

/************************************************/
// CONFIG helper
/************************************************/
#define CAUTIOUS_WAIT 0
#define BIT_CHECK 1
#define WAIT_DIE 2
#define DEADLOCK WAIT_DIE
#define REAL_TIME_PRIORITY false
#define TRACK_FULL_DEPENDENCY false
#define ONLY_COUNT_PASSIVE_WAIT true
#define PROFILING(expr) (expr)
#define PROFILE_LOCK false

#if DEADLOCK == BIT_CHECK
    #define DL_TID_TO_BIT1(id) (1ULL<<((id)%53))
    #define DL_TID_TO_BIT2(id) (1ULL<<((id)%59))
#endif

enum OpType {
    OpRead,
    OpUpdate,
    OpCommit,
    OpInsert,
    OpScan,
    OpNone
};

const int log2_values[] = {0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4,
                           4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5};

inline bool sigmoid(double x, double scale) {
    return scale / (1 + exp(-x));
}


enum EncodingType {
    EncodeIgnore = 0,
    EncodeIfNot,
    EncodeLog,
    EncodeLinear
};

struct contention_encoder {
    // global features.
    ALIGN_MEM EncodingType encode_type[ENCODER_N_FEATURES];
    // learn a cap.
    ALIGN_MEM int encoding_cap[ENCODER_N_FEATURES];
    ALIGN_MEM int rev_prod[ENCODER_N_FEATURES];
    int max_state = 1;
    bool access_only = false;
    CACHE_PADOUT;

    contention_encoder() {
        memset(encode_type, 0, sizeof (encode_type));
        memset(encoding_cap, 0, sizeof (encoding_cap));
        memset(rev_prod, 0, sizeof (rev_prod));
    }

    ~contention_encoder() {}

    inline int var_range(int i, int n) {
        if (likely(encode_type[i] == EncodeIgnore)) return 1;
        else if (likely(encode_type[i] == EncodeLinear))
            return n;
        else if (encode_type[i] == EncodeIfNot) return 2;
        else if (encode_type[i] == EncodeLog)
            return log2_values[n] + 1;
        else assert(false);
    }

    void load(const std::string &encoder_f) {
        access_only = true;
        if (encoder_f == "step") {
            max_state = TXN_TYPE * encoder_feature_cap[ENCODER_TX_N_OP];
            for (int i = 0; i < ENCODER_N_FEATURES; i++) {
                encode_type[i] = EncodeIgnore;
                encoding_cap[i] = 1;
            }
            encode_type[ENCODER_TX_TYPE] = EncodeLinear;
            encode_type[ENCODER_TX_N_OP] = EncodeLinear;
            encoding_cap[ENCODER_TX_TYPE] = encoder_feature_cap[ENCODER_TX_TYPE];
            encoding_cap[ENCODER_TX_N_OP] = encoder_feature_cap[ENCODER_TX_N_OP];
        } else {
            max_state = 1;
            std::ifstream pol_file(encoder_f);
            if (!pol_file) {
                std::cerr << "Could not open file: " << encoder_f << ". "
                          << "Please specify correct encoder file with the --encoder option"
                          << std::endl;
                ALWAYS_ASSERT(false);
            }
            for (int i = 0; i < ENCODER_N_FEATURES; i++) {
                int tmp;
                pol_file >> tmp;
                encode_type[i] = EncodingType(tmp);
                if (i != ENCODER_TX_TYPE && i != ENCODER_TX_N_OP && encode_type[i] != EncodeIgnore) {
                    access_only = false;
                }
            }
            for (int i = 0; i < ENCODER_N_FEATURES; i++) {
                pol_file >> encoding_cap[i];
                assert(encoding_cap[i] > 0);
                if (encoding_cap[i] > encoder_feature_cap[i]) {
                    encoding_cap[i] = encoder_feature_cap[i];
                }
            }
            pol_file.close();
            for (int i = 0; i < ENCODER_N_FEATURES; i++)
                max_state *= var_range(i, encoding_cap[i]);
        }
        if (access_only) {
            // In case of access only, we only need to cache productions for txn_length.
            assert(encoding_cap[ENCODER_TX_TYPE] <= ENCODER_N_FEATURES);  // hack.
            for(int i=0;i<encoding_cap[ENCODER_TX_TYPE];i++)
                rev_prod[i] = i * encoding_cap[ENCODER_TX_N_OP];
        } else {
            int cur_prod = 1;
            for (int i=ENCODER_N_FEATURES-1;i>=0;i--) {
                rev_prod[i] = cur_prod;
                cur_prod *= var_range(i, encoding_cap[i]);
            }
        }
    }

    ALWAYS_INLINE int inference(const int x[ENCODER_N_FEATURES]) {
        assert(!access_only);
        int res = 0;
        for (int i=ENCODER_N_FEATURES-1;i>=0;i--) {
            auto tmp = x[i] >= encoding_cap[i]? encoding_cap[i]-1 :x[i];
            if (encode_type[i] == EncodeIgnore)
                continue ;
            else if (encode_type[i] == EncodeIfNot)
                res += tmp > 0?  rev_prod[i]: 0;
            else if (encode_type[i] == EncodeLog)
                res += log2_values[tmp] * rev_prod[i];
            else if (encode_type[i] == EncodeLinear)
                res += tmp * rev_prod[i];
        }
//        printf("encoded = %d\n", res);
        return res;
    }
};

extern contention_encoder global_encoder;

struct xact {
    uint64_t tid;
    uint8_t conflict_mask;
    bool validating;
    bool is_blocked;
    const uint8_t tx_type = 1;  // there is no tx_type in interactive mode.
    uint32_t tx_n_op;      // number of executed transaction operations.
    OpType tx_cur_op;          // the type of currently executed operation.

    uint64_t deadlock_check_bits_mask1, deadlock_check_bits_mask2;
    PolicyAction *cached_policy;
#if TRACK_FULL_DEPENDENCY
    std::set<xact*> blocking, blocked_on;
    pthread_spinlock_t latch;
#else
    uint32_t tx_n_dep_on;   // number of transactions current transaction depend on (current tx block to read their data).
    uint32_t tx_n_dep_by;   // number of transactions that depends on current transaction (they read this tx data).
#endif
    uint8_t debug_bits; // for debugging purpose.
    uint32_t state;
    CACHE_PADOUT;

    ALWAYS_INLINE uint32_t encode() const {
        if (likely(global_encoder.access_only)) {
            // aggressive optimization for this branch (fast path).
            return tx_n_op;
        }
        const int feature[ENCODER_N_FEATURES] = {
                (tx_type-1),
                tx_cur_op,
                tx_n_op,
                tx_n_dep_on,
                tx_n_dep_by,
                0}; // currently, we ignore this feature.
        return global_encoder.inference(feature);
    }

    explicit xact(uint64_t _tid) {
        tid = _tid;
        conflict_mask = 0;
        is_blocked = false;
        cached_policy = nullptr;
#if TRACK_FULL_DEPENDENCY
        blocking.clear();
        blocking.insert(this);
        blocked_on.clear();
        pthread_spin_init(&latch, PTHREAD_PROCESS_PRIVATE);
#else
        tx_n_op = 0;
        tx_n_dep_by = 0;
        tx_n_dep_on = 0;
        tx_cur_op = OpNone;
#endif
        validating = false;
#if DEADLOCK == BIT_CHECK
        deadlock_check_bits_mask1 = 0;
        deadlock_check_bits_mask2 = 0;
#endif
        state = 0;
    }

    // check_deadlock could report false positive.
    bool has_deadlock(xact* blocked_on) const {
#if DEADLOCK == BIT_CHECK
        return (deadlock_check_bits_mask1 & DL_TID_TO_BIT1(blocked_on->tid))
                && (deadlock_check_bits_mask2 & DL_TID_TO_BIT2(blocked_on->tid));
#elif DEADLOCK == CAUTIOUS_WAIT
        assert(!is_blocked);
        return blocked_on->is_blocked;
#elif DEADLOCK == WAIT_DIE
        return blocked_on->tid < tid;
#endif
    }

    void merge(xact *blocked_on) {
#if DEADLOCK == BIT_CHECK
        deadlock_check_bits_mask1 |= blocked_on->deadlock_check_bits_mask1;
        deadlock_check_bits_mask2 |= blocked_on->deadlock_check_bits_mask2;
        deadlock_check_bits_mask1 |= DL_TID_TO_BIT1(blocked_on->tid);
        deadlock_check_bits_mask2 |= DL_TID_TO_BIT2(blocked_on->tid);
#elif DEADLOCK == CAUTIOUS_WAIT
        assert(!blocked_on->is_blocked);
#elif DEADLOCK == WAIT_DIE
        assert(blocked_on->tid > tid);
#endif
    }

    ALWAYS_INLINE PolicyAction* get_cur_policy(const Policy *pg) {
        assert(tx_type > 0);
        state = encode();
        if (state == UINT32_MAX) {
            assert(false);
            return nullptr;
        }
        auto tmp = pg->inference(state);
        return tmp;
    }

    // update dependency graph, from blocked transaction to current xact.
    void update_dependency(xact *blocked, bool is_remove);
    void refresh_state();
    inline bool need_lock()
    {
        return cached_policy &&
               cached_policy->access >= detect_existing;
    }
    inline bool need_validate()
    {
        return cached_policy
               && cached_policy->access == detect_critical;
    }
};

extern void load_policy(const std::string &f);
extern void profiling();

struct plan_listener {
    int abort_distribution[14] = {0};
    int state_distribution[MAX_STATE] = {0};
    uint64_t blocking_span[6] = {0};
    int n_lock_get = 0;
    int num_state = 0;

    plan_listener()
    {
        num_state = 0;
        memset(abort_distribution, 0, sizeof abort_distribution);
        memset(state_distribution, 0, sizeof state_distribution);
    }

    void print_state_distribution()
    {
        printf("Profile: the state distribution\n"
               "<--------------------------------------->\n");
        REP(i, 0, global_encoder.max_state) printf("%d,", state_distribution[i]);
        printf("<--------------------------------------->\n");
    }

    void print_abort_distribution()
    {
        printf("Profile: the abort reason distribution\n"
               "<--------------------------------------->\n");
        std::string names[] = {
                "ABORT_REASON_NONE",
                "ABORT_REASON_USER",
                "ABORT_REASON_UNSTABLE_READ",
                "ABORT_REASON_FUTURE_TID_READ",
                "ABORT_REASON_NODE_SCAN_WRITE_VERSION_CHANGED",
                "ABORT_REASON_NODE_SCAN_READ_VERSION_CHANGED",
                "ABORT_REASON_WRITE_NODE_INTERFERENCE",
                "ABORT_REASON_INSERT_NODE_INTERFERENCE",
                "ABORT_REASON_READ_NODE_INTEREFERENCE",
                "ABORT_REASON_READ_ABSENCE_INTEREFERENCE",
                "ABORT_REASON_LOCK_CONFLICT",
                "ABORT_REASON_LEARNED_ABORT",
                "ABORT_REASON_VALIDATION_LOCK_FAIL",
                "ABORT_REASON_EARLY_VALIDATION_FAIL"
        };
        for (int i=0;i<14;i++)
            printf("%s: %d\n", names[i].c_str(), abort_distribution[i]);
        printf("<--------------------------------------->\n");
    }

    void print_lock_latency_breakdown()
    {
        printf("Profile: the get lock latency breakdown\n"
               "<--------------------------------------->\n");
        std::string names[] = {
                "PThread Latch",
                "Get Policy",
                "Check Deadlock and Wait Priority",
                "Update Conflict Info",
                "Add Waiter",
                "Busy Loop",
        };
        for (int i=0;i<6;i++)
            printf("%s: %.2f\n", names[i].c_str(), (double )blocking_span[i] / 1000000.0);
        printf("<--------------------------------------->\n");
    }

};


extern uint64_t get_clock_ts();

extern plan_listener global_listener;

#define INC_TIME_SPAN(v, span)  {\
    global_listener.v += span; }

#endif //FLEXIL_LEARN_H
