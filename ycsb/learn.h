#ifndef FLEXIL_LEARN_H
#define FLEXIL_LEARN_H

#include "amd64.h"
#include "cstring"
#include "policy.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <pthread.h>
#include <set>
#include "fstream"
#include "iostream"
#include "macros.h"

const double eps = 1e-3;  // float point number correction.

/************************************************/
// CONFIG helper
/************************************************/
#define TRACK_FULL_DEPENDENCY false // track fine-grained dependency information, this could be too time consuming.

enum OpType {
  OpRead,
  OpUpdate,
  OpCommit,
  OpInsert,
  OpScan,
  OpNone
};

#define REP(i, s, t) for(int (i)=(s);(i)<(t);(i)++)

extern uint64_t get_clock_ts();
struct plan_listener;
extern plan_listener global_listener;
struct xact;

struct plan_listener {
  int tx_n_blocked = 0;
  int tx_n_pending = 0;
  ALIGN_MEM int abort_distribution[14] = {0};
  ALIGN_MEM int state_distribution[MAX_STATE] = {0};
  int n_lock_get = 0;
  int num_state = 0;
  CACHE_PADOUT;

  plan_listener() {
    num_state = 0;
    memset(abort_distribution, 0, sizeof abort_distribution);
    memset(state_distribution, 0, sizeof state_distribution);
  }

  void print_state_distribution(const std::string &s) {
    printf("Profile: the state distribution\n"
           "<--------------------------------------->\n");
    if (s == "tpcc") {
      REP(i, 0, global_listener.num_state)
        printf("%d,", state_distribution[i]);
    } else if (s == "ycsb") {
    }
    printf("<--------------------------------------->\n");
  }

  void print_abort_distribution() {
    printf("Profile: the abort reason distribution\n"
           "<--------------------------------------->\n");

    std::string names[] = {
        "ABORT_REASON_NONE",
        "ABORT_REASON_USER",
        "ABORT_REASON_CASCADING",
        "ABORT_REASON_UNSTABLE_READ",
        "ABORT_REASON_FUTURE_TID_READ",
        "ABORT_REASON_NODE_SCAN_WRITE_VERSION_CHANGED",
        "ABORT_REASON_NODE_SCAN_READ_VERSION_CHANGED",
        "ABORT_REASON_WRITE_NODE_INTERFERENCE",
        "ABORT_REASON_INSERT_NODE_INTERFERENCE",
        "ABORT_REASON_READ_NODE_INTEREFERENCE",
        "ABORT_REASON_READ_ABSENCE_INTEREFERENCE",
        "ABORT_REASON_LOCK_CONFLICT",
        "ABORT_REASON_TIMEOUT",
        "ABORT_REASON_EARLY_VALIDATION_FAIL",
        ""
    };

    for (int i=0;i<14;i++)
      printf("%s: %d\n", names[i].c_str(), abort_distribution[i]);
    printf("<--------------------------------------->\n");
  }
};

extern std::atomic<uint64_t> cur_max_ts;

inline uint64_t get_dl_ts(bool is_largest)
{
  assert(is_largest); // if not the largest,
                      // a pending transaction could have read the current transaction and get aborted.
  return cur_max_ts.fetch_add(1);
}

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

  ALWAYS_INLINE int inference_step(const int &steps, const int &tx_type) {
    int capped_steps = likely(steps < encoding_cap[ENCODER_TX_N_OP]) ? steps: encoding_cap[ENCODER_TX_N_OP] - 1;
    // avoid branching and function stack call.
    return capped_steps + rev_prod[tx_type-1];
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
    return res;
  }
};

extern contention_encoder global_encoder;

struct xact {
  const uint64_t tid;
  const uint8_t tx_type;        // transaction type.
  // Event information.
  uint32_t tx_n_op;      // number of executed transaction operations.
  uint32_t cur_acc_id = 0;
  OpType tx_cur_op;          // the type of currently executed operation.

  // Graphical information.
#if TRACK_FULL_DEPENDENCY
  // tracking full depedencies in a thread-safe way could be too time consuming.
  std::set<xact*> blocking, blocked_on;
  pthread_spinlock_t latch;
#else
  uint32_t tx_n_dep_on;   // number of transactions current transaction depend on (current tx read their dirty data).
  uint32_t tx_n_dep_by;   // number of transactions that depends on current transaction (they read this tx dirty data).
#endif
  uint8_t debug_bits; // for debugging purpose.
  // Some bugs only happen regarding high concurrency and are not reproducible in GDB.
  // In this case, we use the debug bits for static debugging purpose.
  uint32_t state;
  CACHE_PADOUT;

  ALWAYS_INLINE uint32_t encode() const {
    if (likely(global_encoder.access_only)) {
      return cur_acc_id;
//      // aggressive optimization for this branch (fast path).
//      return global_encoder.inference_step(tx_n_op, tx_type);
    }
    const int feature[ENCODER_N_FEATURES] = {
        (tx_type-1),
        tx_cur_op,
        tx_n_op,
        tx_n_dep_on,
        tx_n_dep_by,
        global_listener.tx_n_blocked};
    return global_encoder.inference(feature);
  }

  std::string debug_info() const {
  }

  explicit xact(uint64_t _tid, uint8_t _tx_type): tid(_tid), tx_type(_tx_type) {
    tx_n_op = 0;
    tx_n_dep_by = 0;
    tx_n_dep_on = 0;
    tx_cur_op = OpNone;
    debug_bits = 0;
#if TRACK_FULL_DEPENDENCY
    blocking.clear();
    blocking.insert(this);
    blocked_on.clear();
    pthread_spin_init(&latch, PTHREAD_PROCESS_PRIVATE);
#else
#endif
    state = 0;
  }

  ~xact() {
  }

  ALWAYS_INLINE PolicyAction* get_cur_policy(const Policy *pg) {
    assert(tx_type > 0);
    state = encode();
    auto tmp = pg->inference(state);
    if (likely(tx_cur_op != OpCommit || tmp->lazy_mark)) {
      return tmp;
    } else if (tx_n_op != txn_access_num[tx_type]) {
      // if not the final operation, we use the wait policy for next operation as expose wait policy.
      tx_n_op ++;
      cur_acc_id ++;
      state = encode();
      auto next_tmp = pg->inference(state);
      tmp->expose_access = next_tmp->access;
      tmp->expose_rank = next_tmp->rank;
      tmp->expose_timeout = next_tmp->timeout;
      for (int i=0;i<TXN_TYPE;i++)
        tmp->expose_safeguard[i] = next_tmp->safeguard[i];
      tmp->lazy_mark = true;
      tx_n_op --;
      cur_acc_id ++;
    }
    return tmp;
  }
};

extern void profiling(const std::string& s);

#endif //FLEXIL_LEARN_H