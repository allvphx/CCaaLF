#ifndef POLICY_H_
#define POLICY_H_

#include <cmath>
#include <map>
#include <string>
#include <vector>
#include "macros.h"
#include "cstring"

#define RETRY_TIMES 3

#define WL_TPCC 0
#define WL_YCSB 1
#define WORKLOAD_TYPE WL_TPCC

// Type features: all or nothing.
#define ENCODER_TX_TYPE       0
#define ENCODER_TX_OP_TYPE    1

// Value features: nothing, < x, all, sqrt, or log.
#define ENCODER_TX_N_OP       2
#define ENCODER_TX_BLOCKED_ON 3
#define ENCODER_TX_BLOCKING   4
#define ENCODER_TX_N_BLOCKED  5

#define ENCODER_N_FEATURES    6

/************************************************/
// Workload helper
/************************************************/
#define MAX_ACC_ID std::numeric_limits<uint32_t>::max()
// type1: neworder, type2: payment, type3: delivery
#if WORKLOAD_TYPE == WL_TPCC
#define TXN_TYPE 3
#define ACCESSES 26
const uint32_t txn_access_num[] = { 0 /*just a placeholder*/, 11, 7, 8};
const uint32_t base_access_num[] = {0, 0, 11, 18};
const bool can_guard[]  = {0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0};
const bool can_expose[] = {1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1};
const int32_t encoder_feature_cap[ENCODER_N_FEATURES] = {1, 3, 26, 16, 16, 16};
// the guard needed to avoid triggering cycle.
extern uint32_t txn_guard[TXN_TYPE][TXN_TYPE][11];
#elif WORKLOAD_TYPE == WL_YCSB
#define TXN_TYPE 1
#define ACCESSES 20
const uint32_t txn_access_num[] = { 0 /*just a placeholder*/, 16};
const uint32_t base_access_num[] = {0, 0};
const bool can_expose[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
const int32_t encoder_feature_cap[ENCODER_N_FEATURES] = {1, 5, 16, 16, 16, 16};
#endif

/************************************************/
// Learning helper
/************************************************/
// We encode the dependency graph by embedding its key features into a 0/1 vector.
// The lower EVENT_BITS bits capture the feature of each transaction itself, the rest GRAPH_BITS capture graphic features.
#define MAX_STATE 3000

// Optimization 1: if there is no dependency related to the current transaction, there is no reason for us to trigger abort.
#define NO_REASON_TO_ABORT(state) ((((state)>>EVENT_BITS) & MAX_GRAPH_MASK) == GRAPHIC_NO_CONFLICT)
// Optimization 2: add learned timeout to wait policies.
#define ADD_TIMEOUT true

const uint32_t blocked_wait = 100 * 1000;
//const uint32_t timeout_choices[6] = {1, 10, 100, 1000, 10000, 100000};

enum AccessPolicy : unsigned char {
  no_detect,
  // do not detect anything and read committed version.
  detect_track_dirty,
  // read from dirty data and track the dependency.
  detect_guarded,
  // wait all pending transactions to safe guards and read dirty data.
  detect_all,
  // wait all pending transactions to commit/abort and read dirty data.
  predict // TODO: implement the predict policy.
};

typedef float WaitPriority;
const WaitPriority lowest_priority = 0;
// for an operation from T with lowest priority
// - all transactions depend on T skips this operation during do_wait.
// - before the execution of this operation, all dependents are blocked to wait to finish.
const WaitPriority highest_priority = 1;
// for an operation from T with highest priority
// - all transactions depend on T wait after this operation during do_wait.
// - before the execution of this operation, all dependents are skipped.

enum AgentDecision : unsigned char {
  agent_inc_backoff,
  agent_dec_backoff,
  agent_nop_backoff
};

typedef std::pair<AgentDecision, double> backoff_action;

/************************************************/
// Profiling helper
/************************************************/
// The detailed abort reason for each transaction that got aborted during FlexiCC's conflict resolving.
#define N_TX_LOCK_ABORT_REASON 6
#define TX_DEP_ORDER_FAIL 0         // aborted due to violation of dependency order during expose.
#define TX_CASCADING_ABORT 1        // aborted due to the read of data from a dirty transaction.
#define TX_LOCK_TLE 2               // aborted due to the timeout of wait.
#define TX_DEADLOCK_CHECK_FAIL 3    // aborted due to the failure of deadlock check.
#define TX_NONE 4                   // did not get aborted.
#define TX_OTHERS 5                 // aborted due to other reasons.

// PolicyAction contains the basic unit of policy.
struct PolicyAction {
  AccessPolicy access;    // control conflict detection.
  bool expose;            // shall we expose current data version.
  uint32_t safeguard[TXN_TYPE];
  uint32_t expose_safeguard[TXN_TYPE];
  WaitPriority rank;

  // we shall make the expose of current write as close as possible with the next op to avoid depend cycle.
  volatile bool lazy_mark = false;  // we calculate the expose in a lazy manner.
  WaitPriority expose_rank;
  AccessPolicy expose_access;
#if ADD_TIMEOUT
  uint32_t timeout;
  uint32_t expose_timeout;
#endif
  CACHE_PADOUT;

  PolicyAction() {
    access = detect_all;
    expose = false;
    memset(safeguard, 0, sizeof safeguard);
    rank = lowest_priority;
    expose_rank = lowest_priority;
    expose_access = detect_all;
#if ADD_TIMEOUT
    timeout = blocked_wait;
    expose_timeout = blocked_wait;
#endif
  }

  void copy(PolicyAction *act) {
    access = act->access;
    expose = act->expose;
    for (int i=0;i<TXN_TYPE;i++) safeguard[i] = act->safeguard[i];
    rank = act->rank;
    expose_rank = act->expose_rank;
    expose_access = act->expose_access;
#if ADD_TIMEOUT
    timeout = act->timeout;
    expose_timeout = act->timeout;
#endif
  }

  PolicyAction(AccessPolicy c_detect, double c_rank, bool c_expose, uint32_t c_resolve_tl) {
    access = c_detect;
    rank = c_rank;
    expose = c_expose;
    // By default, we assume the operation will commit and
    // thus take the highest priority to minimize potential conflicts.
    expose_rank = highest_priority;
    expose_access = c_detect;
#if ADD_TIMEOUT
    timeout = c_resolve_tl;
    expose_timeout = c_resolve_tl;
#endif
  }
};

extern PolicyAction before_commit_policy;


using backoff_info = double[2][RETRY_TIMES][TXN_TYPE + 1];
// Policy contains a cached policy inside memory.
class Policy {
private:
  const std::string identifier;
  // frequently accessed, we perform memory alignment for it.
  PolicyAction *policy;
  backoff_info backoff;
  uint32_t txn_buf_size = 32;
  CACHE_PADOUT;

public:
  Policy();
  Policy(const std::string &s) {
    ALIGN_PTR(policy, MAX_STATE, PolicyAction);
    policy_gradient(s);
  }
  ~Policy();

  void print_policy(const std::string &bench) const;
  void init_2pl();
  void init_pipeline_execution();
  void init_occ();
  // Load policy from target stream
  void init(std::ifstream *pol_file);
  void policy_gradient(const std::string &policy_f);

  ALWAYS_INLINE PolicyAction* inference(const uint32_t &state) const {
    return &policy[state];
  }


  ALWAYS_INLINE uint32_t get_txn_buf_size() {
    return txn_buf_size;
  }

  std::string get_identifier() {
    return identifier;
  }

  backoff_action inference_backoff_action(bool commit_success, uint16_t which_retry, uint32_t txn_type) const {
    if (txn_type == 0) return  backoff_action(agent_nop_backoff, 0);
    uint16_t retry_times = which_retry > 2 ? 2 : which_retry;
    if (commit_success)
      return backoff_action(agent_dec_backoff, backoff[1][retry_times][txn_type]);
    else
      return backoff_action(agent_inc_backoff, backoff[0][retry_times][txn_type]);
  }
};
#endif /* POLICY_H_ */