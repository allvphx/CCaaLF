#ifndef POLICY_H_
#define POLICY_H_

#include <cmath>
#include <map>
#include <string>
#include <vector>
#include "macros.h"

#define TXN_TYPE 1

/************************************************/
// Learning helper
/************************************************/
//#define F_CURRENT_IS_UPDATE     1
//#define F_CURRENT_IS_RISKY      1
//#define F_CURRENT_MODIFY_PROTECTED      2
//#define CONFLICT_BITS 0
//#define MAX_CONFLICT_MASK ((1 << (CONFLICT_BITS))-1)
//// K : How many transactions are blocked to wait on the current transaction.
//#define K_BITS 3
//#define MAX_K_MASK ((1 << (K_BITS))-1)
//// Step : the access type.
//#define STEP_BITS 4
//#define MAX_STEP_MASK ((1 << (STEP_BITS))-1)
//
//#define MAX_STATE ((1 << (CONFLICT_BITS + K_BITS + STEP_BITS)))
//#define ENCODE_STATE(c, k, s) (((c)&MAX_CONFLICT_MASK) | ((k)<<CONFLICT_BITS) | ((s)<<(CONFLICT_BITS + K_BITS)))
//// if there is no transaction depending on current transaction, & there is no clue that validation would fail, do not trigger abort,
//#define NO_REASON_TO_ABORT(state) (((state) & MAX_CONFLICT_MASK) <= 1 &&  (((state)>>CONFLICT_BITS) & MAX_K_MASK) == 0)
//#define IS_DEFAULT_POLICY (state == uint32_t(-1))
//#define SET_DEFAULT_POLICY (state = uint32_t(-1))
#define RETRY_TIMES 3

#define MAX_STATE 3000
#define NO_REASON_TO_ABORT(state) ((((state)>>EVENT_BITS) & MAX_GRAPH_MASK) == GRAPHIC_NO_CONFLICT)
#define ADD_TIMEOUT true
const uint32_t blocked_wait = 100 * 1000;

// Type features: all or nothing.
#define ENCODER_TX_TYPE       0     // this is not differentiable!
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
const int32_t encoder_feature_cap[ENCODER_N_FEATURES] = {1, 3, 11, 16, 16, 16};


typedef float WaitPriority;
const WaitPriority maximum_priority = 1.0;

#define DEFAULT_TIMEOUT 100000

enum AccessPolicy : unsigned char {
    no_detect = 0,
    detect_critical = 1,
    detect_existing = 2,
    detect_future = 3
};

// PolicyAction contains the basic unit of policy.
struct PolicyAction {
    AccessPolicy access;    // control conflict detection.
    WaitPriority rank;
#if ADD_TIMEOUT
    uint32_t timeout;
#endif
    CACHE_PADOUT;

    PolicyAction() {
        access = detect_existing;
        rank = 0;
#if ADD_TIMEOUT
        timeout = blocked_wait;
#endif
    }

    void copy(PolicyAction *act) {
        access = act->access;
        rank = act->rank;
#if ADD_TIMEOUT
        timeout = act->timeout;
#endif
    }

    PolicyAction(AccessPolicy c_detect, WaitPriority c_rank, uint32_t c_resolve_tl) {
        access = c_detect;
        rank = c_rank;
#if ADD_TIMEOUT
        timeout = c_resolve_tl;
#endif
    }
};


// Policy contains a cached policy inside memory.
class Policy {
private:
    const std::string identifier;
    // frequently accessed, we perform memory alignment for it.
    PolicyAction *policy;
    CACHE_PADOUT;

public:
    Policy();
    Policy(const std::string &s) {
        ALIGN_PTR(policy, MAX_STATE, PolicyAction);
        policy_gradient(s);
    }
    ~Policy();

    void print_policy(std::string bench);
    void init_2pl_wait_die();
    void init_2pl_no_wait();
    void init_occ();
    // Load policy from target stream
    void init(std::ifstream *pol_file);
    void policy_gradient(const std::string &policy_f);

    ALWAYS_INLINE PolicyAction* inference(const uint32_t &state) const {
        return &policy[state];
    }

    std::string get_identifier() {
        return identifier;
    }
};
#endif /* POLICY_H_ */