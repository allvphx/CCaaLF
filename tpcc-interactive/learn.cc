#include "learn.h"
#include "tuple.h"
#include <cassert>
#include <chrono>

std::chrono::milliseconds zeroMicrosecondsDefault =  std::chrono::milliseconds(0);
uint8_t s = 0;
Policy policy[2];
int max_k = 0;
plan_listener global_listener;
contention_encoder global_encoder;

void xact::update_dependency(xact *blocked, bool is_remove)
{
#if TRACK_FULL_DEPENDENCY
    pthread_spin_lock(&latch);
    if (!is_remove)
    {
        blocked->blocked_on.insert(this);
        // calculate cascading dependencies.
        blocking.insert(blocked->blocking.begin(), blocked->blocking.end());
        // there is no cascading add set, since we take cautious wait policy.
    }
    else
    {
        auto itr = blocking.find(blocked);
        // blocked got released, thus no need to update it.
        if (itr == blocking.end()) return;
        blocking.erase(itr);
        for (auto it : blocked_on)
            it->update_dependency(blocked, false);
    }
    pthread_spin_unlock(&latch);
#else
    if (!is_remove) {
        tx_n_dep_by++;
        blocked->tx_n_dep_on ++;
    }
    else {
        blocked->tx_n_dep_on --;
        tx_n_dep_by --;
        assert(tx_n_dep_by >= 0);
    }
#endif
}

inline int min(int x, int y)
{
    if (x < y) return x;
    return y;
}

inline int cap(int x, int maxv)
{
    return min(x, maxv);
}

uint64_t get_clock_ts()
{
    auto now = std::chrono::system_clock::now();
    auto now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch());
    return now_ns.count();
}

void xact::refresh_state()
{
    cached_policy = get_cur_policy(&policy[0]);
#if PROFILE_LOCK
    if (cached_policy) {
        (global_listener.state_distribution[state] ++);
        (global_listener.num_state ++);
    }
#endif
}

void load_policy(const std::string &f)
{
    policy[0].policy_gradient(f);
}

void profiling()
{
    global_listener.print_abort_distribution();
    global_listener.print_state_distribution();
    global_listener.print_lock_latency_breakdown();
    policy[0].print_policy("tpcc");
}