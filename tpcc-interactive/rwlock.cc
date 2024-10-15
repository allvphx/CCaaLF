#include "rwlock.h"
#include "ctime"
#include "chrono"

LockEntry::LockEntry() {
    type = LOCK_NONE;
    tid = 0;
    rank = 0;
    lock_ready = false;
    locked_xact = nullptr;
    next = nullptr;
    prev = nullptr;
}

RWLock::RWLock() {
    owners = NULL;
    waiters_head = NULL;
    waiters_tail = NULL;
    owner_cnt = 0;
    sorted = true;

    latch = new pthread_mutex_t;
    pthread_mutex_init(latch, NULL);

    lock_type = LOCK_NONE;
}

bool RWLock::lockEmpty(uint64_t tid) const {
    auto tmp = owners;
    auto typ = lock_type;
    return typ == LOCK_NONE || tmp == nullptr || (tmp->next == nullptr && tmp->tid == tid);
}

bool RWLock::lockNotModified(uint64_t tid) const {
    auto tmp = owners;
    auto typ = lock_type;
    return typ == LOCK_NONE || typ == LOCK_SH || tmp == nullptr || (tmp->next == nullptr && tmp->tid == tid);
}

bool RWLock::lockW(uint64_t tid, xact* xact, bool not_sorted) {
    assert(tid == xact->tid);
    return lock_get(LOCK_EX, tid, not_sorted, xact);
}

void RWLock::unlockW(uint64_t tid) {
    return lock_release(tid);
}

bool RWLock::lockR(uint64_t tid, xact* xact, bool not_sorted) {
    assert(tid == xact->tid);
    return lock_get(LOCK_SH, tid, not_sorted, xact);
}

void RWLock::unlockR(uint64_t tid) {
    return lock_release(tid);
}

bool RWLock::lock_get(uint16_t type, uint64_t tid, bool not_sorted, xact* xact) {
#if PROFILE_LOCK
    auto begin_ts = get_clock_ts();
    int stage = 0;
    global_listener.n_lock_get ++;
#endif
    pthread_mutex_lock(latch);
    if (not_sorted) sorted = false;
#if PROFILE_LOCK
    // blocking stage 0: get pthread lock.
    INC_TIME_SPAN(blocking_span[stage++], get_clock_ts() - begin_ts);
    begin_ts = get_clock_ts();
#endif
    assert(xact->cached_policy);
    bool conflict = conflict_lock(lock_type, type);
    WaitPriority rank = xact->cached_policy->rank;
    uint32_t timeout = xact->cached_policy->timeout;
    assert(!xact->is_blocked);
    assert(xact->conflict_mask == 0);
#if PROFILE_LOCK
    // blocking stage 1: get operation policy.
    INC_TIME_SPAN(blocking_span[stage++], get_clock_ts() - begin_ts);
    begin_ts = get_clock_ts();
#endif

    if (!conflict) {
        if (waiters_head && IS_PRIORI(waiters_head->rank, rank))
            conflict = true;
    }

    if (conflict) {
        bool can_wait = true;
        if (not_sorted) {
            for (auto en = owners; en != nullptr && can_wait; en = en->next) {
                if (xact->has_deadlock(en->locked_xact)) can_wait = false;
            }
        }
#if PROFILE_LOCK
        // blocking stage 2: check transaction deadlock and wait priority.
        INC_TIME_SPAN(blocking_span[stage++], get_clock_ts() - begin_ts);
        begin_ts = get_clock_ts();
#endif

        if (can_wait) {
            LockEntry* entry = get_entry();
            entry->tid = tid;
            entry->rank = rank;
            entry->type = type;
            entry->lock_ready = false;
            entry->locked_xact = xact;
            for (auto en = owners; en; en = en->next) {
                en->locked_xact->update_dependency(xact, false);
                if (not_sorted) xact->merge(en->locked_xact);
            }
#if PROFILE_LOCK
            // blocking stage 3: in case of wait, update of conflict information.
            INC_TIME_SPAN(blocking_span[stage++], get_clock_ts() - begin_ts);
            begin_ts = get_clock_ts();
#endif

            put_waiter(entry);
            xact->is_blocked = true;
            pthread_mutex_unlock(latch);
#if PROFILE_LOCK
            // blocking stage 4: in case of wait, add waiter and lock release.
            INC_TIME_SPAN(blocking_span[stage++], get_clock_ts() - begin_ts);
            begin_ts = get_clock_ts();
#endif
            if (timeout == 0)
                while (!entry->lock_ready) nop_pause();
            else
            {
                timespec start{}, now{};
                clock_gettime(CLOCK_MONOTONIC, &start);
                while (!entry->lock_ready)
                {
                    clock_gettime(CLOCK_MONOTONIC, &now);
                    long elapsed = (now.tv_sec - start.tv_sec) * 1000000 + (now.tv_nsec - start.tv_nsec) / 1000;
                    if (elapsed >= timeout * 1000) break;
                    nop_pause();
                }
            }
#if PROFILE_LOCK
            // blocking stage 5: in case of wait, busy loop.
            INC_TIME_SPAN(blocking_span[stage++], get_clock_ts() - begin_ts);
            begin_ts = get_clock_ts();
#endif
            xact->is_blocked = false;
            return entry->lock_ready;
        } else {
            pthread_mutex_unlock(latch);
            return false;
        }
    } else {
        LockEntry* entry = get_entry();
        entry->type = type;
        entry->tid = tid;
        entry->rank = rank;
        entry->locked_xact = xact;
        STACK_PUSH(owners, entry);
#if !ONLY_COUNT_PASSIVE_WAIT
        for (auto it = waiters_head; it; it = it->next)
            entry->locked_xact->update_dependency(it->locked_xact, false);
#endif
        owner_cnt++;
        lock_type = type;
        check_correctness();
        pthread_mutex_unlock(latch);
        return true;
    }
}

void RWLock::lock_release(uint64_t tid) {
    pthread_mutex_lock(latch);
    LockEntry* en = owners;
    LockEntry* prev = nullptr;

    while (en != nullptr && en->tid != tid) {
        prev = en;
        en = en->next;
    }
    if (en) {
        if (prev) prev->next = en->next;
        else owners = en->next;
        return_entry(en);
        owner_cnt--;
        if (owner_cnt == 0)
        lock_type = LOCK_NONE;
    } else {
        en = waiters_head;
        while (en != nullptr && en->tid != tid) en = en->next;
        if (!en)
        {
            pthread_mutex_unlock(latch);
            return;
        }
        LIST_REMOVE_HT(en, waiters_head, waiters_tail);
        for (auto it = owners; it; it = it->next)
            it->locked_xact->update_dependency(en->locked_xact, true);
        return_entry(en);
    }
    promote();
    pthread_mutex_unlock(latch);
}

void RWLock::put_waiter(LockEntry *entry)
{
#if REAL_TIME_PRIORITY
    LIST_PUT_TAIL(waiters_head, waiters_tail, entry);
    check_correctness();
#else
    auto en = waiters_head;
    while (en != nullptr && IS_PRIORI(en->rank, entry->rank))
        en = en->next;
    if (en) {
        LIST_INSERT_BEFORE(en, entry);
        if (en == waiters_head)
            waiters_head = entry;
    } else {
        LIST_PUT_TAIL(waiters_head, waiters_tail, entry);
    }
    check_correctness();
#endif
}

LockEntry* RWLock::pop_best()
{
    LockEntry *res  = nullptr;
    LockEntry* entry;
#if REAL_TIME_PRIORITY
    float optimal_rank = 1.0;
    for (auto it = waiters_head; it; it = it->next)
        if (policy_wait_priority(it->locked_xact->state) < optimal_rank) res = it;
    if (!res || conflict_lock(lock_type, res->type)) return nullptr;
    if(res)
        LIST_REMOVE_HT(res, waiters_head, waiters_tail);
#else
    if (waiters_head && !conflict_lock(lock_type, waiters_head->type))
        LIST_GET_HEAD(waiters_head, waiters_tail, res);
#endif
    return res;
}

void RWLock::promote()
{
    for (auto entry=pop_best(); entry; entry = pop_best())
    {
#if !ONLY_COUNT_PASSIVE_WAIT
        for (auto it = owners; it; it = it->next)
            it->locked_xact->update_dependency(entry->locked_xact, true);
        for (auto it = waiters_head; it; it = it->next)
            entry->locked_xact->update_dependency(it->locked_xact, false);
#endif
        STACK_PUSH(owners, entry);
        owner_cnt++;
        entry->lock_ready = true;
        lock_type = entry->type;
    }
}

bool RWLock::conflict_lock(uint16_t l1, uint16_t l2) {
    if (l1 == LOCK_NONE || l2 == LOCK_NONE)
        return false;
    else if (l1 == LOCK_EX || l2 == LOCK_EX)
        return true;
    else
        return false;
}

LockEntry* RWLock::get_entry() {
    return new LockEntry();
}

void RWLock::return_entry(LockEntry* entry) {
    delete entry;
}

void RWLock::check_correctness() {
#if DEBUG
    // waiters sorted by the priority.
    for (auto it = waiters_head; it; it = it->next)
        if (it->next) assert(IS_PRIORI_OR_EQ(it->rank, it->next->rank));
#endif
}