#ifndef _RWLOCK_H_
#define _RWLOCK_H_

#include <stdint.h>
#include <cstdint>
#include <atomic>
#include <pthread.h>
#include <cassert>
#include "amd64.h"
#include "macros.h"
#include "util.h"
#include "learn.h"
#include "math.h"

/************************************************/
// LIST helper (read from head & write to tail)
/************************************************/
#define LIST_GET_HEAD(lhead, ltail, en) {\
    en = lhead; \
    lhead = lhead->next; \
    if (lhead) lhead->prev = NULL; \
    else ltail = NULL; \
    en->next = NULL; }

#define LIST_PUT_TAIL(lhead, ltail, en) {\
    en->next = NULL; \
    en->prev = NULL; \
    if (ltail) { en->prev = ltail; ltail->next = en; ltail = en; } \
    else { lhead = en; ltail = en; }}

#define LIST_INSERT_BEFORE(entry, newentry) { \
    newentry->next = entry; \
    newentry->prev = entry->prev; \
    if (entry->prev) entry->prev->next = newentry; \
    entry->prev = newentry; }

#define LIST_REMOVE(entry) { \
    if (entry->next) entry->next->prev = entry->prev; \
    if (entry->prev) entry->prev->next = entry->next; }

#define LIST_REMOVE_HT(entry, head, tail) { \
    if (entry->next) entry->next->prev = entry->prev; \
    else { assert(entry == tail); tail = entry->prev; } \
    if (entry->prev) entry->prev->next = entry->next; \
    else { assert(entry == head); head = entry->next; } }

/************************************************/
// STACK helper (push & pop)
/************************************************/
#define STACK_POP(stack, top) { \
    if (stack == NULL) top = NULL; \
    else { top = stack; stack = stack->next; } }

#define STACK_PUSH(stack, entry) {\
    entry->next = stack; stack = entry; }

const float eps = 1e-8; // floating point error.
#define LOCK_EX 0
#define LOCK_SH 1
#define LOCK_NONE 2
#define LOW_TS_BITS 50
#define LOW_TS_MASK ((1ULL<<LOW_TS_BITS)-1)
#define IS_PRIORI(x, y) ((x) >= (y) + eps)
#define IS_PRIORI_OR_EQ(x, y) ((x) >= (y) + eps || (fabs(x - y) < eps))
#define LOOP_TIME_OUT (1000 * 1000) // microseconds

struct LockEntry {
    uint16_t type;
    uint64_t tid;
    WaitPriority rank;
    volatile bool lock_ready;
    xact* locked_xact;
    LockEntry *next;
    LockEntry *prev;

    LockEntry();
};

class RWLock {
public:
    RWLock();
    bool lockEmpty(uint64_t tid) const;
    bool lockNotModified(uint64_t tid) const;
    bool lockW(uint64_t tid, xact* xact, bool not_sorted = false);
    void unlockW(uint64_t tid);
    bool lockR(uint64_t tid, xact* xact, bool not_sorted = false);
    void unlockR(uint64_t tid);
    bool lock_get(uint16_t type, uint64_t tid, bool not_sorted, xact* xact);
    void lock_release(uint64_t tid);
    void promote();
    LockEntry* pop_best();  // pop out the transaction with optimal priority.
    void put_waiter(LockEntry *entry);

private:
    pthread_mutex_t *latch;
    uint16_t lock_type;
    uint32_t owner_cnt;
    bool sorted;
    LockEntry *owners;
    LockEntry *waiters_head;
    LockEntry *waiters_tail;

    bool conflict_lock(uint16_t l1, uint16_t l2);
    LockEntry* get_entry();
    void return_entry(LockEntry* entry);
    void check_correctness();
};

#endif // _RWLOCK_H_
