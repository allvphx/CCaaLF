#ifndef _ABSTRACT_DB_H_
#define _ABSTRACT_DB_H_

#include <stddef.h>
#include <stdint.h>
#include <sys/types.h>

#include <map>
#include <string>

#include "abstract_ordered_index.h"
#include "../str_arena.h"
#include "../conflict_graph.h"
#include "../policy.h"

/**
 * Abstract interface for a DB. This is to facilitate writing
 * benchmarks for different systems, making each system present
 * a unified interface
 */
class abstract_db {
public:
 Policy *pg;

  /**
   * both get() and put() can throw abstract_abort_exception. If thrown,
   * abort_txn() must be called (calling commit_txn() will result in undefined
   * behavior).  Also if thrown, subsequently calling get()/put() will also
   * result in undefined behavior)
   */
  class abstract_abort_exception {};

  // ctor should open db
  abstract_db(Policy *p = nullptr) {
    pg = p;
  }

  // dtor should close db
  virtual ~abstract_db() {}

  /**
   * an approximate max batch size for updates in a transaction.
   *
   * A return value of -1 indicates no maximum
   */
  virtual ssize_t txn_max_batch_size() const { return -1; }

  virtual bool index_has_stable_put_memory() const { return false; }

  // XXX(stephentu): laziness
  virtual size_t
  sizeof_txn_object(uint64_t txn_flags) const { NDB_UNIMPLEMENTED("sizeof_txn_object"); };

  /**
   * XXX(stephentu): hack
   */
  virtual void do_txn_epoch_sync() const {}

  /**
   * XXX(stephentu): hack
   */
  virtual void do_txn_finish() const {}

  /** loader should be used as a performance hint, not for correctness */
  virtual void thread_init(bool loader) {}

  virtual void thread_end() {}

  // [ntxns_persisted, ntxns_committed, avg latency]
  virtual std::tuple<uint64_t, uint64_t, double>
    get_ntxn_persisted() const { return std::make_tuple(0, 0, 0.0); }

  virtual void reset_ntxn_persisted() { }

  enum TxnProfileHint {
    HINT_DEFAULT,

    // consistency check
    HINT_CONSISTENCY_CHECK,

    // ycsb profiles
    HINT_KV_GET_PUT, // KV workloads over a single key
    HINT_KV_RMW, // get/put over a single key
    HINT_KV_SCAN, // KV scan workloads (~100 keys)

    // tpcc profiles
    HINT_TPCC_NEW_ORDER,
    HINT_TPCC_PAYMENT,
    HINT_TPCC_DELIVERY,
    HINT_TPCC_ORDER_STATUS,
    HINT_TPCC_ORDER_STATUS_READ_ONLY,
    HINT_TPCC_STOCK_LEVEL,
    HINT_TPCC_STOCK_LEVEL_READ_ONLY,

    // micro profiles
    HINT_MICRO,
    HINT_MICRO_LOADER,
    
    // tpce
    HINT_TPCE_BROKER_VOLUME,
    HINT_TPCE_CUSTOMER_POSITION,
    HINT_TPCE_MARKET_FEED,
    HINT_TPCE_MARKET_WATCH,
    HINT_TPCE_SECURITY_DETAIL,
    HINT_TPCE_TRADE_LOOKUP,
    HINT_TPCE_TRADE_ORDER,
    HINT_TPCE_TRADE_RESULT,
    HINT_TPCE_TRADE_STATUS,
    HINT_TPCE_TRADE_UPDATE,

    HINT_SEATS_DEFAULT,
    HINT_SEATS_READ_ONLY_DEFAULT,
  };

  /**
   * Initializes a new txn object the space pointed to by buf
   *
   * Flags is only for the ndb protocol for now
   *
   * [buf, buf + sizeof_txn_object(txn_flags)) is a valid ptr
   */
  virtual void *new_txn(
      uint64_t txn_flags,
      str_arena &arena,
      void *buf,
      TxnProfileHint hint = HINT_DEFAULT,
      uint8_t txn_type = 0) = 0;

  typedef std::map<std::string, uint64_t> counter_map;
  typedef std::map<std::string, counter_map> txn_counter_map;

  virtual void init_txn(
    void *txn,
    conflict_graph* cg,
    uint8_t type,
    Policy *p = nullptr){}

  virtual uint64_t get_tid(void *txn) {return 0;}

  /**
   * Reports things like read/write set sizes
   */
  virtual counter_map
  get_txn_counters(void *txn) const
  {
    return counter_map();
  }

  /**
   * Returns true on successful commit.
   *
   * On failure, can either throw abstract_abort_exception, or
   * return false- caller should be prepared to deal with both cases
   */
  virtual bool commit_txn(void *txn) = 0;

  virtual void one_op_begin(void *txn) {}
  virtual bool one_op_end(void *txn) { return false; }
  virtual void mul_ops_begin(void *txn) {}
  virtual bool mul_ops_end(void *txn) { return false; }
  virtual bool atomic_ops_abort(void *txn) { return false; }

  virtual bool should_abort(void *txn) {return false;}

  virtual std::pair<bool, uint32_t> expose_uncommitted(void *txn, uint32_t acc_id = MAX_ACC_ID) {};

  virtual void set_failed_records(void *txn, std::vector<void *>& records) {};
  virtual std::vector<void *> *get_failed_records(void *txn) {};

  virtual uint16_t get_txn_contention(void *txn) {};

  /**
   * XXX
   */
  virtual void abort_txn(void *txn) = 0;

  virtual void print_txn_debug(void *txn) const {}

  virtual abstract_ordered_index *
  open_index(const std::string &name,
             size_t value_size_hint,
             bool mostly_append = false) = 0;

  virtual void
  close_index(abstract_ordered_index *idx) = 0;

  //Only debug
  virtual uint32_t
  get_write_number(void* txn) {return 0; }
};

#endif /* _ABSTRACT_DB_H_ */
