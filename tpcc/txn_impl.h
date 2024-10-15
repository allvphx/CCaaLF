#ifndef _NDB_TXN_IMPL_H_
#define _NDB_TXN_IMPL_H_

// XXX(Conrad): Use for double locking bug
#include <map>

#include "txn.h"
#include "lockguard.h"
#include "piece_impl.h"
#include "learn.h"

// base definitions

static bool sync_after_exec = true;

template <template <typename> class Protocol, typename Traits>
transaction<Protocol, Traits>::transaction(uint64_t flags, string_allocator_type &sa, uint8_t tx_type)
  : transaction_base(flags), sa(&sa), pnum(0), mix_op(this), tid(cast()->MakeTid(coreid::core_id(), coreid::core_count(), 0)), txn_type(tx_type)
//  : transaction_base(flags), sa(&sa), pnum(0), piece(this), mix_op(this)
{

  INVARIANT(rcu::s_instance.in_rcu_region());

//  tid = cast()->MakeTid(coreid::core_id(), coreid::core_count(), 0);
#ifdef COUNT_TX_BLOCK
  global_listener.tx_n_pending ++;
#endif

  memory_barrier();
  //Ad-hoc sync: clear the cur_step after reset tid
  //After reading cur_step, tid will be checked
  cur_step = 0;

  read_set_start = 0;
  write_set_start = 0;
  // insert_set_start = 0;
  dep_queue_start = 0;

  read_access_start = 0;
  write_access_start = 0;

  txn_first_acc_id = MAX_ACC_ID;
  ongoing_acc_id = MAX_ACC_ID;
  valid_acc_id = MAX_ACC_ID;
  last_commit_piece_acc_id = MAX_ACC_ID;

  rw_highest_contention = 0;
  piece_contention = 0;

  // check and tune this size
  clean_read_failed_tuples.reserve(4);

#ifdef BTREE_LOCK_OWNERSHIP_CHECKING
  concurrent_btree::NodeLockRegionBegin();
#endif
}

template <template <typename> class Protocol, typename Traits>
transaction<Protocol, Traits>::~transaction()
{
  // transaction shouldn't fall out of scope w/o resolution
  // resolution means TXN_EMBRYO, TXN_COMMITED, and TXN_ABRT
  // safe_gc();
  INVARIANT(state != TXN_ACTIVE);
  INVARIANT(rcu::s_instance.in_rcu_region());
  const unsigned cur_depth = rcu_guard_->sync()->depth();
  rcu_guard_.destroy();
  global_listener.tx_n_pending --;
  if (cur_depth == 1) {
    INVARIANT(!rcu::s_instance.in_rcu_region());
    cast()->on_post_rcu_region_completion();
  }
#ifdef BTREE_LOCK_OWNERSHIP_CHECKING
  concurrent_btree::AssertAllNodeLocksReleased();
#endif
}

template <template <typename> class Protocol, typename Traits>
void
transaction<Protocol, Traits>::safe_gc()
{
  // typename read_set_map::iterator it     = read_set.begin();
  // typename read_set_map::iterator it_end = read_set.end();
  // for (; it != it_end; ++it) {
  //     access_entry* e = it->get_access_entry();
  //     while(e && e->has_reader())
  //       nop_pause();
  // }
}

template <template <typename> class Protocol, typename Traits>
void
transaction<Protocol, Traits>::clear()
{
  // not used in chamcc
  ALWAYS_ASSERT(false);
}

template <template <typename> class Protocol, typename Traits>
void
transaction<Protocol, Traits>::abort_impl(abort_reason reason)
{
  bool lock_mode = true;
  chamcc_abort_impl(nullptr, lock_mode);
}

template <template <typename> class Protocol, typename Traits>
void
transaction<Protocol, Traits>::chamcc_abort_impl(dbtuple_write_info_vec *write_dbtuples, bool lock_mode)
{
  // This function is responsible for 2 tasks:
  // 1. clean up unused insert tuple markers
  // 2. unlink all the linked access entries on tuples

  // TODO - refactor this part, such a mess

  if (write_dbtuples != nullptr) {
    // reach here with write_dbtuples != nullptr means we are aborted from failed commit process
    // we may already holding some locks
    typename dbtuple_write_info_vec::iterator it = write_dbtuples->begin();
    typename dbtuple_write_info_vec::iterator it_end = write_dbtuples->end();
    dbtuple_write_info *last_px = nullptr;
    bool inserted_last_run = false;
    for (; it != it_end; last_px = &(*it), ++it) {
      if (likely(last_px && last_px->tuple != it->tuple)) {
        dbtuple *tuple = last_px->get_tuple();
        // access_entry *e = last_px->entry->get_access_entry();
        // on boundary
        if (inserted_last_run) { // last group contains insertion, we have to clean it up
          if (!last_px->is_locked()) {// if it's not locked we have to locked it first
            // deadlock is impossible. Assertion can be made that all locked has been released
            tuple->chamcc_lock(lock_mode, nullptr, true);
          }

          this->cleanup_inserted_tuple_marker(
              tuple, last_px->entry->get_key(), last_px->entry->get_btree());

          tuple->chamcc_unlock(lock_mode, nullptr);

          inserted_last_run = false;

          if (it->is_insert())
            inserted_last_run = true;

          continue;
        }

        if (last_px->is_locked()) {
          tuple->chamcc_unlock(lock_mode, nullptr);
        }

        inserted_last_run = false;
      }

      if (it->is_insert())
        inserted_last_run = true;
    }

    if (likely(last_px)) {
      dbtuple *tuple = last_px->get_tuple();
      // access_entry *e = last_px->entry->get_access_entry();
      // on boundary
      if (inserted_last_run) { // last group contains insertion, we have to clean it up
        if (!last_px->is_locked()) {// if it's not locked we have to locked it first
          // deadlock is impossible. Assertion can be made that all locked has been released
          tuple->chamcc_lock(lock_mode, nullptr, true);
        }

        this->cleanup_inserted_tuple_marker(
            tuple, last_px->entry->get_key(), last_px->entry->get_btree());

        tuple->chamcc_unlock(lock_mode, nullptr);
      } else if (last_px->is_locked()) {
        tuple->chamcc_unlock(lock_mode, nullptr);
      }
    }
  }

  // TODO - do entry unlink along with cleanup_inserted_tuple_marker
  // TODO - will save one extra round of locking
  // TODO - challenge - how to determine whether a it is locked or not
  if (!write_set.empty()) {

    // reach here with write_dbtuples == nullptr means we are aborted from user instruction
    // we are not holding any locks
    typename write_set_map::iterator it = write_set.begin();
    typename write_set_map::iterator it_end = write_set.end();

    for (; it != it_end; ++it) {

      access_entry *e = it->get_access_entry();
      dbtuple *tuple = it->get_tuple();

      if (!e) continue;

      dbtuple *target = static_cast<dbtuple *>(e->get_commit_target());
      if (target != nullptr && target != tuple)
        tuple = target;

      tuple->chamcc_lock(lock_mode, nullptr);


      // unlink the write entry
      if (e->is_linked()) {
        tuple->unlink_entry(e);
      }
      // unlink the related read entry (if exists)
      access_entry *read_e = it->get_read_access_entry();
      if (read_e && read_e->is_linked()) {
        tuple->unlink_entry(read_e);
      }
      tuple->chamcc_unlock(lock_mode, nullptr);
      // tuple->decrease_counter();
    }
  }

  // TODO - duplicate code with commit() phase5
  // Unlink all the other entries(read access entries with no overwritten)
  if (!read_set.empty()) {

    typename read_set_map::iterator it     = read_set.begin();
    typename read_set_map::iterator it_end = read_set.end();
    for (; it != it_end; ++it) {
      if (it->is_occ()) continue;

      access_entry* e = it->get_access_entry();

      if (!e) continue;

      // if (!it->is_occ() && !e->is_overwrite()) it->get_tuple()->decrease_counter();

      if(e->is_linked()) {

        INVARIANT(!e->is_overwrite());
        dbtuple* tuple = static_cast<dbtuple *>(e->get_commit_target());

        tuple->chamcc_lock(lock_mode, nullptr);

        dbtuple* target = static_cast<dbtuple *>(e->get_commit_target());
        if(target != nullptr && target != tuple)
          tuple = target;

        if(e->is_linked()) {
          e->set_doomed();
          tuple->unlink_entry(e);
        }

        tuple->chamcc_unlock(lock_mode, nullptr);
      }
    }
  }

  state = TXN_ABRT;
}

template <template <typename> class Protocol, typename Traits>
void
transaction<Protocol, Traits>::cleanup_inserted_tuple_marker(
    dbtuple *marker, const std::string &key, concurrent_btree *btr)
{
  //fprintf(stderr, "[cleanup_inserted_tuple_marker] remove key %ld\n", varkey(key).slice());
  // XXX: this code should really live in txn_ic3_impl.h
//  INVARIANT(marker->version == dbtuple::MAX_TID);
  INVARIANT(marker->is_locked());
  INVARIANT(marker->is_lock_owner());

  if (marker->version != dbtuple::MAX_TID
      || !marker->is_latest()) // patch(zhouz)
    return; // the tuple has been over-written by others

  typename concurrent_btree::value_type removed = 0;
  const bool did_remove = btr->remove(varkey(key), &removed);
  if (unlikely(!did_remove)) {
#ifdef CHECK_INVARIANTS
    std::cerr << " *** could not remove key: " << util::hexify(key)  << std::endl;
#ifdef TUPLE_CHECK_KEY
    std::cerr << " *** original key        : " << util::hexify(marker->key) << std::endl;
#endif
#endif
    INVARIANT(false);
  }
  INVARIANT(removed == (typename concurrent_btree::value_type) marker);
  INVARIANT(marker->is_latest());
  marker->clear_latest();
  dbtuple::release(marker); // rcu free
}

namespace {
  inline const char *
  transaction_state_to_cstr(transaction_base::txn_state state)
  {
    switch (state) {
    case transaction_base::TXN_EMBRYO: return "TXN_EMBRYO";
    case transaction_base::TXN_ACTIVE: return "TXN_ACTIVE";
    case transaction_base::TXN_FINISH_EXEC: return "TXN_FINISH_EXEC";
    case transaction_base::TXN_ABRT: return "TXN_ABRT";
    case transaction_base::TXN_COMMITED: return "TXN_COMMITED";
    }
    ALWAYS_ASSERT(false);
    return 0;
  }

  inline std::string
  transaction_flags_to_str(uint64_t flags)
  {
    bool first = true;
    std::ostringstream oss;
    if (flags & transaction_base::TXN_FLAG_LOW_LEVEL_SCAN) {
      oss << "TXN_FLAG_LOW_LEVEL_SCAN";
      first = false;
    }
    if (flags & transaction_base::TXN_FLAG_READ_ONLY) {
      if (first)
        oss << "TXN_FLAG_READ_ONLY";
      else
        oss << " | TXN_FLAG_READ_ONLY";
      first = false;
    }
    return oss.str();
  }
}

template <template <typename> class Protocol, typename Traits>
void
transaction<Protocol, Traits>::dump_debug_info() const
{
  std::cerr << "Transaction (obj=" << util::hexify(this) << ") -- state "
       << transaction_state_to_cstr(state) << std::endl;
  std::cerr << "  Abort Reason: " << AbortReasonStr(reason) << std::endl;
  std::cerr << "  Flags: " << transaction_flags_to_str(flags) << std::endl;
  std::cerr << "  Read set size: " << read_set.size()
            << "  Write set size: " << write_set.size()
            << "  Absent set size: " << absent_set.size()
            << std::endl;
  std::cerr << "  Read/Write sets:" << std::endl;

  std::cerr << "      === Read Set ===" << std::endl;
  // read-set
  for (typename read_set_map::const_iterator rs_it = read_set.begin();
       rs_it != read_set.end(); ++rs_it)
    std::cerr << *rs_it << std::endl;

  std::cerr << "      === Write Set ===" << std::endl;
  // write-set
  for (typename write_set_map::const_iterator ws_it = write_set.begin();
       ws_it != write_set.end(); ++ws_it)
    std::cerr << *ws_it << std::endl;

  std::cerr << "      === Absent Set ===" << std::endl;
  // absent-set
  for (typename absent_set_map::const_iterator as_it = absent_set.begin();
       as_it != absent_set.end(); ++as_it)
    std::cerr << "      B-tree Node " << util::hexify(as_it->first)
              << " : " << as_it->second << std::endl;

}

template <template <typename> class Protocol, typename Traits>
std::map<std::string, uint64_t>
transaction<Protocol, Traits>::get_txn_counters() const
{
  std::map<std::string, uint64_t> ret;

  // max_read_set_size
  ret["read_set_size"] = read_set.size();;
  ret["read_set_is_large?"] = !read_set.is_small_type();

  // max_absent_set_size
  ret["absent_set_size"] = absent_set.size();
  ret["absent_set_is_large?"] = !absent_set.is_small_type();

  // max_write_set_size
  ret["write_set_size"] = write_set.size();
  ret["write_set_is_large?"] = !write_set.is_small_type();

  return ret;
}

template <template <typename> class Protocol, typename Traits>
bool
transaction<Protocol, Traits>::handle_last_tuple_in_group(
    dbtuple_write_info &last,
    bool did_group_insert,
    bool lock_mode)
{
  // No different handling is required for insertion, so `did_group_insert` is unused.
  // both insert and update operations, lock is grabbed here
  // Note: about abort
  // When aborted inserted tuple has to be removed. However, this is conditional:
  // the tuple has to be locked in the first place. Then checking whether its
  // version == MAX_TID. Only in that case the tuple is removed from underlying tree
  // and clear its latest marker.
  // So here, if another transaction lock a tuple first, then the clear process sees
  // version != MAX_TID and don't remove it. Otherwise the clear process takes first,
  // the handle_last_tuple_in_group fails and signal abort.
  dbtuple *tuple = last.get_tuple();

  // access_entry* e = last.entry->get_access_entry();
  const dbtuple::version_t v = tuple->chamcc_lock(lock_mode, nullptr, true);

  INVARIANT(dbtuple::IsLatest(v) == tuple->is_latest());
  last.mark_locked();
  if (unlikely(!dbtuple::IsLatest(v) ||
               !cast()->can_read_tid(tuple->version))) {
    return false; // signal abort
  }
  last.entry->set_do_write();
  return true;
}


template <template <typename> class Protocol, typename Traits>
template <typename ValueReader>
bool
transaction<Protocol, Traits>::do_snapshot_get(concurrent_btree &btr,
                                  const std::string* key_str, ValueReader& value_reader)
{
  typename concurrent_btree::value_type underlying_v{};
  concurrent_btree::versioned_node_t search_info;
  const bool found = btr.search(varkey(*key_str), underlying_v, &search_info);
  if (found) {
    const dbtuple * const tuple = reinterpret_cast<const dbtuple *>(underlying_v);
    return do_tuple_read(tuple, value_reader);
  } else {
    return false;
  }
}


template <template <typename> class Protocol, typename Traits>
template <typename ValueReader>
bool
transaction<Protocol, Traits>::do_get(concurrent_btree &btr,
                                  const std::string* key_str, ValueReader& value_reader,
                                  uint32_t acc_id)
{
  if(is_snapshot()) {
    return do_snapshot_get(btr, key_str, value_reader);
  } else {
    return mix_op.do_get(btr, key_str, value_reader, acc_id);
  }
}


template <template <typename> class Protocol, typename Traits>
template <typename ValueReader>
bool
transaction<Protocol, Traits>::profile_do_get(concurrent_btree &btr,
                                  const std::string* key_str, ValueReader& value_reader, ic3_profile* prof)
{
  if(is_snapshot())
    return do_snapshot_get(btr, key_str, value_reader);

#if 1
  // not used in chamcc
  ALWAYS_ASSERT(false);
#else
  return piece.profile_do_get(btr, key_str, value_reader, prof);
#endif
}

template <template <typename> class Protocol, typename Traits>
template <typename ValueWriter>
bool
transaction<Protocol, Traits>::do_put(concurrent_btree &btr,
                                  const std::string* key_str, void* val_str,
                                  ValueWriter& value_writer, bool expect_new, uint32_t acc_id)
{
  return mix_op.do_put(btr, key_str, val_str, value_writer, expect_new, acc_id);
}

template <template <typename> class Protocol, typename Traits>
template <typename ValueReader, typename ValueWriter>
bool
transaction<Protocol, Traits>::do_update(concurrent_btree &btr, const std::string *key_str,
                                         ValueReader &value_reader, update_callback *callback,
                                         ValueWriter &value_writer, uint32_t acc_id)
{
  return mix_op.do_update(btr, key_str, value_reader, callback, value_writer, acc_id);
}

template <template <typename> class Protocol, typename Traits>
bool
transaction<Protocol, Traits>::handle_last_tuple_in_group_expose_piece(
    dbtuple_read_write_info &last,
    bool is_write,
    bool lock_mode)
{
  dbtuple *tuple = last.get_tuple();

  // access_entry* e = last.get_access_entry();
  VERBOSE(std::cerr << "locking - addr: " << last.mcslock << " is_write_op: " << is_write << std::endl);
  const dbtuple::version_t v = tuple->chamcc_lock(lock_mode, nullptr);
  last.mark_locked();
  
  return true;
}

// template <template <typename> class Protocol, typename Traits>
// uint16_t
// transaction<Protocol, Traits>::get_unexposed_set_contention() {
//   uint16_t set_contention = 0;
//   for(int i = write_set_start; i < write_set.size(); ++i) {
//     uint16_t record_contention = write_set[i].get_contention();
//     if (record_contention > set_contention) set_contention = record_contention;
//   }
//   for(int i = read_set_start; i < read_set.size(); ++i) {
//     if (read_set[i].is_occ()) continue;
//     uint16_t record_contention = read_set[i].get_contention();
//     if (record_contention > set_contention) set_contention = record_contention;
//   }
//   return set_contention;
// }

template <template <typename> class Protocol, typename Traits>
std::pair<bool, uint32_t>
transaction<Protocol, Traits>::expose_uncommitted(uint32_t acc_id) {

  auto pa = refresh_policy(acc_id, OpCommit);
  // check whether should do the commit piece
  auto tmp = acc_id-ACCESSES;
  if (!can_expose[tmp] || !inference_commit_piece(pa))
    return std::pair<bool, uint32_t>(true, 0);
  // there is no reason for us to expose the final operation w/o wait commit.
  assert(cur_step == tmp - txn_first_acc_id + 1);
  before_commit_piece_operation(pa, tmp - txn_first_acc_id + 1 == txn_access_num[txn_type]);

  bool lock_mode = true;

  // ------------- prepare access entry -------------
  uint16_t dirty_read_idx[traits_type::read_set_expected_size] = {}; 
  uint16_t dirty_read_count = 0;

  for(int i = read_set_start; i < read_set.size(); i++) {
    if(read_set[i].get_access_entry() == nullptr) {
      read_set[i].set_access_entry(insert_read_access_entry(dbtuple::MIN_TID, get_tid(), this));
    }
    // all reads need validation
    dirty_read_idx[dirty_read_count++] = i;
  }

  for(int i = write_set_start; i < write_set.size(); i++) {
    if(write_set[i].get_access_entry() == nullptr) {
      write_set[i].set_access_entry(insert_write_access_entry(dbtuple::MIN_TID, get_tid(), this));
    }
  }

  // ------------- locking -------------
  dbtuple_read_write_info_vec rw_dbtuples;
  for(int i = write_set_start; i < write_set.size(); ++i) {
    write_record_t *record = &write_set[i];
    rw_dbtuples.emplace_back(record->get_tuple(), &(*record), nullptr, true/*is_write*/, i);
  }
  if (dirty_read_count != 0) {
    for(int i = 0; i < dirty_read_count; i++) {
      read_record_t *record = &read_set[dirty_read_idx[i]];
      rw_dbtuples.emplace_back(record->get_tuple(), nullptr, &(*record), false/*is_write*/, dirty_read_idx[i]);
    }
  }

  if (!rw_dbtuples.empty()) {
    // sort order
    rw_dbtuples.sort();
    // lock the tuples in order
    typename dbtuple_read_write_info_vec::iterator it     = rw_dbtuples.begin();
    typename dbtuple_read_write_info_vec::iterator it_end = rw_dbtuples.end();
    dbtuple_read_write_info *last_px = nullptr;
    bool write_intention_last_run = false;
    for (; it != it_end; last_px = &(*it), ++it) {
      if (likely(last_px && last_px->tuple != it->tuple)) {
        // on boundary
        handle_last_tuple_in_group_expose_piece(*last_px, write_intention_last_run, lock_mode);
        write_intention_last_run = false;
      }
      if (it->is_write())
        write_intention_last_run = true;
    }
    if (likely(last_px))
      handle_last_tuple_in_group_expose_piece(*last_px, write_intention_last_run, lock_mode);
  }

  // ------------- exposing read/write set -------------
  for(int i = write_set_start; i < write_set.size(); ++i) {
    mix_op.check_rmw(i);
  }

  bool early_abort_flag = false;

  // expose read set
  if (likely(!early_abort_flag && dirty_read_count != 0)) {
    for(int i = 0; i < dirty_read_count; i++) {
      if (!mix_op.do_expose_ic3_action(dirty_read_idx[i], false /*is_write*/, lock_mode)) {
        early_abort_flag = true;
        break;
      }
    }
  }
  // expose write set
  if (likely(!early_abort_flag)) {
    for(int i = write_set_start; i < write_set.size(); ++i) {
      if (!mix_op.do_expose_ic3_action(i, true /*is_write*/, lock_mode)) {
        early_abort_flag = true;
        break;
      }
    }
  }

  // ------------- unlocking -------------
  if (!rw_dbtuples.empty()) {
    // lock the tuples in order
    typename dbtuple_read_write_info_vec::iterator it     = rw_dbtuples.begin();
    typename dbtuple_read_write_info_vec::iterator it_end = rw_dbtuples.end();
    for (; it != it_end; ++it) {
      VERBOSE(std::cerr << "unlocking - addr: " << it->mcslock 
                        << " is_write: " << it->is_write() 
                        << " is_locked: " << it->is_locked() 
                        << std::endl);
      if (it->is_locked()) {
        // access_entry *entry = it->get_access_entry();
        dbtuple *tuple = it->get_tuple();
        tuple->chamcc_unlock(lock_mode, nullptr);
      }
    }
  }

  // ------------- post exposing (undo exposing or store success info) -------------
  if (early_abort_flag) {
    // clear this piece, prepare for retry
    // all already exposed entries should be rolled back
    if (!read_set.empty()) {
      typename read_set_map::iterator it     = read_set.begin() + read_set_start;
      typename read_set_map::iterator it_end = read_set.end();
      for (; it != it_end; ++it) {
        access_entry *entry = it->get_access_entry();
        dbtuple *tuple = it->get_tuple();
        if(entry && entry->is_linked()) {
          tuple->chamcc_lock(lock_mode, nullptr);
          dbtuple* target = static_cast<dbtuple *>(entry->get_commit_target());
          if(target != nullptr && target != tuple)
            tuple = target;
          tuple->unlink_entry(entry);
          tuple->chamcc_unlock(lock_mode, nullptr);
        }
        // if (!it->is_occ() && !entry->is_overwrite()) tuple->decrease_counter();
      }
    }
    if (!write_set.empty()) {
      typename write_set_map::iterator it     = write_set.begin() + write_set_start;
      typename write_set_map::iterator it_end = write_set.end();
      for (; it != it_end; ++it) {
        access_entry *entry = it->get_access_entry();
        dbtuple *tuple = it->get_tuple();
        if(entry && entry->is_linked()) {
          tuple->chamcc_lock(lock_mode, nullptr);
          dbtuple* target = static_cast<dbtuple *>(entry->get_commit_target());
          if(target != nullptr && target != tuple)
            tuple = target;
          tuple->unlink_entry(entry);
          tuple->chamcc_unlock(lock_mode, nullptr);
        }
        // tuple->decrease_counter();
      }
    }
  } else {
    // piece commit successfully, store the valid read/write set position
    read_set_start = read_set.size();
    read_access_start = read_access_map.size();
    write_set_start = write_set.size();
    write_access_start = write_access_map.size();
    dep_queue_start = dep_queue.size();
    // insert_set_start = insert_set.size();
    valid_absent_set = absent_set;
    valid_acc_id = ongoing_acc_id;
    last_commit_piece_acc_id = acc_id;
  }

  // // ------------- clear buffer -------------
  // clear_unexposed_operations();
  piece_contention = 0;

  // ------------- retry piece -------------
  if (early_abort_flag) {
    read_set.shrink(read_set_start);
    read_access_map.shrink(read_access_start);
    write_set.shrink(write_set_start);
    write_access_map.shrink(write_access_start);
    dep_queue.shrink(dep_queue_start);
    feature->tx_n_dep_on = dep_queue_start;
    // insert_set.erase(insert_set.begin() + insert_set_start, insert_set.end());
    absent_set = valid_absent_set;
    ongoing_acc_id = valid_acc_id;
    // todo - normally, we should also reset the absent_set here
    // or there could be false abort in mix_op::do_node_read() and mix_op::try_insert_new_tuple()

    return std::pair<bool, uint32_t>(false, get_retry_point());
  }

  // ------------- piece successfully done -------------
  return std::pair<bool, uint32_t>(true, 0);
}


//Acquire all tuples' locks in read set since rs_start
template <template <typename> class Protocol, typename Traits>
void
transaction<Protocol, Traits>::acquire_rs_locks(uint32_t rs_start)
{
#if 1
  // not used in chamcc
  ALWAYS_ASSERT(false);
#else
  dbtuple_read_info_vec all_dbtuples;

  for(int i = rs_start; i < read_set.size(); i++) {
    // XXX(Conrad): Given OCC reads skip dependency stuff in piece_impl.h
    // anyway, should not lock them here (and if you do, they must be unlocked
    // in atomic_mul_ops<Transaction>::commit_piece).
    if (read_set[i].is_occ())
      continue;
    dbtuple *tuple =  const_cast<dbtuple*>(read_set[i].get_tuple());
    all_dbtuples.emplace_back(tuple, &read_set[i], read_set[i].is_insert(), i);
  }

  if(all_dbtuples.empty())
    return;

  all_dbtuples.sort();

  typename dbtuple_read_info_vec::iterator it     = all_dbtuples.begin();
  typename dbtuple_read_info_vec::iterator it_end = all_dbtuples.end();

  dbtuple_read_info *last_px = nullptr;

  for (; it != it_end; last_px = &(*it), ++it) {
    if(last_px && it->tuple == last_px->tuple)
      continue;

    if(!it->is_insert()) {
      dbtuple *tuple =  const_cast<dbtuple*>(it->get_tuple());
      //fprintf(stderr, "core[%d] try to aquire [%lx] lock\n", coreid::core_id(), tuple);
#ifdef USE_MCS_LOCK
      access_entry* e = it->entry->get_access_entry();
      tuple->mcs_lock(&e->m_node);
#else
      tuple->lock();
#endif

    }

  }
#endif
}

template <template <typename> class Protocol, typename Traits>
bool
transaction<Protocol, Traits>::commit(bool doThrow)
{

  if(is_snapshot()) {
    state = TXN_COMMITED;
    return true;
  }

  bool lock_mode = true;

  // XXX(Jiachen): inherit from OCC implementation, using dbtuple_write_info_vec for sorting
  dbtuple_write_info_vec write_dbtuples;
  std::pair<bool, tid_t> commit_tid(false, 0);

  VERBOSE(std::cerr << "Txn Info:"
            << "read_set size - " << read_set.size()
            << "write_set size - " << write_set.size()
            << "absent_set size - " << absent_set.size()
            << std::endl);

  // ------------- add all dirty reads into dep list -------------
  for(int i = read_set_start; i < read_set.size(); i++) {
    if (read_set[i].is_occ()) continue;

    if(read_set[i].get_access_entry() == nullptr) {
      read_set[i].set_access_entry(insert_read_access_entry(dbtuple::MIN_TID, get_tid(), this));
    }
    access_entry *entry = read_set[i].get_access_entry();
    dbtuple *tuple = read_set[i].get_tuple();
    ALWAYS_ASSERT(!read_set[i].is_occ());
    ALWAYS_ASSERT(entry && tuple);
    ALWAYS_ASSERT(!entry->is_linked());
    ALWAYS_ASSERT(!entry->is_prepare());

    access_entry *de = tuple->get_dependent_entry(entry->is_write());
    if (likely(de)) {
      transaction<Protocol, Traits>* d_txn = (transaction<Protocol, Traits>*) de->get_txn();
      uint64_t d_tid = de->get_tid();

      if (!d_txn->is_commit(d_tid)
          && d_tid != get_tid()
          && d_txn != this) {
        add_dep_txn(de->get_tid(), de->get_txn(), true/*dirty_read_dep*/);
      }
    }
  }

  //Phase1. Wait txns in dependency queue
  if (!dep_queue.empty()){
    uint64_t start_time = util::timer::cur_usec();
    typename dep_queue_map::iterator it = dep_queue.begin();
#ifdef COUNT_TX_BLOCK
    global_listener.tx_n_blocked ++;
#endif

    for (; it != dep_queue.end(); ++it) {
      transaction<Protocol, Traits>* d_txn = (transaction<Protocol, Traits>*)it->txn;
#ifdef COUNT_TX_BLOCK
      d_txn->txn_current_blocking ++;
#endif

      while(!(d_txn->is_commit(it->tid) || d_txn->is_abort(it->tid))) {
        memory_barrier();
        nop_pause();

        // check timeout
        if (likely(TIME_OUT != 0) && unlikely((util::timer::cur_usec() - start_time) > TIME_OUT)) {
#ifdef COUNT_TX_BLOCK
          d_txn->txn_current_blocking --;
          global_listener.tx_n_blocked --;
#endif
          abort_trap((reason = ABORT_REASON_TIMEOUT));
          goto do_abort;
        }
      }
      if (d_txn->is_abort(it->tid) && it->dirty_read_dep) {
#ifdef COUNT_TX_BLOCK
        d_txn->txn_current_blocking --;
        global_listener.tx_n_blocked --;
#endif
        VERBOSE(std::cerr << "Cascading abort happen sue to txn id -  " << d_txn->tid << std::endl);
        abort_trap((reason = ABORT_REASON_CASCADING));
        goto do_abort;
      }
#ifdef COUNT_TX_BLOCK
      d_txn->txn_current_blocking --;
#endif
    }

#ifdef COUNT_TX_BLOCK
    global_listener.tx_n_blocked --;
#endif
  }

  state = TXN_FINISH_EXEC;


  //Phase2. Lock write set
  // copy write tuples to vector for sorting
  if (!write_set.empty()) {
    PERF_DECL(
        static std::string probe1_name(
        std::string(__PRETTY_FUNCTION__) + std::string(":lock_write_nodes:")));
    ANON_REGION(probe1_name.c_str(), &transaction_base::g_txn_commit_probe1_cg);
    INVARIANT(!is_snapshot());
    typename write_set_map::iterator it     = write_set.begin();
    typename write_set_map::iterator it_end = write_set.end();
    for (size_t pos = 0; it != it_end; ++it, ++pos) {
      // this invariant does not make sense, since we do not hold lock for inserts anymore
      // all write set locks are grabbed during commit phase
//      INVARIANT(!it->is_insert() || it->get_tuple()->is_locked());
      write_dbtuples.emplace_back(it->get_tuple(), &(*it), it->is_insert(), pos);
    }
  }

  // lock write nodes
  if (!write_dbtuples.empty()) {
    PERF_DECL(
        static std::string probe2_name(
        std::string(__PRETTY_FUNCTION__) + std::string(":lock_write_nodes:")));
    ANON_REGION(probe2_name.c_str(), &transaction_base::g_txn_commit_probe2_cg);
    // lock the logical nodes in sort order
    {
      PERF_DECL(
          static std::string probe6_name(
          std::string(__PRETTY_FUNCTION__) + std::string(":sort_write_nodes:")));
      ANON_REGION(probe6_name.c_str(), &transaction_base::g_txn_commit_probe6_cg);
      write_dbtuples.sort(); // in-place
    }
    typename dbtuple_write_info_vec::iterator it     = write_dbtuples.begin();
    typename dbtuple_write_info_vec::iterator it_end = write_dbtuples.end();
    dbtuple_write_info *last_px = nullptr;
    bool inserted_last_run = false;
    for (; it != it_end; last_px = &(*it), ++it) {
      if (likely(last_px && last_px->tuple != it->tuple)) {
        // on boundary
        if (unlikely(!handle_last_tuple_in_group(*last_px, inserted_last_run, lock_mode))) {
          abort_trap((reason = ABORT_REASON_WRITE_NODE_INTERFERENCE));
          goto do_abort;
        }
        inserted_last_run = false;
      }
      if (it->is_insert()) {
        INVARIANT(!last_px || last_px->tuple != it->tuple);
        // INVARIANT(!it->is_locked());
        // INVARIANT(it->get_tuple()->is_locked());
        // INVARIANT(it->get_tuple()->is_lock_owner());
        // it->entry->set_do_write(); // all inserts are marked do-write
        // no need to mark do_write. since handle_last_tuple_in_group will handle
        inserted_last_run = true;
      } else {
        INVARIANT(!it->is_locked());
      }
    }
    if (likely(last_px) &&
        unlikely(!handle_last_tuple_in_group(*last_px, inserted_last_run, lock_mode))) {
      abort_trap((reason = ABORT_REASON_WRITE_NODE_INTERFERENCE));
      goto do_abort;
    }
    PERF_DECL(
        static std::string probe5_name(
        std::string(__PRETTY_FUNCTION__) + std::string(":gen_commit_tid:")));
    ANON_REGION(probe5_name.c_str(), &transaction_base::g_txn_commit_probe5_cg);
  } else {
    VERBOSE(std::cerr << "commit tid: <read-only>" << std::endl);
  }


  //Phase3. Read set validation
  {
    // check the nodes we actually read are still the latest version
    if (!read_set.empty()) {
      typename read_set_map::iterator it     = read_set.begin();
      typename read_set_map::iterator it_end = read_set.end();
      for (; it != it_end; ++it) {
        if (it->is_occ() || it->get_dep_txn() == nullptr) {
          // XXX(Jiachen): OCC validation - check the up to date tuple version equals the one we read before
          // IC3 action which returns a clean version should also be validated here
          const bool found = sorted_dbtuples_contains(write_dbtuples, it->get_tuple());
          if (likely(found ?
                     it->get_tuple()->is_latest_version(it->clean_tid()) :
                     it->get_tuple()->stable_is_latest_version(it->clean_tid())))
            continue;

          // occ action fail, increase the contention counter
          // if (it->is_occ() && add_clean_read_failed_tuples(it->get_tuple())) {
          //   it->get_tuple()->increase_counter();
          //   // printf("txn %ld ++ on %ld \n", get_tid(), it->get_tuple());
          // }

          VERBOSE(std::cerr << "fail occ validation fail! self id: "<< get_tid()
                            << " In Write set:" << found
                            << ", Clean tid: " << it->clean_tid()
                           << ", Current tid: " << it->get_tuple()->version
                           << ", Txn Type: " << int(txn_type) << ";" << std::endl);
        } else {
          // XXX(Jiachen): IC3 validation - check the up to date tuple version equals the dependent tid
          // only IC3 action which returns a dirty version should reach here
          Protocol<Traits>* d_txn = (Protocol<Traits>*)it->get_dep_txn();

          // have to guarantee dependent txn has already committed
          if (unlikely(!d_txn->is_commit(it->get_dep_tid()))) {
            VERBOSE(std::cerr << "Cascading abort happen sue to txn id -  " << d_txn->tid << std::endl);
            abort_trap((reason = ABORT_REASON_CASCADING));
            goto do_abort;
          }

          // skip if: 1. read a value written by its own 2. version unchanged compared to commit_tid
          const dbtuple *tuple = it->get_tuple();
          const bool found = sorted_dbtuples_contains(write_dbtuples, tuple);
          if ((d_txn->get_tid() == it->get_dep_tid()) &&
              (d_txn->get_tid() == get_tid() || (d_txn->get_commit_tid() == tuple->version && 
                                                 tuple->get_last_writer_seq() == it->get_dep_write_seq())) &&
              (found || !tuple->is_write_intent()))
            continue;

          VERBOSE(std::cerr << "fail ic3 validation fail! self id: "<< get_tid()
                            << " Dep txn tid:" << d_txn->get_tid()
                            << ", Dep commit tid: " << d_txn->get_commit_tid()
                            << ", Current tid: " << tuple->version
                            << ", Txn Type: " << int(txn_type) << std::endl);
        }
        abort_trap((reason = ABORT_REASON_READ_NODE_INTEREFERENCE));
        goto do_abort;
      }
    }

    // check btree node versions have not changed
    if (!absent_set.empty()) {
      typename absent_set_map::iterator it     = absent_set.begin();
      typename absent_set_map::iterator it_end = absent_set.end();
      for (; it != it_end; ++it) {
        const uint64_t v = concurrent_btree::ExtractVersionNumber(it->first);
        if (unlikely(v != it->second.version)) {
          VERBOSE(std::cerr << "expected node " << util::hexify(it->first) << " at v="
                            << it->second.version << ", got v=" << v << std::endl);
          abort_trap((reason = ABORT_REASON_NODE_SCAN_READ_VERSION_CHANGED));
          goto do_abort;
        }
      }
    }
  }

  // XXX(Jiachen): Stamp all the tree node in the absent set
  if (!absent_set.empty()) {
    auto it = absent_set.begin();
    for (; it != absent_set.end(); ++it) {
      if(!it->second.occ) {
        concurrent_btree::node_opaque_t *node = const_cast<concurrent_btree::node_opaque_t *>(it->first);
        node->set_last_txn(get_tid(), this, it->second.low_key, it->second.up_key);
      }
    }
  }

  // promise commit, decrease the counters of all failed clean read tuples in history
  // for (auto it : clean_read_failed_tuples) {
  //   if (dbtuple *tuple = static_cast<dbtuple *>(it)) {
  //     tuple->decrease_counter();
  //     // printf("txn %ld -- on %ld \n", get_tid(), tuple);
  //   }
  // }
  clean_read_failed_tuples.clear();

  //Phase4. Commit the updates
  if(!write_set.empty()) {

    if(commit_tid.first != true) {
      commit_tid.first = true;
      commit_tid.second = cast()->gen_commit_tid();
      set_commit_tid(commit_tid.second);
    }

    typename write_set_map::iterator it     = write_set.begin();
    typename write_set_map::iterator it_end = write_set.end();

    for(int seq = 0; it != it_end; ++it, ++seq) {

      access_entry *e = it->get_access_entry();
      dbtuple *tuple = it->get_tuple();

      if (e) {
        dbtuple* target = static_cast<dbtuple *>(e->get_commit_target());
      if(target != nullptr && target != tuple)
        tuple = target;
      }

      // reclaim contention counter
      // tuple->decrease_counter();

      if (unlikely(!it->do_write())) {
        // unlink the write entry
        if(e && e->is_linked()) {
          tuple->unlink_entry(e);
        }
        // unlink all related read entry (if exists)
        access_entry *read_e = it->get_read_access_entry();
        if(read_e && read_e->is_linked()) {
          tuple->unlink_entry(read_e);
        }
        continue;
      }

      // Since we fix the insert operation, here all writes can be handled as normal put

      tuple->prefetch();
      const dbtuple::write_record_ret ret =
          tuple->write_record_at(
              cast(), commit_tid.second,
              it->get_value(), it->get_writer());

      tuple->set_last_writer_seq(seq);

      bool unlock_head = false;

      if (unlikely(ret.head_ != tuple)) {
        // tuple was replaced by ret.head_
        ALWAYS_ASSERT(ret.rest_ == tuple);

        if (lock_mode)
          // if spinlock, we need to lock the head
          // if msc, the spilled tuple's mcs_lock is passed from original tuple, which means we are already holding it
          ret.head_->chamcc_lock(lock_mode, nullptr, true);
        
        unlock_head = true;

        // need to unlink tuple from underlying btree, replacing
        // with ret.rest_ (atomically)
        typename concurrent_btree::value_type old_v = 0;
        if (it->get_btree()->insert(
            varkey(it->get_key()), (typename concurrent_btree::value_type) ret.head_, &old_v, NULL))
          // should already exist in tree
          INVARIANT(false);
        INVARIANT(old_v == (typename concurrent_btree::value_type) tuple);
        // we don't RCU free this, because it is now part of the chain
        // (the cleaners will take care of this)
        ++evt_dbtuple_latest_replacement;
      }
      if (unlikely(ret.rest_))
        // spill happened: schedule GC task
        cast()->on_dbtuple_spill(ret.head_, ret.rest_);
      if (!it->get_value())
        // logical delete happened: schedule GC task
        cast()->on_logical_delete(ret.head_, it->get_key(), it->get_btree());


      if (unlikely(unlock_head)) {

        if (lock_mode)
          // if spinlock, we need to unlock the head
          // if msc, no need to unlock with mcs_lock
          ret.head_->chamcc_unlock(lock_mode, nullptr);

        //Unlink the entry
        if (e && e->is_linked()) {
          ret.head_->unlink_entry(e);
        }
      }

      if (e && e->is_linked()) {
        tuple->unlink_entry(e);
      }

      access_entry *read_e = it->get_read_access_entry();
      if(read_e && read_e->is_linked()) {
        tuple->unlink_entry(read_e);
      }

      tuple->chamcc_unlock(lock_mode, nullptr);
    }
  }

  //Phase5. Unlink all the other entries(read access entries with no overwritten)
  if (!read_set.empty()) {

    typename read_set_map::iterator it     = read_set.begin();
    typename read_set_map::iterator it_end = read_set.end();
    for (; it != it_end; ++it) {
      if (it->is_occ()) continue;

      access_entry* e = it->get_access_entry();
      if (!e) continue;

      dbtuple* tuple = static_cast<dbtuple *>(e->get_commit_target());

      // reclaim contention counter
      // if (!it->is_occ() && !e->is_overwrite()) it->get_tuple()->decrease_counter();

      if(e->is_linked()) {

        INVARIANT(!e->is_overwrite());

        tuple->chamcc_lock(lock_mode, nullptr);

        dbtuple* target = static_cast<dbtuple *>(e->get_commit_target());
        if(target != nullptr && target != tuple)
          tuple = target;

        tuple->unlink_entry(e);

        tuple->chamcc_unlock(lock_mode, nullptr);

      }
    }
  }


  //Phase6. Final stage, store some metadata and about to commit
  // XXX(Jiachen): Need to store its commit_tid, for transactions which depend on itself to do the ic3 validation
  VERBOSE(std::cerr << "Txn with id " << get_tid() << " commit, commit_tid is " << commit_tid << std::endl);

#ifndef PIPELINE_COMMIT
  state = TXN_COMMITED;
#endif   
 //fprintf(stderr, "[transaction<Protocol, Traits>::commit] core[%d] txn t[%d]< %lx, %lx> finish  commit (write set size %d)\n",
        //coreid::core_id(), txn_type, tid, this, write_set.size());

  if (commit_tid.first)
    cast()->on_tid_finish(commit_tid.second);

  return true;

do_abort:

  chamcc_abort_impl(&write_dbtuples, lock_mode);

  //TODO: Support user initial aborts
  // ALWAYS_ASSERT(false);
  // XXX: these values are possibly un-initialized
//  abort_impl(ABORT_REASON_CASCADING);

  return false;
}

template <template <typename> class Protocol, typename Traits>
std::pair< dbtuple *, bool >
transaction<Protocol, Traits>::try_insert_new_tuple(
    concurrent_btree &btr,
    const std::string *key,
    const void *value,
    dbtuple::tuple_writer_t writer)
{
  ALWAYS_ASSERT(false);
  INVARIANT(key);
  const size_t sz =
    value ? writer(dbtuple::TUPLE_WRITER_COMPUTE_NEEDED,
      value, nullptr, 0) : 0;

  // perf: ~900 tsc/alloc on istc11.csail.mit.edu
  dbtuple * const tuple = dbtuple::alloc_first(sz, true);
  if (value)
    writer(dbtuple::TUPLE_WRITER_DO_WRITE,
        value, tuple->get_value_start(), 0);
  INVARIANT(find_read_set(tuple) == read_set.end());
  INVARIANT(tuple->is_latest());
  INVARIANT(tuple->version == dbtuple::MAX_TID);
  INVARIANT(tuple->is_locked());
  INVARIANT(tuple->is_write_intent());
#ifdef TUPLE_CHECK_KEY
  tuple->key.assign(key->data(), key->size());
  tuple->tree = (void *) &btr;
#endif

  // XXX: underlying btree api should return the existing value if insert
  // fails- this would allow us to avoid having to do another search
  typename concurrent_btree::insert_info_t insert_info;
  if (unlikely(!btr.insert_if_absent(
          varkey(*key), (typename concurrent_btree::value_type) tuple, &insert_info))) {
    VERBOSE(std::cerr << "insert_if_absent failed for key: " << util::hexify(key) << std::endl);
    tuple->clear_latest();
    tuple->unlock();
    dbtuple::release_no_rcu(tuple);
    ++transaction_base::g_evt_dbtuple_write_insert_failed;
    return std::pair< dbtuple *, bool >(nullptr, false);
  }
  VERBOSE(std::cerr << "insert_if_absent suceeded for key: " << util::hexify(key) << std::endl
                    << "  new dbtuple is " << util::hexify(tuple) << std::endl);
  // update write_set
  // too expensive to be practical
  // INVARIANT(find_write_set(tuple) == write_set.end());
  //write_set.emplace_back(tuple, key, value, writer, &btr, true);

  access_entry* e = insert_read_set(tuple, dbtuple::MAX_TID, dbtuple::MAX_TID, get_tid(), this);
  insert_write_set(tuple, key, value, writer, &btr, true, e);
  e->write(const_cast<void *>(value), dbtuple::MAX_TID, string_allocator());

  // update node #s
  INVARIANT(insert_info.node);
  if (!absent_set.empty()) {
    auto it = absent_set.find(insert_info.node);
    if (it != absent_set.end()) {
      if (unlikely(it->second.version != insert_info.old_version)) {
        abort_trap((reason = ABORT_REASON_WRITE_NODE_INTERFERENCE));
        return std::make_pair(tuple, true);
      }
      VERBOSE(std::cerr << "bump node=" << util::hexify(it->first) << " from v=" << insert_info.old_version
                        << " -> v=" << insert_info.new_version << std::endl);
      // otherwise, bump the version
      it->second.version = insert_info.new_version;
      SINGLE_THREADED_INVARIANT(concurrent_btree::ExtractVersionNumber(it->first) == it->second);
    }
  }
  return std::make_pair(tuple, false);
}

template <template <typename> class Protocol, typename Traits>
template <typename ValueReader>
bool
transaction<Protocol, Traits>::do_tuple_read(
    const dbtuple *tuple, ValueReader &value_reader, bool occ = true)
{
  INVARIANT(tuple);
  ++evt_local_search_lookups;

  const bool is_snapshot_txn = is_snapshot();

  if(!is_snapshot_txn)
    return mix_op.do_tuple_read(const_cast<dbtuple*>(tuple), value_reader, occ);

  const transaction_base::tid_t snapshot_tid = is_snapshot_txn ?
    cast()->snapshot_tid() : static_cast<transaction_base::tid_t>(dbtuple::MAX_TID);
  transaction_base::tid_t start_t = 0;

  if (Traits::read_own_writes) {
    // this is why read_own_writes is not performant, because we have
    // to do linear scan
    auto write_set_it = find_write_set(const_cast<dbtuple *>(tuple));
    if (unlikely(write_set_it != write_set.end())) {
      ++evt_local_search_write_set_hits;
      if (!write_set_it->get_value()) {
        return false;
      }
      const typename ValueReader::value_type * const px =
        reinterpret_cast<const typename ValueReader::value_type *>(
            write_set_it->get_value());
      value_reader.dup(*px, this->string_allocator());
      return true;
    }
  }

  // do the actual tuple read
  dbtuple::ReadStatus stat;
  {
    PERF_DECL(static std::string probe0_name(std::string(__PRETTY_FUNCTION__) + std::string(":do_read:")));
    ANON_REGION(probe0_name.c_str(), &private_::txn_btree_search_probe0_cg);
    tuple->prefetch();
    stat = tuple->stable_read(snapshot_tid, start_t, value_reader, this->string_allocator(), is_snapshot_txn);
    if (unlikely(stat == dbtuple::READ_FAILED)) {
      const transaction_base::abort_reason r = transaction_base::ABORT_REASON_UNSTABLE_READ;
      abort_impl(r);
      throw transaction_abort_exception(r);
    }
  }
  if (unlikely(!cast()->can_read_tid(start_t))) {
    const transaction_base::abort_reason r = transaction_base::ABORT_REASON_FUTURE_TID_READ;
    abort_impl(r);
    throw transaction_abort_exception(r);
  }
  INVARIANT(stat == dbtuple::READ_EMPTY ||
            stat == dbtuple::READ_RECORD);
  const bool v_empty = (stat == dbtuple::READ_EMPTY);
  if (v_empty)
    ++transaction_base::g_evt_read_logical_deleted_node_search;

  //In IC3, only read-only transaction will call this function
  INVARIANT (is_snapshot_txn);
    // read-only txns do not need read-set tracking
    // (b/c we know the values are consistent)
    //read_set.emplace_back(tuple, start_t);
  return !v_empty;
}

template <template <typename> class Protocol, typename Traits>
void
transaction<Protocol, Traits>::do_node_read(
    const typename concurrent_btree::node_opaque_t *n, uint64_t v,
    const std::string* lkey, const std::string* ukey, bool occ = true)
{
  INVARIANT(n);
  if (is_snapshot())
    return;

  mix_op.do_node_read(n, v, lkey, ukey, occ);
}

template <template <typename> class Protocol, typename Traits>
inline void
transaction<Protocol, Traits>::add_dep_txn(uint64_t tid, void* txn, bool dirty_read_dep = false)
{

  typename dep_queue_map::iterator it = dep_queue.begin();
  for (; it != dep_queue.end(); ++it) {
    if(it->tid == tid) {
      if (dirty_read_dep) it->dirty_read_dep = true;
      return;
    }
  }

  assert(dep_queue.size() < traits_type::dep_queue_expected_size);
  dep_queue.emplace_back(dirty_read_dep, tid, txn);
  feature->tx_n_dep_on ++;

}


template <template <typename> class Protocol, typename Traits>
void
transaction<Protocol, Traits>::check_abort()
{
#ifdef CASCADING_ABORT_BY_EXCEPTION
  if(state == TXN_ABRT) {
    const transaction_base::abort_reason r = transaction_base::ABORT_REASON_CASCADING;
    throw transaction_abort_exception(r);
  }
#endif
}

template <template <typename> class Protocol, typename Traits>
bool
transaction<Protocol, Traits>::should_abort()
{
  return (state == TXN_ABRT);
}


template <template <typename> class Protocol, typename Traits>
void
transaction<Protocol, Traits>::atomic_piece_begin(bool one_tuple){}


template <template <typename> class Protocol, typename Traits>
bool
transaction<Protocol, Traits>::atomic_piece_end(){}

template <template <typename> class Protocol, typename Traits>
bool
transaction<Protocol, Traits>::atomic_piece_abort() {}

template <template <typename> class Protocol, typename Traits>
void
transaction<Protocol, Traits>::do_wait(AccessPolicy access, double rank, uint32_t timeout, uint32_t safeguard[TXN_TYPE])
{
  if (access == detect_track_dirty || access == no_detect)
    // do not detect any conflict --> no need to wait.
    return ;

  uint64_t start_time = util::timer::cur_usec();
  typename dep_queue_map::iterator it = dep_queue.begin();
#ifdef COUNT_TX_BLOCK
  global_listener.tx_n_blocked ++;
#endif

  if (access == detect_all) {
    for (; it != dep_queue.end(); ++it) {
      auto d_txn = (transaction<Protocol, Traits> *)it->txn;
#ifdef COUNT_TX_BLOCK
      d_txn->txn_current_blocking++;
#endif
        // detect all conflicts between transaction operations.
        while (!(d_txn->is_commit(it->tid) || d_txn->is_abort(it->tid))) {
          memory_barrier();
          nop_pause();

          // check timeout
          if (unlikely((util::timer::cur_usec() - start_time) > timeout)) {
#ifdef COUNT_TX_BLOCK
            global_listener.tx_n_blocked--;
            d_txn->txn_current_blocking--;
#endif
            state = TXN_ABRT;
            const transaction_base::abort_reason r = transaction_base::ABORT_REASON_TIMEOUT;
            throw transaction_abort_exception(r);
          }
        }
        if (d_txn->is_abort(it->tid) && it->dirty_read_dep) {
#ifdef COUNT_TX_BLOCK
          global_listener.tx_n_blocked--;
          d_txn->txn_current_blocking--;
#endif
          state = TXN_ABRT;
          const transaction_base::abort_reason r = transaction_base::ABORT_REASON_CASCADING;
          throw transaction_abort_exception(r);
        }
#ifdef COUNT_TX_BLOCK
        d_txn->txn_current_blocking--;
#endif
      }
  } else if (access == detect_guarded) {
      for (; it != dep_queue.end(); ++it) {
        auto d_txn = (transaction<Protocol, Traits> *)it->txn;
#ifdef COUNT_TX_BLOCK
        d_txn->txn_current_blocking++;
#endif
        if (d_txn->txn_type == 0)
          continue ;
        auto to_step = safeguard[d_txn->txn_type-1];
        while (!(d_txn->is_commit(it->tid) || d_txn->is_abort(it->tid) || to_step < d_txn->cur_step)) {
          memory_barrier();
          nop_pause();

          // check timeout
          if (unlikely((util::timer::cur_usec() - start_time) > timeout)) {
#ifdef COUNT_TX_BLOCK
            global_listener.tx_n_blocked--;
            d_txn->txn_current_blocking--;
#endif
            state = TXN_ABRT;
            const transaction_base::abort_reason r = transaction_base::ABORT_REASON_TIMEOUT;
            throw transaction_abort_exception(r);
          }
        }
        if (d_txn->is_abort(it->tid) && it->dirty_read_dep) {
#ifdef COUNT_TX_BLOCK
          global_listener.tx_n_blocked--;
          d_txn->txn_current_blocking--;
#endif
          state = TXN_ABRT;
          const transaction_base::abort_reason r = transaction_base::ABORT_REASON_CASCADING;
          throw transaction_abort_exception(r);
        }
#ifdef COUNT_TX_BLOCK
        d_txn->txn_current_blocking--;
#endif
      }
  }
#ifdef COUNT_TX_BLOCK
  global_listener.tx_n_blocked --;
#endif
}

#endif /* _NDB_TXN_IMPL_H_ */

