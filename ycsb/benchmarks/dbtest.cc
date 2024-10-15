#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <utility>
#include <string>
#include <set>

#include <getopt.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/sysinfo.h>

#include "../allocator.h"
#include "../stats_server.h"
#include "bench.h"
#include "bdb_wrapper.h"
#include "ldb_wrapper.h"
#include "ndb_wrapper.h"
#include "ndb_wrapper_impl.h"
#include "kvdb_wrapper.h"
#include "kvdb_wrapper_impl.h"
#include "../policy.h"

using namespace std;
using namespace util;

static vector<string>
split_ws(const string &s)
{
  vector<string> r;
  istringstream iss(s);
  copy(istream_iterator<string>(iss),
       istream_iterator<string>(),
       back_inserter<vector<string>>(r));
  return r;
}

static size_t
parse_memory_spec(const string &s)
{
  string x(s);
  size_t mult = 1;
  if (x.back() == 'G') {
    mult = static_cast<size_t>(1) << 30;
    x.pop_back();
  } else if (x.back() == 'M') {
    mult = static_cast<size_t>(1) << 20;
    x.pop_back();
  } else if (x.back() == 'K') {
    mult = static_cast<size_t>(1) << 10;
    x.pop_back();
  }
  return strtoul(x.c_str(), nullptr, 10) * mult;
}

int
main(int argc, char **argv)
{
  abstract_db *db = NULL;
  void (*test_fn)(abstract_db *, int argc, char **argv) = NULL;
  string bench_type = "ycsb";
  string db_type = "ndb-ic3";
  string policy = "";
  string encoder = "./encoder/default_ycsb_encoder.txt";
  char *curdir = get_current_dir_name();
  string basedir = curdir;
  string bench_opts;
  size_t numa_memory = 0;
  free(curdir);
  int saw_run_spec = 0;
  int nofsync = 0;
  int do_compress = 0;
  int fake_writes = 0;
  int disable_gc = 0;
  int disable_snapshots = 0;
  vector<string> logfiles;
  vector<vector<unsigned>> assignments;
  string stats_server_sockfile;
  while (1) {
    static struct option long_options[] =
    {
      {"verbose"                    , no_argument       , &verbose                   , 1}   ,
      {"consistency-check"          , no_argument       , &consistency_check         , 1}   ,
      {"dynamic-workload"           , no_argument       , &dynamic_workload          , 1}   ,
      {"parallel-loading"           , no_argument       , &enable_parallel_loading   , 1}   ,
      {"pin-cpus"                   , no_argument       , &pin_cpus                  , 1}   ,
      {"slow-exit"                  , no_argument       , &slow_exit                 , 1}   ,
      {"retry-aborted-transactions" , no_argument       , &retry_aborted_transaction , 1}   ,
      {"backoff-aborted-transactions" , no_argument     , &backoff_aborted_transaction , 1}   ,
      {"backoff-alpha"              , required_argument , 0                          , 'A'}   ,
      {"policy"                     , required_argument , 0                          , 'p'}   ,
      {"encoder"                    , required_argument , 0                          , 'e'}   ,
      {"bench"                      , required_argument , 0                          , 'b'} ,
      {"scale-factor"               , required_argument , 0                          , 's'} ,
      {"kid-start"                  , required_argument , 0                          , 'y'} ,
      {"kid-end"                    , required_argument , 0                          , 'z'} ,
      {"num-threads"                , required_argument , 0                          , 't'} ,
      {"db-type"                    , required_argument , 0                          , 'd'} ,
      {"basedir"                    , required_argument , 0                          , 'B'} ,
      {"txn-flags"                  , required_argument , 0                          , 'f'} ,
      {"runtime"                    , required_argument , 0                          , 'r'} ,
      {"ops-per-worker"             , required_argument , 0                          , 'n'} ,
      {"bench-opts"                 , required_argument , 0                          , 'o'} ,
      {"numa-memory"                , required_argument , 0                          , 'm'} , // implies --pin-cpus
      {"logfile"                    , required_argument , 0                          , 'l'} ,
      {"assignment"                 , required_argument , 0                          , 'a'} ,
      {"log-nofsync"                , no_argument       , &nofsync                   , 1}   ,
      {"log-compress"               , no_argument       , &do_compress               , 1}   ,
      {"log-fake-writes"            , no_argument       , &fake_writes               , 1}   ,
      {"disable-gc"                 , no_argument       , &disable_gc                , 1}   ,
      {"disable-snapshots"          , no_argument       , &disable_snapshots         , 1}   ,
      {"stats-server-sockfile"      , required_argument , 0                          , 'x'} ,
      {"no-reset-counters"          , no_argument       , &no_reset_counters         , 1}   ,
      {0, 0, 0, 0}
    };
    int option_index = 0;
    int c = getopt_long(argc, argv, "A:p:b:s:t:d:B:f:r:n:o:m:l:a:x:", long_options, &option_index);
    if (c == -1)
      break;

    switch (c) {
    case 0:
      if (long_options[option_index].flag != 0)
        break;
      abort();
      break;

    case 'p':
      policy = optarg;
      break;

    case 'e':
      encoder = optarg;
      break;

    case 'A':
      backoff_alpha = strtod(optarg, NULL);
      ALWAYS_ASSERT(backoff_alpha >= 0.0);
      break;

    case 'b':
      bench_type = optarg;
      break;

    case 's':
      scale_factor = strtod(optarg, NULL);
      ALWAYS_ASSERT(scale_factor > 0.0);
      break;

    case 'y':
      kid_start = strtoul(optarg, NULL, 10);
      break;

    case 'z':
      kid_end = strtoul(optarg, NULL, 10);
      break;

    case 't':
      nthreads = strtoul(optarg, NULL, 10);
      ALWAYS_ASSERT(nthreads > 0);
      break;

    case 'd':
      db_type = optarg;
      break;

    case 'B':
      basedir = optarg;
      break;

    case 'f':
      txn_flags = strtoul(optarg, NULL, 10);
      break;

    case 'r':
      ALWAYS_ASSERT(!saw_run_spec);
      saw_run_spec = 1;
      runtime = strtoul(optarg, NULL, 10);
      ALWAYS_ASSERT(runtime > 0);
      run_mode = RUNMODE_TIME;
      break;

    case 'n':
      ALWAYS_ASSERT(!saw_run_spec);
      saw_run_spec = 1;
      ops_per_worker = strtoul(optarg, NULL, 10);
      ALWAYS_ASSERT(ops_per_worker > 0);
      run_mode = RUNMODE_OPS;

    case 'o':
      bench_opts = optarg;
      break;

    case 'm':
      {
        pin_cpus = 1;
        const size_t m = parse_memory_spec(optarg);
        ALWAYS_ASSERT(m > 0);
        numa_memory = m;
      }
      break;

    case 'l':
      logfiles.emplace_back(optarg);
      break;

    case 'a':
      assignments.emplace_back(
          ParseCSVString<unsigned, RangeAwareParser<unsigned>>(optarg));
      break;

    case 'x':
      stats_server_sockfile = optarg;
      break;

    case '?':
      /* getopt_long already printed an error message. */
      exit(1);

    default:
      abort();
    }
  }

  if (bench_type == "ycsb")
    test_fn = ycsb_do_test;
  else if (bench_type == "tpcc")
    test_fn = tpcc_do_test;
  else if (bench_type == "micro_badcount")
    test_fn = micro_do_test;
  else if (bench_type == "micro_insert")
    test_fn = micro_insert_do_test;
  else if (bench_type == "micro_delete")
    test_fn = micro_delete_do_test;
  else if (bench_type == "micro_range")
    test_fn = micro_range_do_test;
  else if (bench_type == "micro_transitive")
    test_fn = micro_do_trasitive_test;
  else if (bench_type == "micro_transitive2")
    test_fn = micro_do_trasitive2_test;
  else if (bench_type == "micro_mem")
    test_fn = micro_do_mem_test;
  else if (bench_type == "lock_perf")
    test_fn = micro_lock_perf_test;
  else if (bench_type == "ic3_perf")
    test_fn = micro_ic3_perf_test;
  else if (bench_type == "queue")
    test_fn = queue_do_test;
  else if (bench_type == "encstress")
    test_fn = encstress_do_test;
  else if (bench_type == "tpce")
    test_fn = tpce_do_test;
  else if (bench_type == "smallbank")
    test_fn = small_do_test;
  else if (bench_type == "microbench")
    test_fn = microbench_do_test;
  else if (bench_type == "lockbench")
    test_fn = lockbench_do_test;
  else
    ALWAYS_ASSERT(false);

  if (do_compress && logfiles.empty()) {
    cerr << "[ERROR] --log-compress specified without logging enabled" << endl;
    return 1;
  }

  if (fake_writes && logfiles.empty()) {
    cerr << "[ERROR] --log-fake-writes specified without logging enabled" << endl;
    return 1;
  }

  if (nofsync && logfiles.empty()) {
    cerr << "[ERROR] --log-nofsync specified without logging enabled" << endl;
    return 1;
  }

  if (fake_writes && nofsync) {
    cerr << "[WARNING] --log-nofsync has no effect with --log-fake-writes enabled" << endl;
  }

#ifndef ENABLE_EVENT_COUNTERS
  if (!stats_server_sockfile.empty()) {
    cerr << "[WARNING] --stats-server-sockfile with no event counters enabled is useless" << endl;
  }
#endif

  // initialize the numa allocator
  if (numa_memory > 0) {
    const size_t maxpercpu = util::iceil(
        numa_memory / nthreads, ::allocator::GetHugepageSize());
    numa_memory = maxpercpu * nthreads;
    ::allocator::Initialize(nthreads, maxpercpu);
  }

  const set<string> can_persist({"ndb-ic3"});
  if (!logfiles.empty() && !can_persist.count(db_type)) {
    cerr << "[ERROR] benchmark " << db_type
         << " does not have persistence implemented" << endl;
    return 1;
  }

#ifdef PROTO2_CAN_DISABLE_GC
  const set<string> has_gc({"ndb-proto1", "ndb-ic3"});
  if (disable_gc && !has_gc.count(db_type)) {
    cerr << "[ERROR] benchmark " << db_type
         << " does not have gc to disable" << endl;
    return 1;
  }
#else
  if (disable_gc) {
    cerr << "[ERROR] macro PROTO2_CAN_DISABLE_GC was not set, cannot disable gc" << endl;
    return 1;
  }
#endif

#ifdef PROTO2_CAN_DISABLE_SNAPSHOTS
  const set<string> has_snapshots({"ndb-ic3"});
  if (disable_snapshots && !has_snapshots.count(db_type)) {
    cerr << "[ERROR] benchmark " << db_type
         << " does not have snapshots to disable" << endl;
    return 1;
  }
#else
  if (disable_snapshots) {
    cerr << "[ERROR] macro PROTO2_CAN_DISABLE_SNAPSHOTS was not set, cannot disable snapshots" << endl;
    return 1;
  }
#endif

  ALWAYS_ASSERT(db_type == "ndb-ic3");
  db = new ndb_wrapper<transaction_ic3>(
        logfiles, assignments, !nofsync, do_compress, fake_writes);
  ALWAYS_ASSERT(!transaction_ic3_static::get_hack_status());
#ifdef PROTO2_CAN_DISABLE_GC
    if (!disable_gc)
      transaction_ic3_static::InitGC();
#endif
#ifdef PROTO2_CAN_DISABLE_SNAPSHOTS
  if (disable_snapshots)
    transaction_ic3_static::DisableSnapshots();
#endif

#ifdef DEBUG
  cerr << "WARNING: benchmark built in DEBUG mode!!!" << endl;
#endif

#ifdef CHECK_INVARIANTS
  cerr << "WARNING: invariant checking is enabled - should disable for benchmark" << endl;
#ifdef PARANOID_CHECKING
  cerr << "  *** Paranoid checking is enabled ***" << endl;
#endif
#endif

  if (verbose) {
    const unsigned long ncpus = coreid::num_cpus_online();
    cerr << "Database Benchmark:"                           << endl;
    cerr << "  pid: " << getpid()                           << endl;
    cerr << "settings:"                                     << endl;
    cerr << "  par-loading : " << enable_parallel_loading   << endl;
    cerr << "  pin-cpus    : " << pin_cpus                  << endl;
    cerr << "  slow-exit   : " << slow_exit                 << endl;
    cerr << "  retry-txns  : " << retry_aborted_transaction << endl;
    cerr << "  backoff-txns: " << backoff_aborted_transaction << endl;
    cerr << "  bench       : " << bench_type                << endl;
    cerr << "  scale       : " << scale_factor              << endl;
    cerr << "  num-cpus    : " << ncpus                     << endl;
    cerr << "  num-threads : " << nthreads                  << endl;
    cerr << "  db-type     : " << db_type                   << endl;
    cerr << "  basedir     : " << basedir                   << endl;
    cerr << "  txn-flags   : " << hexify(txn_flags)         << endl;
    if (run_mode == RUNMODE_TIME)
      cerr << "  runtime     : " << runtime                 << endl;
    else
      cerr << "  ops/worker  : " << ops_per_worker          << endl;
#ifdef USE_VARINT_ENCODING
    cerr << "  var-encode  : yes"                           << endl;
#else
    cerr << "  var-encode  : no"                            << endl;
#endif

#ifdef USE_JEMALLOC
    cerr << "  allocator   : jemalloc"                      << endl;
#elif defined USE_TCMALLOC
    cerr << "  allocator   : tcmalloc"                      << endl;
#elif defined USE_FLOW
    cerr << "  allocator   : flow"                          << endl;
#else
    cerr << "  allocator   : libc"                          << endl;
#endif
    if (numa_memory > 0) {
      cerr << "  numa-memory : " << numa_memory             << endl;
    } else {
      cerr << "  numa-memory : disabled"                    << endl;
    }
    cerr << "  logfiles : " << logfiles                     << endl;
    cerr << "  assignments : " << assignments               << endl;
    cerr << "  disable-gc : " << disable_gc                 << endl;
    cerr << "  disable-snapshots : " << disable_snapshots   << endl;
    cerr << "  stats-server-sockfile: " << stats_server_sockfile << endl;

    cerr << "system properties:" << endl;
    cerr << "  btree_internal_node_size: " << concurrent_btree::InternalNodeSize() << endl;
    cerr << "  btree_leaf_node_size    : " << concurrent_btree::LeafNodeSize() << endl;

#ifdef TUPLE_PREFETCH
    cerr << "  tuple_prefetch          : yes" << endl;
#else
    cerr << "  tuple_prefetch          : no" << endl;
#endif

#ifdef BTREE_NODE_PREFETCH
    cerr << "  btree_node_prefetch     : yes" << endl;
#else
    cerr << "  btree_node_prefetch     : no" << endl;
#endif

  }

  if (!stats_server_sockfile.empty()) {
    stats_server *srvr = new stats_server(stats_server_sockfile);
    thread(&stats_server::serve_forever, srvr).detach();
  }

  global_encoder.load(encoder);
  Policy *pg = new Policy();  // the same as PolyJuice training script setting.
                                                                // policy initialization
  if (!policy.empty()) pg->policy_gradient(policy);

  db->pg = pg;
  vector<string> bench_toks = split_ws(bench_opts);
  int argc = 1 + bench_toks.size();
  char *argv[argc];
  argv[0] = (char *) bench_type.c_str();
  for (size_t i = 1; i <= bench_toks.size(); i++)
    argv[i] = (char *) bench_toks[i - 1].c_str();
  test_fn(db, argc, argv);
  if (verbose)
    pg->print_policy(bench_type);
  if (verbose)
    profiling(bench_type);
  delete db;
  return 0;
}
