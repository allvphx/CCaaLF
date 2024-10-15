#include <algorithm>
#include <fstream>
#include <iostream>
#include <regex>

#include "macros.h"
#include "policy.h"
#include "learn.h"

#define REP(i, s, t) for(int (i)=(s);(i)<(t);(i)++)

PolicyAction before_commit_policy = PolicyAction(detect_all, highest_priority, 0, blocked_wait);

void Policy::print_policy(const std::string &bench) const {
  printf("Profile: the policy\n"
         "<--------------------------------------->\n");
  if (bench == "tpcc") {
    printf("\naccess\n");
    REP(j, 0, global_encoder.max_state) printf("%d,", policy[j].access);
    printf("\npriority\n");
    REP(j, 0, global_encoder.max_state) printf("%.2f,", policy[j].rank);
    printf("\ntimeout\n");
    REP(j, 0, global_encoder.max_state) printf("%d,", policy[j].timeout);
    printf("\nexpose\n");
    REP(j, 0, global_encoder.max_state) printf("%d,", policy[j].expose);
    printf("\nexpose\n");
    REP(j, 0, global_encoder.max_state) printf("%d,", policy[j].expose);
    printf("access type\n");
    REP(i, 0, global_encoder.max_state) {
      if (policy[i].access == no_detect || policy[i].access == detect_track_dirty) {
        printf("%s\t%s\t\t%d (ns)\n", policy[i].access == no_detect? "NO DE":"DIRTY",
               policy[i].expose? "EXPO":"HIDE", policy[i].timeout);
      } else if (policy[i].access == detect_all)
        printf("ALL\t%s\t\t%d (ns)\n", policy[i].expose? "EXPO":"HIDE", policy[i].timeout);
      else {
        printf("GUARD\t%s\t(%d %d %d)\t%d (ns)\n", policy[i].expose? "EXPO":"HIDE",
               policy[i].safeguard[0], policy[i].safeguard[1], policy[i].safeguard[2],
               policy[i].timeout);
      }
    }
  } else if (bench == "ycsb") {
    assert(false);
  }
  printf("<--------------------------------------->\n");
}

void Policy::init_occ() {
  REP(state, 0, MAX_STATE)
  policy[state] = PolicyAction(no_detect, lowest_priority, false, blocked_wait);
}

void Policy::init_2pl() {
  REP(state, 0, MAX_STATE)
  policy[state] = PolicyAction(detect_all, lowest_priority, false, blocked_wait);
}

void Policy::init_pipeline_execution() {
  REP(state, 0, MAX_STATE)
  policy[state] = PolicyAction(detect_track_dirty, lowest_priority, true, blocked_wait);
}


Policy::Policy() {
  policy = new PolicyAction [MAX_STATE];
  init_occ();
}

std::vector<float> parseFloatString(const std::string& str) {
  std::vector<float> result;
  std::stringstream ss(str);
  float num;
  while (ss >> num) result.push_back(num);
  return result;
}

void Policy::init(std::ifstream *pol_file) {
  std::string not_using;
  std::string access_policy_str;
  std::string rank_str;
  std::string timeout_str;
  std::string expose_str;
  std::string wait_chop_str;
  std::string extra_str;
  std::getline(*pol_file, not_using);
  std::getline(*pol_file, access_policy_str);
  std::getline(*pol_file, not_using);
  std::getline(*pol_file, rank_str);
  std::getline(*pol_file, not_using);
  std::getline(*pol_file, timeout_str);
  std::getline(*pol_file, not_using);
  std::getline(*pol_file, expose_str);
  std::getline(*pol_file, not_using);
  std::getline(*pol_file, wait_chop_str);
  std::getline(*pol_file, not_using);
  std::getline(*pol_file, extra_str);
  int s;

  std::vector<AccessPolicy> access_vec;
  access_vec.resize(MAX_STATE);
  std::transform(access_policy_str.begin(), access_policy_str.end(), access_vec.begin(),
                 [](char ac) -> AccessPolicy {
                   if (ac == '0') return no_detect;
                   else if (ac == '1') return detect_guarded;
                   else if (ac == '2') return detect_guarded;
                   else if (ac == '3') return detect_all;
                   else ALWAYS_ASSERT(false);
                 });
  s = 0;
  for (auto it:access_vec) policy[s++].access = it;
  std::vector<WaitPriority> rank_vec = parseFloatString(rank_str);
  s = 0;
  for (auto it:rank_vec) policy[s++].rank = it;
  std::vector<float> timeout_vec = parseFloatString(timeout_str);
  s = 0;
  for (auto it:timeout_vec) policy[s++].timeout = static_cast<uint32_t>(it);

  std::vector<bool> expose_vec;
  expose_vec.resize(MAX_STATE);
  std::transform(expose_str.begin(), expose_str.end(), expose_vec.begin(),
                 [](char ac) -> bool {
                   if (ac == '0') return false;
                   else if (ac == '1') return true;
                   else ALWAYS_ASSERT(false);
                 });
  s = 0;
  for (auto it:expose_vec) policy[s++].expose = it;
  std::vector<float> wait_chop = parseFloatString(wait_chop_str);
  int n = global_encoder.max_state;
  for (int i=0;i<TXN_TYPE;i ++) {
    for (int j=0;j<n;j++) {
      policy[j].safeguard[i] = uint32_t(wait_chop[j + n*i]);
    }
  }

  std::vector<float> extra_learn = parseFloatString(extra_str);
  auto it = extra_learn.begin();
  txn_buf_size = uint32_t (*it);
  for (int op = 0; op < 2; op ++) {
    for (int i = 0; i < RETRY_TIMES; ++i) {
      backoff[op][i][0] = 0.0;
      it ++;
      for (int j = 0; j < TXN_TYPE; j ++) backoff[op][i][j] = *it;
    }
  }
  assert(it == extra_learn.end());
}

void Policy::policy_gradient(const std::string &policy_f) {
  if (policy_f == "2pl") {
    init_2pl();
  } else if (policy_f == "pipe") {
    init_pipeline_execution();
  } else {
    std::ifstream pol_file;
    pol_file.open(policy_f);
    if (!pol_file.is_open()) {
      std::cerr << "Could not open file: " << policy_f << ". "
                << "Please specify correct policy file with the --policy option"
                << std::endl;
      ALWAYS_ASSERT(false);
    }
    init(&pol_file);
  }
}

Policy::~Policy() {
  delete[] policy;
}