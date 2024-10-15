#include <algorithm>
#include <fstream>
#include <iostream>
#include <regex>

#include "macros.h"
#include "policy.h"
#include "learn.h"

#define REP(i, s, t) for(int (i)=(s);(i)<(t);(i)++)

void Policy::print_policy(std::string bench) {
    printf("Profile: the policy\n"
           "<--------------------------------------->\n");
    if (bench == "tpcc") {
        printf("\naccess\n");
        REP(j, 0, global_encoder.max_state) printf("%d,", policy[j].access);
        printf("\npriority\n");
        REP(j, 0, global_encoder.max_state) printf("%.2f,", policy[j].rank);
        printf("\ntimeout\n");
        REP(j, 0, global_encoder.max_state) printf("%d,", policy[j].timeout);
        printf("\n");
    } else if (bench == "ycsb") {
        assert(false);
    }
    printf("<--------------------------------------->\n");
}

void Policy::init_2pl_wait_die() {
    REP(state, 0, global_encoder.max_state)
        policy[state] = PolicyAction(detect_existing, maximum_priority, DEFAULT_TIMEOUT);
}

void Policy::init_occ() {
    REP(state, 0, global_encoder.max_state)
        policy[state] = PolicyAction(no_detect, 0, 0);
}

void Policy::init_2pl_no_wait() {
    REP(state, 0, global_encoder.max_state)
        policy[state] = PolicyAction(detect_existing, 0, 0);
}


Policy::Policy() {
    policy = new PolicyAction [MAX_STATE];
    init_2pl_wait_die();
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
    std::getline(*pol_file, not_using);
    std::getline(*pol_file, access_policy_str);
    std::getline(*pol_file, not_using);
    std::getline(*pol_file, rank_str);
    std::getline(*pol_file, not_using);
    std::getline(*pol_file, timeout_str);
    int s;

    std::vector<AccessPolicy> access_vec;
    access_vec.resize(global_encoder.max_state);
    std::transform(access_policy_str.begin(), access_policy_str.end(), access_vec.begin(),
                   [](char ac) -> AccessPolicy {
                       if (ac == '0') return no_detect;
                       else if (ac == '1')
                           return detect_critical;
                       else if (ac == '2')
                           return detect_existing;
                       else if (ac == '3')
                           return detect_future;
                       else
                           ALWAYS_ASSERT(false);
                   });
    s = 0;
    for (auto it: access_vec) policy[s++].access = it;
    std::vector<WaitPriority> rank_vec = parseFloatString(rank_str);
    s = 0;
    for (auto it: rank_vec) policy[s++].rank = it;
    std::vector<float> timeout_vec = parseFloatString(timeout_str);
    s = 0;
    for (auto it: timeout_vec) policy[s++].timeout = static_cast<uint32_t>(it);
}

void Policy::policy_gradient(const std::string &policy_f)
{
    if (policy_f == "2pl") {
        init_2pl_wait_die();
    } else if (policy_f == "occ") {
        init_occ();
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