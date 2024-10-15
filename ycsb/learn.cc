#include "learn.h"
#include "tuple.h"
#include <cassert>

contention_encoder global_encoder;
plan_listener global_listener;

void profiling(const std::string& bench)
{
  global_listener.print_abort_distribution();
  global_listener.print_state_distribution(bench);
}