#pragma once

#include <algorithm>
#include <vector>

#include <xsimd/xsimd.hpp> 

/*********************************************************************/
/* Random number generator                                           */
/* https://en.wikipedia.org/wiki/Xorshift                            */
/* xorshift32                                                        */
/*********************************************************************/

static uint_fast32_t rng_uint32(std::shared_ptr<uint_fast32_t> rng_state) {
  uint_fast32_t local = *rng_state;
  local ^= local << 13; // a
  local ^= local >> 17; // b
  local ^= local << 5;  // c
  *rng_state = local;
  return local;
}

static std::shared_ptr<uint_fast32_t> rng_new_state(uint_fast32_t seed = 88172645463325252LL) {
  return std::make_shared<uint64_t>(seed);
}

static float rng_float(std::shared_ptr<uint_fast32_t> state) {
  uint_fast32_t rnd = rng_uint32(state);
  const auto r      = static_cast<float>(rnd) / static_cast<float>(UINT_FAST32_MAX);
  if (std::isfinite(r)) {
    return r;
  }
  return rng_float(state);
}

static std::vector<float,xsimd::aligned_allocator<float, XSIMD_DEFAULT_ALIGNMENT>> random(std::shared_ptr<uint_fast32_t> rng_state, int m, int n = 1) {
  std::vector<float,xsimd::aligned_allocator<float, XSIMD_DEFAULT_ALIGNMENT>> out(m*n);
  std::generate(out.begin(), out.end(), [=]() { return rng_float(rng_state); });
  return out;
}

