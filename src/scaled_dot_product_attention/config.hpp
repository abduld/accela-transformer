#pragma once

#include "utils.hpp"

// Attention layer hyperparameters taken from
// https://huggingface.co/transformers/v2.2.0/pretrained_models.html for GPT-2 small (note: full
// model has 12 layers)

// a sqrt that works on costexpr
namespace {
constexpr std::size_t isqrt_impl(std::size_t sq, std::size_t dlt, std::size_t value) {
  return sq <= value ? isqrt_impl(sq + dlt, dlt + 2, value) : (dlt >> 1) - 1;
}

constexpr std::size_t isqrt(std::size_t value) {
  return isqrt_impl(1, 3, value);
}
} // namespace

static constexpr size_t BATCH_SIZE      = 1;
static constexpr size_t SEQUENCE_LENGTH = 10;
static constexpr size_t DM              = 768;
static constexpr size_t DFF             = 3072;
static constexpr size_t DK              = 64;
static constexpr size_t DV              = 64;
static constexpr size_t NUM_HEADS       = 12;
static constexpr float TEMPERATURE      = isqrt(DK);
static constexpr float TEMPERATURE_INV  = 1.0f / isqrt(DK);