#pragma once 

#include "utils.hpp"

// Attention layer hyperparameters taken from https://huggingface.co/transformers/v2.2.0/pretrained_models.html
// for GPT-2 small (note: full model has 12 layers)

static constexpr size_t BATCH_SIZE = 1;
static constexpr size_t SEQUENCE_LENGTH = 10;
static constexpr size_t DM =  768;
static constexpr size_t DFF = 3072;
static constexpr size_t DK = 64;
static constexpr size_t DV = 64;
static constexpr size_t NUM_HEADS = 12;
