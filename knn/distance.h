#pragma once
#include "util.h"

namespace knn {

// ----------------------------------------------------------------------------
// similar - cosine similarity distance, get/set optional dim & epsilon
// pairwise - pairwise distance, get/set optional power,eps,deep dimension flag
// ----------------------------------------------------------------------------
torch::nn::CosineSimilarityOptions similar(K,J,Cast);
K similar(bool,const torch::nn::CosineSimilarityOptions&);

torch::nn::PairwiseDistanceOptions pairwise(K,J,Cast);
K pairwise(bool,const torch::nn::PairwiseDistanceOptions&);

} // namespace knn
