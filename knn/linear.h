#pragma once
#include "util.h"

namespace knn {

torch::nn::LinearOptions linear(K,J,Cast);
K linear(bool,const torch::nn::LinearOptions&);

torch::nn::BilinearOptions bilinear(K,J,Cast);
K bilinear(bool,const torch::nn::BilinearOptions&);

} // namespace knn
