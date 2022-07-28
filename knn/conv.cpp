#include "../ktorch.h"
#include "conv.h"

namespace knn {

torch::nn::detail::conv_padding_mode_t padmode(S s) {
 switch(emap(s)) {
  case Enum::zeros:     return torch::kZeros;
  case Enum::reflect:   return torch::kReflect;
  case Enum::replicate: return torch::kReplicate;
  case Enum::circular:  return torch::kCircular;
  default: TORCH_ERROR("unrecognized padding mode: ",s); break;
 }
}

S padmode(const torch::nn::detail::conv_padding_mode_t& p) {
 return ESYM(p);
}

} // knn namespace
