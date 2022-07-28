#pragma once
#include "util.h"
  
namespace knn {

// ------------------------------------------------------------------
// OneHot - module for torch.nn.functional.one_hot(tensor,numclasses)
// ------------------------------------------------------------------
struct TORCH_API OneHotOptions {
 OneHotOptions(int64_t n=-1) : num_classes_(n) {}
 TORCH_ARG(int64_t, num_classes);
 TORCH_ARG(c10::optional<torch::Dtype>, dtype) = c10::nullopt;
};

class TORCH_API OneHotImpl : public torch::nn::Cloneable<OneHotImpl> {
 public:
 OneHotImpl(const OneHotOptions& o);
 void reset() override;
 void pretty_print(std::ostream& s) const override;
 Tensor forward(const torch::Tensor& x);

 OneHotOptions options;
};
TORCH_MODULE(OneHot);

// ------------------------------------------------
// onehot - get/set number of classes & data type 
//        - also, call functional form with options
// ------------------------------------------------
OneHotOptions onehot(K,J,Cast);
K onehot(bool,const OneHotOptions&);
Tensor onehot(const Tensor&,const OneHotOptions&);

} // namespace knn
