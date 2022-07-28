#pragma once
#include "util.h"

namespace knn {

struct TORCH_API SqueezeOptions {
 SqueezeOptions(int64_t d,bool b=false) : dim_(d),inplace_(b) {}
 SqueezeOptions() {}
 TORCH_ARG(c10::optional<int64_t>, dim) = c10::nullopt;
 TORCH_ARG(bool, inplace) = false;
};

// -------------------------------------------------------------
//  squeeze - remove dimension(s) from tensor
// -------------------------------------------------------------
class TORCH_API SqueezeImpl : public torch::nn::Cloneable<SqueezeImpl> {
 public:
  SqueezeImpl(int64_t d,bool b=false);
  SqueezeImpl();
  explicit SqueezeImpl(const SqueezeOptions& o);
  void reset() override;
  void pretty_print(std::ostream& s) const override;
  Tensor forward(const Tensor& x);
  SqueezeOptions options;
};
TORCH_MODULE(Squeeze);

// -------------------------------------------------------------
//  unsqueeze - add dimension to tensor
// -------------------------------------------------------------
class TORCH_API UnsqueezeImpl : public torch::nn::Cloneable<UnsqueezeImpl> {
 public:
  UnsqueezeImpl(int64_t d,bool b=false);
  explicit UnsqueezeImpl(const SqueezeOptions& o);
  void reset() override;
  void pretty_print(std::ostream& s) const override;
  Tensor forward(const Tensor& x);
  SqueezeOptions options;
};
TORCH_MODULE(Unsqueeze);

// -----------------------------------------------------------------------------
// squeeze/unsqueeze - set options from k args & retrieve to k dictionary
// -----------------------------------------------------------------------------
SqueezeOptions squeeze(K,J,Cast);
K squeeze(bool,const SqueezeOptions&);

} // namespace knn
