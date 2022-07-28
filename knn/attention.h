#pragma once
#include "util.h"

#ifdef __clang__
# pragma clang diagnostic push
# pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"    // FORWARD_HAS_DEFAULT_ARGS
#endif

namespace knn {

// -------------------------------------------------------------------------
// SelfAttentionOptions - subset of Multihead options, w'optional layer norm
// -------------------------------------------------------------------------
struct TORCH_API SelfAttentionOptions {
 SelfAttentionOptions(int64_t d,int64_t h) : dim_(d),heads_(h) {}
 TORCH_ARG(int64_t, dim);
 TORCH_ARG(int64_t, heads);
 TORCH_ARG(double,  dropout) = 0.0;
 TORCH_ARG(bool,    norm) = false;
};

// -------------------------------------------------------------------------
// SelfAttention - subclass of Multihead attention
// -------------------------------------------------------------------------
class TORCH_API SelfAttentionImpl : public torch::nn::Cloneable<SelfAttentionImpl> {
  public:
  SelfAttentionImpl(const SelfAttentionOptions&);

  void reset() override;
  void pretty_print(std::ostream& s) const override;
  Tensor forward(const Tensor& x,const Tensor& m={},const Tensor& p={});

  SelfAttentionOptions options;
  torch::nn::LayerNorm norm = nullptr;
  torch::nn::Linear in=nullptr;
  torch::nn::Dropout drop;
  torch::nn::Linear out=nullptr;

 protected:
  FORWARD_HAS_DEFAULT_ARGS({1, torch::nn::AnyValue(Tensor())},
                           {2, torch::nn::AnyValue(Tensor())})
};
TORCH_MODULE(SelfAttention);

// -----------------------------------------------------
// attention - set/get options for multi-head attention
// selfattn - set/get options for self attention
// -----------------------------------------------------
torch::nn::MultiheadAttentionOptions attention(K,J,Cast);
K attention(bool,const torch::nn::MultiheadAttentionOptions&);
SelfAttentionOptions selfattn(K,J,Cast);
K selfattn(bool,const SelfAttentionOptions&);

} // namespace knn

#ifdef __clang__
# pragma clang diagnostic pop
#endif
