#pragma once
#include "util.h"

namespace knn {

// ---------------------------------------------
// droppath - drop paths per sample
// ---------------------------------------------
struct TORCH_API DropPathOptions {
 DropPathOptions(double x=0) : p_(x) {}
 TORCH_ARG(double, p);
};

class TORCH_API DropPathImpl : public torch::nn::Cloneable<DropPathImpl> {
 public:
 DropPathImpl(const DropPathOptions& o);
 void reset() override;
 void pretty_print(std::ostream& s) const override;
 Tensor forward(const Tensor& x);
 DropPathOptions options;
};
TORCH_MODULE(DropPath);

// -------------------------------------------------
// drop - set/get dropout probability & inplace flag
// droppath - set/get dropout probability only
// -------------------------------------------------
torch::nn::DropoutOptions drop(K,J,Cast);
K drop(bool,const torch::nn::DropoutOptions&);
DropPathOptions droppath(K,J,Cast);
K droppath(bool,const DropPathOptions&);

} // namespace knn
