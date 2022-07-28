#pragma once
#include "util.h"
  
namespace knn {

// -------------------------------------------------------------
// expand,reshape & permute - modules with size options
// -------------------------------------------------------------
struct TORCH_API SizeOptions {
 SizeOptions(std::vector<int64_t> s) : size_(std::move(s)) {}
 TORCH_ARG(std::vector<int64_t>, size);
};

class TORCH_API ExpandImpl : public torch::nn::Cloneable<ExpandImpl> {
 public:
 ExpandImpl(std::vector<int64_t> s);
 explicit ExpandImpl(const SizeOptions& o);
 void reset() override;
 void pretty_print(std::ostream& s) const override;
 Tensor forward(const Tensor& x);
 SizeOptions options;
};
TORCH_MODULE(Expand);

class TORCH_API ReshapeImpl : public torch::nn::Cloneable<ReshapeImpl> {
 public:
 ReshapeImpl(std::vector<int64_t> s);
 explicit ReshapeImpl(const SizeOptions& o);
 void reset() override;
 void pretty_print(std::ostream& s) const override;
 Tensor forward(const Tensor& x);
 SizeOptions options;
};
TORCH_MODULE(Reshape);

class TORCH_API PermuteImpl : public torch::nn::Cloneable<PermuteImpl> {
 public:
 PermuteImpl(std::vector<int64_t> s);
 explicit PermuteImpl(const SizeOptions& o);
 void reset() override;
 void pretty_print(std::ostream& s) const override;
 Tensor forward(const Tensor& x);
 SizeOptions options;
};
TORCH_MODULE(Permute);

// ----------------------------------------------
// transpose - transpose given 2 dimensions
// ----------------------------------------------
struct TORCH_API TransposeOptions {
  TransposeOptions(int64_t x=-2,int64_t y=-1) : dim0_(x),dim1_(y) {}
  TORCH_ARG(int64_t, dim0);
  TORCH_ARG(int64_t, dim1);
};

class TORCH_API TransposeImpl : public torch::nn::Cloneable<TransposeImpl> {
 public:
 TransposeImpl(const TransposeOptions& o);
 void reset() override;
 void pretty_print(std::ostream& s) const override;
 Tensor forward(const Tensor& x);
 TransposeOptions options;
};
TORCH_MODULE(Transpose);

// ----------------------------------------------
// getsize - set/get size(s) for expand & reshape
// ----------------------------------------------
SizeOptions getsize(K,J,Cast);
K getsize(bool,const SizeOptions&);

// ----------------------------------------
// flatten - get/set start & end dimensions
// transpose - get/set dim0,dim1
// ----------------------------------------
torch::nn::FlattenOptions flatten(K x,J i,Cast c,bool f=false);
K flatten(bool,const torch::nn::FlattenOptions&);
TransposeOptions transpose(K x,J i,Cast c);
K transpose(bool,const TransposeOptions&);

} // namespace knn
