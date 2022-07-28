#pragma once
#include "util.h"
  
namespace knn {

// -------------------------------------------------------
// select - module for tensor.select(dim,ind)
// -------------------------------------------------------
struct TORCH_API SelectOptions {
 SelectOptions(int64_t d,int64_t i) : dim_(d),ind_(i) {}
 TORCH_ARG(int64_t, dim);
 TORCH_ARG(int64_t, ind);
};

class TORCH_API SelectImpl : public torch::nn::Cloneable<SelectImpl> {
 public:
 SelectImpl(const SelectOptions& o);
 void reset() override;
 void pretty_print(std::ostream& s) const override;
 Tensor forward(const Tensor& x);

 SelectOptions options;
};
TORCH_MODULE(Select);

// -------------------------------------------------------
// indexselect - module for tensor.index_select(,dim,ind)
// -------------------------------------------------------
struct TORCH_API IndexSelectOptions {
 IndexSelectOptions(int64_t d,int64_t i) : dim_(d),ind_(torch::full({},torch::Scalar(i))) {}
 IndexSelectOptions(int64_t d,Tensor i) : dim_(d),ind_(i) {}
 TORCH_ARG(int64_t, dim);
 TORCH_ARG(Tensor, ind);
};

class TORCH_API IndexSelectImpl : public torch::nn::Cloneable<IndexSelectImpl> {
 public:
 IndexSelectImpl(const IndexSelectOptions& o);

 void reset() override;
 void pretty_print(std::ostream& s) const override;
 Tensor forward(const Tensor& x);

 IndexSelectOptions options;
 Tensor ind;
};
TORCH_MODULE(IndexSelect);

// ----------------------------------------------------------------
// select - get/set dim & scalar index for Select module
// ----------------------------------------------------------------
SelectOptions select(K,J,Cast);
K select(bool,const SelectOptions&);

// ----------------------------------------------------------------
// indexselect - get/set dim & tensor index for IndexSelect module
// ----------------------------------------------------------------
IndexSelectOptions indexselect(K,J,Cast);
K indexselect(bool,const IndexSelectOptions&);

} // namespace knn
