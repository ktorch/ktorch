#include "../ktorch.h"
#include "fns.h"

namespace knn {

// ---------------------------------------------
// cat - convenience module for cat(tensors,dim)
// ---------------------------------------------
CatImpl::CatImpl(const CatOptions& o) : options(o) {reset();}

void CatImpl::reset() {}
void CatImpl::pretty_print(std::ostream& s) const {
 s << "knn::Cat(dim=" << options.dim() << ")";
}
Tensor CatImpl::forward(const Tensor& x,const Tensor& y) {
 return torch::cat({x,y},options.dim());
}

// ----------------------------------------------------
// mul - convenience module for multiplying two tensors
// ----------------------------------------------------
void MulImpl::reset() {}
void MulImpl::pretty_print(std::ostream& s) const {s << "knn::Mul()";}
Tensor MulImpl::forward(const Tensor& x,const Tensor& y) {
  return torch::mul(x,y);
}

// ------------------------------------------------------
// matmul - convenience module for matrix multiplication
// ------------------------------------------------------
void MatmulImpl::reset() {}
void MatmulImpl::pretty_print(std::ostream& s) const {s << "knn::Matmul()";}
Tensor MatmulImpl::forward(const Tensor& x,const Tensor& y) {
  return x.matmul(y);
}

} // namespace knn
