#pragma once
  
namespace knn {

// ---------------------------------------------
// cat - convenience module for cat(tensors,dim)
// ---------------------------------------------
struct TORCH_API CatOptions {
 CatOptions(int64_t d=0) : dim_(d) {}
 TORCH_ARG(int64_t, dim);
};

class TORCH_API CatImpl : public torch::nn::Cloneable<CatImpl> {
 public:
 CatImpl(const CatOptions& o);
 void reset() override;
 void pretty_print(std::ostream& s) const override;
 Tensor forward(const Tensor& x,const Tensor& y);
 CatOptions options;
};
TORCH_MODULE(Cat);

// ----------------------------------------------------
// mul - convenience module for multiplying two tensors
// ----------------------------------------------------
class TORCH_API MulImpl : public torch::nn::Cloneable<MulImpl> {
 public:
 void reset() override;
 void pretty_print(std::ostream& s) const override;
 Tensor forward(const Tensor& x,const Tensor& y);
};
TORCH_MODULE(Mul);

// ----------------------------------------------------
// matmul - convenience module for multiplying matrices
// ----------------------------------------------------
class TORCH_API MatmulImpl : public torch::nn::Cloneable<MatmulImpl> {
 public:
 void reset() override;
 void pretty_print(std::ostream& s) const override;
 Tensor forward(const Tensor& x,const Tensor& y);
};
TORCH_MODULE(Matmul);

} // namespace knn
