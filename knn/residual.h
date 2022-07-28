#pragma once

#ifdef __clang__
# pragma clang diagnostic push
# pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"    // FORWARD_HAS_DEFAULT_ARGS
#endif
 
namespace knn {

// ------------------------------------------------------------------------
// residual - add up to two Sequentials and an optional activation function
// ------------------------------------------------------------------------
class TORCH_API ResidualImpl : public torch::nn::Cloneable<ResidualImpl> {
 public:
 void reset() override;
 void pretty_print(std::ostream& s) const override;
 Tensor forward(const Tensor& x,const Tensor& y={},const Tensor& z={});
 void push_back(std::string s,const torch::nn::Sequential& q);
 void push_back(const torch::nn::Sequential& q);
 void push_back(std::string s,const torch::nn::AnyModule& m);
 void push_back(const torch::nn::AnyModule& m);
 torch::nn::Sequential q1=nullptr,q2=nullptr;
 torch::nn::AnyModule fn;
 protected:
  FORWARD_HAS_DEFAULT_ARGS({1, torch::nn::AnyValue(Tensor())},
                           {2, torch::nn::AnyValue(Tensor())})
};
TORCH_MODULE(Residual);

} // namespace knn

#ifdef __clang__
# pragma clang diagnostic pop
#endif
