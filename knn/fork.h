#pragma once

namespace knn {

// ---------------------------------------------------------------------------------------
// fork - create two branches with AnyModule/Sequential to separately process input tensor
// ---------------------------------------------------------------------------------------
class TORCH_API ForkImpl : public torch::nn::Cloneable<ForkImpl> {
 public:
 void reset() override;
 void pretty_print(std::ostream& s) const override;

 void push_back(std::string s,const torch::nn::AnyModule& m);
 void push_back(const torch::nn::AnyModule& m);

 void push_back(std::string s,const torch::nn::Sequential& q);
 void push_back(const torch::nn::Sequential& q);

 Tuple forward(const Tensor& x);

 torch::nn::AnyModule a,b;
 torch::nn::Sequential qa=nullptr,qb=nullptr;
};
TORCH_MODULE(Fork);

} // knn namespace
