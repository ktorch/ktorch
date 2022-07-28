#pragma once

#ifdef __clang__
# pragma clang diagnostic push
# pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"    // FORWARD_HAS_DEFAULT_ARGS
#endif

namespace knn {

// ----------------------------------------------------------------------------------
// SeqNest - derived from Sequential to allow nested sequentials 
//         - no templatized forward result means can be stored as an AnyModule
//         - forward method accepts up to three tensors x,y,z (y & z optional)
//           forward result is tensor only
// ---------------------------------------------------------------------------------
class TORCH_API SeqNestImpl : public torch::nn::SequentialImpl {
  public:
  using SequentialImpl::SequentialImpl;

  void pretty_print(std::ostream& stream) const override {
    stream << "knn::SeqNest";
  }

  Tensor forward(const Tensor& x,const Tensor& y={},const Tensor& z={}) {
   if(y.defined())
    return z.defined() ? SequentialImpl::forward(x,y,z) : SequentialImpl::forward(x,y);
   else
    return SequentialImpl::forward(x);
  }
 protected:
  FORWARD_HAS_DEFAULT_ARGS({1, torch::nn::AnyValue(Tensor())},
                           {2, torch::nn::AnyValue(Tensor())})
};
TORCH_MODULE(SeqNest);


// --------------------------------------------------------------------------------------------------
// SeqJoin - define sequential modules for inputs x & y w'layer for joining the output of each module
// --------------------------------------------------------------------------------------------------
class TORCH_API SeqJoinImpl : public torch::nn::Cloneable<SeqJoinImpl> {
 public:
 SeqJoinImpl() = default;
 void reset() override;
 void pretty_print(std::ostream& s) const override;
 void push_back(const torch::nn::Sequential& q);
 void push_back(std::string s,const torch::nn::Sequential& q);
 void push_back(const torch::nn::AnyModule& a);
 void push_back(std::string s,const torch::nn::AnyModule& a);
 Tensor forward(const Tensor& x,const Tensor& y={});
 
 torch::nn::Sequential qx = nullptr;
 torch::nn::Sequential qy = nullptr;
 torch::nn::AnyModule  join;
 protected:
  FORWARD_HAS_DEFAULT_ARGS({1, torch::nn::AnyValue(Tensor())})
};
TORCH_MODULE(SeqJoin);

} // knn namespace

#ifdef __clang__
# pragma clang diagnostic pop
#endif
