#pragma once
#include "util.h"

#ifdef __clang__
# pragma clang diagnostic push
# pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"    // FORWARD_HAS_DEFAULT_ARGS
#endif

namespace knn {

// -----------------------------------------------------------------------------
// SeqDict - a ModuleDict of Sequential modules that takes up to 8 tensors
//           tensor output, updated by each sequential in turn
// -----------------------------------------------------------------------------
class TORCH_API SeqDictImpl : public torch::nn::ModuleDictImpl {
 public:
  using ModuleDictImpl::ModuleDictImpl;

  void reset() override {}

  void pretty_print(std::ostream& stream) const override {
    stream << "knn::SeqDict";
  }

  Tensor forward(const Tensor& x0,    const Tensor& x1={}, const Tensor& x2={}, const Tensor& x3={},
                 const Tensor& x4={}, const Tensor& x5={}, const Tensor& x6={}, const Tensor& x7={}) {

   Tensor x = x0;
   size_t n = x7.defined() ? 8 : (x6.defined() ? 7 : (x5.defined() ? 6 : (x4.defined() ? 5 : 
             (x3.defined() ? 4 : (x2.defined() ? 3 : (x1.defined() ? 2 : 1))))));

   for(const auto& p:items()) {
    const auto& q=p.second->as<torch::nn::SequentialImpl>();
    TORCH_CHECK(q, "seqdict: key `", p.first, " is an unexpected type: ",mlabel(p.second));
    if(q->size()) {
     auto m=knn::maxargs(*q->begin(),"seqdict");
     switch(n>m ? m : n){
      case 1: x=q->forward(x); break;
      case 2: x=q->forward(x,x1); break;
      case 3: x=q->forward(x,x1,x2); break;
      case 4: x=q->forward(x,x1,x2,x3); break;
      case 5: x=q->forward(x,x1,x2,x3,x4); break;
      case 6: x=q->forward(x,x1,x2,x3,x4,x5); break;
      case 7: x=q->forward(x,x1,x2,x3,x4,x5,x6); break;
      case 8: x=q->forward(x,x1,x2,x3,x4,x5,x6,x7); break;
     }
    }
   }
   return x;
  }

 protected:
  FORWARD_HAS_DEFAULT_ARGS({1, torch::nn::AnyValue(Tensor())},
                           {2, torch::nn::AnyValue(Tensor())},
                           {3, torch::nn::AnyValue(Tensor())},
                           {4, torch::nn::AnyValue(Tensor())},
                           {5, torch::nn::AnyValue(Tensor())},
                           {6, torch::nn::AnyValue(Tensor())},
                           {7, torch::nn::AnyValue(Tensor())})
};
TORCH_MODULE(SeqDict);

} // knn namespace

#ifdef __clang__
# pragma clang diagnostic pop
#endif
