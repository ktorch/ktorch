#include "../ktorch.h"
#include "residual.h"

namespace knn {

// ------------------------------------------------------------------------
// residual - add up to two Sequentials and an optional activation function
// ------------------------------------------------------------------------
void ResidualImpl::reset() {}
void ResidualImpl::pretty_print(std::ostream& s) const {s << "knn::Residual";}

Tensor ResidualImpl::forward(const Tensor& x,const Tensor& y,const Tensor& z) {
 TORCH_CHECK(!q1.is_empty(), "residual: no modules defined for forward calculation");
 if(y.defined() && z.defined()) {
  if(q2.is_empty())
   return fn.is_empty() ? q1->forward(x,y,z) + x : fn.forward(q1->forward(x,y,z) + x);
  else
   return fn.is_empty() ? q1->forward(x,y,z) + q2->forward(x,y,z) : fn.forward(q1->forward(x,y,z) + q2->forward(x,y,z));
 } else if(y.defined()) {
  if(q2.is_empty())
   return fn.is_empty() ? q1->forward(x,y) + x : fn.forward(q1->forward(x,y) + x);
  else
   return fn.is_empty() ? q1->forward(x,y) + q2->forward(x,y) : fn.forward(q1->forward(x,y) + q2->forward(x,y));
 } else {
  if(q2.is_empty())
   return fn.is_empty() ? q1->forward(x) + x : fn.forward(q1->forward(x) + x);
  else
   return fn.is_empty() ? q1->forward(x) + q2->forward(x) : fn.forward(q1->forward(x) + q2->forward(x));
 }
}

void ResidualImpl::push_back(std::string s,const torch::nn::Sequential& q) {
 if(q1.is_empty())
  q1=register_module(s.size() ? s : "q1", q);
 else if(q2.is_empty() && fn.is_empty())
  q2=register_module(s.size() ? s : "q2", q);
 else
  TORCH_CHECK(false, "residual: ",
             (q2.is_empty() ? "activation function already defined, cannot add a 2nd sequential module" 
                            : "both sequential modules already defined, cannot add another sequential module"));
}

void ResidualImpl::push_back(const torch::nn::Sequential& q) {
 push_back(std::string(),q);
}

void ResidualImpl::push_back(std::string s,const torch::nn::AnyModule& m) {
 TORCH_CHECK(!q1.is_empty(), "residual: cannot add ", mlabel(m.ptr()), " module until sequential module(s) defined");
 TORCH_CHECK( fn.is_empty(), "residual: cannot add ", mlabel(m.ptr()), " module, activation function already defined");
 fn=std::move(m), register_module(s.size() ? s : "fn", m.ptr());
}

void ResidualImpl::push_back(const torch::nn::AnyModule& m) {
 push_back(std::string(),m);
}

} // namespace knn
