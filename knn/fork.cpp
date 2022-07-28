#include "../ktorch.h"
#include "fork.h"

namespace knn {

// ---------------------------------------------------------------------------------------
// fork - create two branches with AnyModule/Sequential to separately process input tensor
// ---------------------------------------------------------------------------------------
void ForkImpl::reset() {}
void ForkImpl::pretty_print(std::ostream& s) const {s << "knn::Fork";}

void ForkImpl::push_back(std::string s,const torch::nn::AnyModule& m) {
 if(a.is_empty() && qa.is_empty()) 
  a=std::move(m), register_module(s.size() ? s : "a", m.ptr());
 else if(b.is_empty() && qb.is_empty())
  b=std::move(m), register_module(s.size() ? s : "b", m.ptr());
 else
  TORCH_CHECK(false, "fork: cannot add ",mlabel(m.ptr())," module, both left & right forks already defined");
}
 
void ForkImpl::push_back(const torch::nn::AnyModule& m) {push_back(std::string(),m);}

void ForkImpl::push_back(std::string s,const torch::nn::Sequential& q) {
 if(a.is_empty() && qa.is_empty()) 
  qa=register_module(s.size() ? s : "qa", q);
 else if(b.is_empty() && qb.is_empty())
  qb=register_module(s.size() ? s : "qb", q);
 else
  TORCH_CHECK(false, "fork: cannot add ",mlabel(q.ptr())," module, both left & right forks already defined");
}

void ForkImpl::push_back(const torch::nn::Sequential& q) {push_back(std::string(),q);}

Tuple ForkImpl::forward(const Tensor& x) {
 Tensor y=a.is_empty() ? (qa.is_empty() ? x : qa->forward(x)) : a.forward(x);
 Tensor z=b.is_empty() ? (qb.is_empty() ? x : qb->forward(x)) : b.forward(x);
 return std::make_tuple(y,z);
}

} // knn namespace
