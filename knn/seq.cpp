#include "../ktorch.h"
#include "seq.h"

namespace knn {

// --------------------------------------------------------------------------------------------------
// SeqJoin - define sequential modules for inputs x & y w'layer for joining the output of each module
// --------------------------------------------------------------------------------------------------
void SeqJoinImpl::reset() {}
void SeqJoinImpl::pretty_print(std::ostream& s) const {s << "knn::SeqJoin";}

void SeqJoinImpl::push_back(const torch::nn::Sequential& q) {
 push_back(children().size()==0 ? "qx" : "qy", q);
}

void SeqJoinImpl::push_back(std::string s,const torch::nn::Sequential& q) {
 TORCH_CHECK(children().size()<2, "seqjoin: both sequential layers already defined");
 if(children().size()==0)
  qx=register_module(s,std::move(q));
 else
  qy=register_module(s,std::move(q));
}

void SeqJoinImpl::push_back(const torch::nn::AnyModule& a) {
 push_back("join",a);
}

void SeqJoinImpl::push_back(std::string s,const torch::nn::AnyModule& a) {
 TORCH_CHECK(children().size(), "seqjoin: at least one sequential layer must be defined first");
 TORCH_CHECK(join.is_empty(), "seqjoin: join layer already defined");
 join=std::move(a);
 register_module(s,join.ptr());
}

Tensor SeqJoinImpl::forward(const Tensor& x,const Tensor& y) {
 TORCH_CHECK(!join.is_empty(), "seqjoin: join layer not defined");
 return join.forward(qx.is_empty() || !qx->children().size() ? x : qx->forward(x),
                     qy.is_empty() || !qy->children().size() ? (y.defined() ? y : x) : qy->forward(y.defined() ? y : x));
}

} // knn namespace
