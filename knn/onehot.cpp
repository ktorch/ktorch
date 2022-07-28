#include "../ktorch.h"
#include "onehot.h"

namespace knn {

// ------------------------------------------------------------------
// OneHot - module for torch.nn.functional.one_hot(tensor,numclasses)
// ------------------------------------------------------------------
OneHotImpl::OneHotImpl(const OneHotOptions& o) : options(o) {reset();}

void OneHotImpl::reset() {}

void OneHotImpl::pretty_print(std::ostream& s) const {
 s << "knn::OneHot(num_classes=" << options.num_classes() << ")";
}

Tensor OneHotImpl::forward(const Tensor& x) {
 return torch::one_hot(x,options.num_classes()).to(options.dtype() ? options.dtype().value() : torch::kFloat);
}

// ------------------------------------------------
// onehot - get/set number of classes & data type 
//        - also, call functional form with options
// ------------------------------------------------
OneHotOptions onehot(K x,J i,Cast c) { 
 OneHotOptions o; Pairs p; J n=xargc(x,i,p);
 for(J j=0;j<n;++j)
  switch(j) {
   case 0: o.num_classes(int64(x,i+j,c,Setting::classes)); break;
   case 1: o.dtype(otype(x,i+j,c,Setting::dtype)); break;
   default: mpos(x,c,i+j); break;
  }
 while(xpair(p))
  switch(mset(p.k,c)) {
   case Setting::classes: o.num_classes(int64(p,c)); break;
   case Setting::dtype:   o.dtype(otype(p,c)); break;
   default: mpair(c,p); break;
  }
 TORCH_CHECK(o.num_classes()>-2, msym(c),": number of classes must be nonnegative or set to -1 to derive from input");
 return o;
}

K onehot(bool a,const OneHotOptions& o) {
 K x=KDICT;
 msetting(x, Setting::classes, kj(o.num_classes()));
 if(o.dtype()) msetting(x, Setting::dtype, ks(stype(o.dtype().value())));
 return x;
}

Tensor onehot(const Tensor& t,const OneHotOptions& o) {
 return torch::one_hot(t,o.num_classes()).to(o.dtype() ? o.dtype().value() : torch::kFloat);
}

} // namespace knn
