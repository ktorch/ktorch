#include "../ktorch.h"
#include "squeeze.h"

namespace knn {

// -------------------------------------------------------------
//  squeeze - remove dimension(s) from tensor
// -------------------------------------------------------------
SqueezeImpl::SqueezeImpl(int64_t d,bool b) : SqueezeImpl(SqueezeOptions(d,b)) {}
SqueezeImpl::SqueezeImpl() : SqueezeImpl(SqueezeOptions()) {}
SqueezeImpl::SqueezeImpl(const SqueezeOptions& o) : options(o) {reset();}

void SqueezeImpl::reset() {}
void SqueezeImpl::pretty_print(std::ostream& s) const {
 s << "knn::Squeeze(dim="; options.dim() ? s << options.dim().value() : s << "None";
 s << ", inplace=" << options.inplace() <<")";
}

Tensor SqueezeImpl::forward(const Tensor& x) {
 if(options.dim().has_value()) {
  if(options.inplace())
   return x.squeeze_(options.dim().value());
  else
   return x.squeeze(options.dim().value());
 } else {
  if(options.inplace())
   return x.squeeze_();
  else
   return x.squeeze();
 }
}

// -------------------------------------------------------------
//  unsqueeze - add dimension to tensor
// -------------------------------------------------------------
UnsqueezeImpl::UnsqueezeImpl(int64_t d,bool b) : UnsqueezeImpl(SqueezeOptions(d,b)) {}
UnsqueezeImpl::UnsqueezeImpl(const SqueezeOptions& o) : options(o) {reset();}

void UnsqueezeImpl::reset() {TORCH_CHECK(options.dim().has_value(),"unsqueeze: no dimension given");}

void UnsqueezeImpl::pretty_print(std::ostream& s) const {
 s << "knn::Unsqueeze(dim="; options.dim() ? s << options.dim().value() : s << "None";
 s << ", inplace=" << options.inplace() <<")";
}

Tensor UnsqueezeImpl::forward(const torch::Tensor& x) {
 if(options.inplace())
  return x.unsqueeze_(options.dim().value());
 else
  return x.unsqueeze(options.dim().value());
}


// -----------------------------------------------------------------------------
// squeeze/unsqueeze - set options from k args & retrieve to k dictionary
//                     squeeze works w'optional dimension, unsqueeze requires it
// -----------------------------------------------------------------------------
SqueezeOptions squeeze(K x,J i,Cast c) {
 SqueezeOptions o; Pairs p; bool b; J n=xargc(x,i,p);
 if(n==1) {
  if(xbool(x,i,b))
    o.inplace(b);
  else
    o.dim(int64n(x,i,c,Setting::dim));
 } else if(n==2) {
   o.dim(   int64n(x,i,   c, Setting::dim));
   o.inplace(mbool(x,i+1, c, Setting::inplace));
 } else if(n) {
  TORCH_ERROR(msym(c), ": unrecognized positional arg(s), expecting dim, inplace flag, or (dim;inplace flag)");
 }
 while(xpair(p))
  switch(mset(p.k,c)) {
   case Setting::dim:     o.dim(int64n(p,c)); break;
   case Setting::inplace: o.inplace(mbool(p,c)); break;
   default: mpair(c,p); break;
  }
 TORCH_CHECK(c==Cast::squeeze || o.dim().has_value(), msym(c),": no dimension given");
 return o;
}

K squeeze(bool a,const SqueezeOptions& o) {
 K x=KDICT;
 if(o.dim().has_value()) msetting(x, Setting::dim,     kj(o.dim().value()));
 if(a || o.inplace())    msetting(x, Setting::inplace, kb(o.inplace()));
 return x;
}


} // namespace knn
