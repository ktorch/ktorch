#include "../ktorch.h"
#include "drop.h"

namespace knn {

// ---------------------------------------------
// droppath - drop paths per sample
// ---------------------------------------------
DropPathImpl::DropPathImpl(const DropPathOptions& o) : options(o) {reset();}

void DropPathImpl::reset() {}
void DropPathImpl::pretty_print(std::ostream& s) const {
 s << "knn::DropPath(p=" << options.p() << ")";
}
Tensor DropPathImpl::forward(const Tensor& x) {
 if(options.p()==0 || !is_training())
  return x;
 auto k=1.0 - options.p();  //keep probability
 auto s=x.sizes().vec(); size_t i=0;
 for(auto& a:s)
  if(0<i++) a=1;
 auto y = k + torch::rand(s,TensorOptions().dtype(x.dtype()).device(x.device()));
 return x.div(k) * y.floor();
}

// -------------------------------------------------
// drop - set/get dropout probability & inplace flag
// droppath - set/get dropout probability only
// -------------------------------------------------
torch::nn::DropoutOptions drop(K x,J i,Cast c) {
 torch::nn::DropoutOptions o; Pairs p; J n=xargc(x,i,p);
 for(J j=0;j<n;++j)
   switch(j) {
    case 0: o.p(mdouble(x,i+j,c,Setting::p)); break;
    case 1: o.inplace(mbool(x,i+j,c,Setting::inplace)); break;
    default: mpos(x,c,i+j); break;
   }
 while(xpair(p))
  switch(mset(p.k,c)) {
   case Setting::p: o.p(mdouble(p,c)); break;
   case Setting::inplace: o.inplace(mbool(p,c)); break;
   default: mpair(c,p); break;
  }
 return o;
}

K drop(bool a,const torch::nn::DropoutOptions& o) {
 K x=KDICT; torch::nn::DropoutOptions d;
 if(a || o.p()       != d.p())       msetting(x, Setting::p,       kf(o.p()));
 if(a || o.inplace() != d.inplace()) msetting(x, Setting::inplace, kb(o.inplace()));
 return resolvedict(x);
}

DropPathOptions droppath(K x,J i,Cast c) {
 DropPathOptions o; Pairs p; J n=xargc(x,i,p);
 for(J j=0;j<n;++j)
   switch(j) {
    case 0: o.p(mdouble(x,i+j,c,Setting::p)); break;
    default: mpos(x,c,i+j); break;
   }
 while(xpair(p))
  switch(mset(p.k,c)) {
   case Setting::p: o.p(mdouble(p,c)); break;
   default: mpair(c,p); break;
  }
 return o;
}

K droppath(bool a,const DropPathOptions& o) {
 K x=KDICT; msetting(x, Setting::p,kf(o.p()));
 return resolvedict(x);
}

} // namespace knn
