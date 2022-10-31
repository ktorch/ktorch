#pragma once
#include "util.h"

namespace knn {

// --------------------------------------------------------------------------
// general pad: create module to match functional call with size, mode, value
// --------------------------------------------------------------------------
class TORCH_API PadImpl : public torch::nn::Cloneable<PadImpl> {
 public:
  PadImpl(std::vector<int64_t> p);
  explicit PadImpl(const torch::nn::functional::PadFuncOptions& o);
  Tensor forward(const Tensor& x);
  void reset() override;
  void pretty_print(std::ostream& s) const override;
  torch::nn::functional::PadFuncOptions options;
};

TORCH_MODULE(Pad);

// -------------------------------------------------------------------------------
// pad - parse k args into options, retrieve module options back into k dictionary
// -------------------------------------------------------------------------------
torch::nn::functional::PadFuncOptions pad(K,J,Cast);
K pad(bool,const torch::nn::functional::PadFuncOptions&);

// ---------------------------------------------------------------------------
// cpad - constant pad w'fixed dimension and optional value (defaults to zero)
// ---------------------------------------------------------------------------
template<size_t D,typename M> M cpad(K x,J i,Cast c) {
 M o(0,0);
 bool sz=false; Pairs p; J n=xargc(x,i,p);
 for(J j=0;j<n;++j)
   switch(j) {
    case 0: o.padding(exarray<D*2>(x,i+j,c,Setting::pad)); sz=true; break;
    case 1: o.value(mdouble(x,i+j,c,Setting::value)); break;
    default: mpos(x,c,i+j); break;
   }
 while(xpair(p))
  switch(mset(p.k,c)) {
   case Setting::pad: o.padding(exarray<D*2>(p,c)); sz=true; break;
   case Setting::value: o.value(mdouble(p,c)); break;
   default: mpair(c,p); break;
  }
 TORCH_CHECK(sz, msym(c),": no padding sizes given");
 return o;
}

template<typename O> K cpad(const O& o) {
 K x=KDICT;
 msetting(x, Setting::pad, KEX(o.padding()));
 msetting(x, Setting::value, kf(o.value()));
 return resolvedict(x);
}

// ----------------------------------------------------------------------------------
// npad - reflect/replicate/zero pad w'fixed dimension
// ----------------------------------------------------------------------------------
template<size_t D,typename M> M npad(K x,J i,Cast c) {
 M o(0);
 bool sz=false; Pairs p; J n=xargc(x,i,p);
 for(J j=0;j<n;++j)
   switch(j) {
    case 0: o.padding(exarray<D*2>(x,i+j,c,Setting::pad)); sz=true; break;
    default: TORCH_ERROR(msym(c),": only 1 positional argument expected, ",n," given");
   }
 while(xpair(p))
  switch(mset(p.k,c)) {
   case Setting::pad: o.padding(exarray<D*2>(p,c)); sz=true; break;
   default: mpair(c,p); break;
  }
 TORCH_CHECK(sz, msym(c),": no padding sizes given");
 return o;
}

template<typename O> K npad(const O& o) {
 K x=KDICT;
 msetting(x, Setting::pad, KEX(o.padding()));
 return resolvedict(x);
}

} //namespace knn
