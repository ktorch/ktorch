#pragma once
#include "util.h"

namespace knn {

// -----------------------------------------------------------------------------------
// momentum  - get/set optional double arg or return true if both args match
// batchnorm - get/set batch norm & instance norm options
//             both module types and dimensions(1,2,3d) use the same options structure
//             except batch norm's momentum is an optional double
// -----------------------------------------------------------------------------------
void   momentum(torch::nn::BatchNormOptions&);
void   momentum(torch::nn::InstanceNormOptions&);
double momentum(c10::optional<double> x);
bool   momentum(c10::optional<double>,c10::optional<double>);

template<typename O> O batchnorm(K x,J i,Cast c) {
 O o(0);
 bool in=false; Pairs p; J n=xargc(x,i,p);
 for(J j=0;j<n;++j)
   switch(j) {
    case 0: o.num_features(int64(x,i+j,c,Setting::in)); in=true; break;
    case 1: o.eps(mdouble(x,i+j,c,Setting::eps));break;
    case 2: o.momentum(mdouble(x,i+j,c,Setting::momentum)); break;
    case 3: o.affine(mbool(x,i+j,c,Setting::affine)); break;
    case 4: o.track_running_stats(mbool(x,i+j,c,Setting::affine)); break;
    default: mpos(x,c,i+j); break;
   }
 while(xpair(p))
  switch(mset(p.k,c)) {
   case Setting::in:       o.num_features(int64(p,c)); in=true; break;
   case Setting::eps:      o.eps(mdouble(p,c)); break;
   case Setting::momentum: o.momentum(mdouble(p,c)); break;
   case Setting::affine:   o.affine(mbool(p,c)); break;
   case Setting::track:    o.track_running_stats(mbool(p,c)); break;
   default: mpair(c,p); break;
  }
 TORCH_CHECK(in,msym(c),": number of input features not defined");
 momentum(o);
 return o;
}

template<typename O> K batchnorm(bool a,const O& o) {
 K x=KDICT; O d(o.num_features());
 msetting(x, Setting::in, kj(o.num_features()));
 if(a || (o.eps()      != d.eps()))            msetting(x, Setting::eps,       kf(o.eps()));
 if(a || !momentum(o.momentum(),d.momentum())) msetting(x, Setting::momentum,  kf(momentum(o.momentum())));
 if(a || (o.affine()   != d.affine()))         msetting(x, Setting::affine,    kb(o.affine()));
 if(a || (o.track_running_stats() != d.track_running_stats())) msetting(x, Setting::track, kb(o.track_running_stats()));
 return resolvedict(x);
}

// -------------------------------------------------------------------------------------
// localnorm - local response norm, cross map 2d norm, get/set options size,alpha,beta,k
// -------------------------------------------------------------------------------------
template<typename O> O localnorm(K x,J i,Cast c) {
 O o(0);
 bool b=c==Cast::localnorm,sz=false; Pairs p; J n=xargc(x,i,p);
 for(J j=0;j<n;++j)
   switch(j) {
    case 0: o.size(int64(x,i+j,c,Setting::size)); sz=true; break;
    case 1: o.alpha(mdouble(x,i+j,c,Setting::alpha)); break;
    case 2: o.beta(mdouble(x,i+j,c,Setting::beta)); break;
    case 3: b ? o.k(mdouble(x,i+j,c,Setting::k)) : o.k(int64(x,i+j,c,Setting::k)); break;
    default: mpos(x,c,i+j); break;
   }
 while(xpair(p))
  switch(mset(p.k,c)) {
   case Setting::size:  o.size(int64(p,c)); sz=true; break;
   case Setting::alpha: o.alpha(mdouble(p,c)); break;
   case Setting::beta:  o.beta(mdouble(p,c)); break;
   case Setting::k:     b ? o.k(mdouble(p,c)) : o.k(int64(p,c)); break;
   default: mpair(c,p); break;
  }
 TORCH_CHECK(sz, msym(c),": specify no. of neighboring channels to use for normalization");
 return o;
}

template<typename O> K localnorm(bool a,Cast c,const O& o) {
 K x=KDICT; O d(o.size());
 msetting(x, Setting::size, kj(o.size()));
 if(a || (o.alpha() != d.alpha())) msetting(x, Setting::alpha, kf(o.alpha()));
 if(a || (o.beta()  != d.beta()))  msetting(x, Setting::beta,  kf(o.beta()));
 if(a || (o.k()     != d.k()))     msetting(x, Setting::k,     c==Cast::localnorm ? kf(o.k()) : kj(o.k()));
 return resolvedict(x);
}

// ---------------------------------------------------------------
// groupnorm - group norm, get/set groups,channels,eps,affine flag
// ---------------------------------------------------------------
torch::nn::GroupNormOptions groupnorm(K,J,Cast);
K groupnorm(bool,const torch::nn::GroupNormOptions&);

// -----------------------------------------------------------------
// layernorm - get/set shape,eps,affine flag for layer normalization
// -----------------------------------------------------------------
torch::nn::LayerNormOptions layernorm(K,J,Cast);
K layernorm(bool,const torch::nn::LayerNormOptions&);

// --------------------------------------------------------------------------
// normalize - pytorch has functional form only, no module as of version 1.10
//             functions to set/get functional options
// --------------------------------------------------------------------------
torch::nn::functional::NormalizeFuncOptions normalize(K,J,Cast,Tensor&);
K normalize(bool,const torch::nn::functional::NormalizeFuncOptions&);

} // namespace knn
