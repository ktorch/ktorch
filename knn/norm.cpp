#include "../ktorch.h"
#include "norm.h"

namespace knn {

// -----------------------------------------------------------------------------------
// momentum  - get/set optional double arg or return true if both args match
// batchnorm - get/set batch norm & instance norm options
//             both module types and dimensions(1,2,3d) use the same options structure
//             except batch norm's momentum is an optional double
// -----------------------------------------------------------------------------------
void momentum(torch::nn::BatchNormOptions& o) {
 if(o.momentum() && o.momentum().value() != o.momentum().value())
  o.momentum(c10::nullopt);
}

void   momentum(torch::nn::InstanceNormOptions& o) {}
double momentum(c10::optional<double> x) {return x ? *x : nf;} 
bool   momentum(c10::optional<double> x, c10::optional<double> y) {
 return (x && y) ?  *x == *y : false;
}

// ---------------------------------------------------------------
// groupnorm - group norm, get/set groups,channels,eps,affine flag
// ---------------------------------------------------------------
torch::nn::GroupNormOptions groupnorm(K x,J i,Cast c) {
 torch::nn::GroupNormOptions o(0,0);
 bool g=false,h=false; Pairs p; J n=xargc(x,i,p);
 for(J j=0;j<n;++j)
   switch(j) {
    case 0: o.num_groups(int64(x,i+j,c,Setting::groups)); g=true; break;
    case 1: o.num_channels(int64(x,i+j,c,Setting::channels)); h=true; break;
    case 2: o.eps(mdouble(x,i+j,c,Setting::eps)); break;
    case 3: o.affine(mbool(x,i+j,c,Setting::affine)); break;
    default: mpos(x,c,i+j); break;
   }
 while(xpair(p))
  switch(mset(p.k,c)) {
   case Setting::groups:   o.num_groups(int64(p,c)); g=true; break;
   case Setting::channels: o.num_channels(int64(p,c)); h=true; break;
   case Setting::eps:      o.eps(mdouble(p,c)); break;
   case Setting::affine:   o.affine(mbool(p,c)); break;
   default: mpair(c,p); break;
  }
 TORCH_CHECK(g, msym(c),": specify no. of groups to separate the channels into");
 TORCH_CHECK(h, msym(c),": specify no. of channels expected in input");
 return o;
}

K groupnorm(bool a,const torch::nn::GroupNormOptions& o) {
 K x=KDICT; torch::nn::GroupNormOptions d(o.num_groups(),o.num_channels());
 msetting(x, Setting::groups,   kj(o.num_groups()));
 msetting(x, Setting::channels, kj(o.num_channels()));
 if(a || (o.eps()    != d.eps()))    msetting(x, Setting::eps,    kf(o.eps()));
 if(a || (o.affine() != d.affine())) msetting(x, Setting::affine, kb(o.affine()));
 return x;
}

// -----------------------------------------------------------------
// layernorm - get/set shape,eps,affine flag for layer normalization
// -----------------------------------------------------------------
torch::nn::LayerNormOptions layernorm(K x,J i,Cast c) {
 torch::nn::LayerNormOptions o({}); Pairs p; J n=xargc(x,i,p);
 for(J j=0;j<n;++j)
   switch(j) {
    case 0: o.normalized_shape(mlongs(x,i+j,c,Setting::shape)); break;
    case 1: o.eps(mdouble(x,i+j,c,Setting::eps)); break;
    case 2: o.elementwise_affine(mbool(x,i+j,c,Setting::affine)); break;
    default: mpos(x,c,i+j); break;
   }
 while(xpair(p))
  switch(mset(p.k,c)) {
   case Setting::shape:  o.normalized_shape(mlongs(p,c)); break;
   case Setting::eps:    o.eps(mdouble(p,c)); break;
   case Setting::affine: o.elementwise_affine(mbool(p,c)); break;
   default: mpair(c,p); break;
  }
 TORCH_CHECK(o.normalized_shape().size(), msym(c),": no normalized shape given");
 return o;
}

K layernorm(bool a,const torch::nn::LayerNormOptions& o) {
 K x=KDICT; torch::nn::LayerNormOptions d(o.normalized_shape());
 msetting(x, Setting::shape, klist(o.normalized_shape().size(),o.normalized_shape().data()));
 if(a || (o.eps()    != d.eps())) msetting(x, Setting::eps, kf(o.eps()));
 if(a || (o.elementwise_affine() != d.elementwise_affine())) msetting(x, Setting::affine, kb(o.elementwise_affine()));
 return x;
}

// --------------------------------------------------------------------------
// normalize - pytorch has functional form only, no module as of version 1.10
//             functions to set/get functional options
// --------------------------------------------------------------------------
torch::nn::functional::NormalizeFuncOptions normalize(K x,J i,Cast c,Tensor& r) {
 Pairs p; J n=xargc(x,i,p); torch::nn::functional::NormalizeFuncOptions o;
 if(n>0 && xten(x,i+n-1,r)) n--;
 for(J j=0;j<n;++j)
  switch(j) {
   case 0: o.p(mdouble(x,i+j,c,Setting::p)); break;
   case 1: o.dim(int64(x,i+j,c,Setting::dim)); break;
   case 2: o.eps(mdouble(x,i+j,c,Setting::eps)); break;
   default: mpos(x,c,i+j); break;
  }
 while(xpair(p))
  switch(mset(p.k,c)) {
   case Setting::p: o.p(mdouble(p,c)); break;
   case Setting::dim: o.dim(int64(p,c)); break;
   case Setting::eps: o.eps(mdouble(p,c)); break;
   case Setting::out: if(!pempty(p)) pten(p,r);
   default: mpair(c,p); break;
  }
 if(r.defined())
  o.out(r);
 return o;
}

K normalize(bool a,const torch::nn::functional::NormalizeFuncOptions& o) {
 K x=KDICT; const torch::nn::functional::NormalizeFuncOptions d;
 if(a || o.p()   != d.p())   msetting(x, Setting::p, kf(o.p()));
 if(a || o.dim() != d.dim()) msetting(x, Setting::dim, kj(o.dim()));
 if(a || o.eps() != d.eps()) msetting(x, Setting::eps, kj(o.eps()));
 return x;
}

} // namespace knn
