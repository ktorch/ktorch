#include "../ktorch.h"
#include "linear.h"

namespace knn {

// ----------------------------------------------------------
// linear - parse/retrieve args for torch::nn::Linear module
// ----------------------------------------------------------
torch::nn::LinearOptions linear(K x,J i,Cast c) {
 bool b=true; int64_t in=nj,out=nj; Pairs p; J n=xargc(x,i,p);
 for(J j=0;j<n;++j) {
   switch(j) {
    case 0:  in=int64(x,i+j,c,Setting::in);   break;
    case 1: out=int64(x,i+j,c,Setting::out);  break;
    case 2:   b=mbool(x,i+j,c,Setting::bias); break;
    default: mpos(x,c,i+j); break;
  }
 }
 while(xpair(p))
  switch(mset(p.k,c)) {
   case Setting::in:   in=int64(p,c); break;
   case Setting::out:  out=int64(p,c); break;
   case Setting::bias: b=mbool(p,c); break;
   default: mpair(c,p); break;
  }
 TORCH_CHECK(in>0,  msym(c), ": positive input size required");
 TORCH_CHECK(out>0, msym(c), ": positive output size required");
 return torch::nn::LinearOptions(in,out).bias(b);
}

K linear(bool a,const torch::nn::LinearOptions& o) {
 K x=KDICT; torch::nn::LinearOptions d(o.in_features(),o.out_features());
 msetting(x, Setting::in,  kj(o.in_features()));
 msetting(x, Setting::out, kj(o.out_features()));
 if(a || (o.bias() != d.bias())) msetting(x, Setting::bias, kb(o.bias()));
 return resolvedict(x);
}

// --------------------------------------------------------------
// bilinear - parse/retrieve args for torch::nn::Bilinear module
// --------------------------------------------------------------
torch::nn::BilinearOptions bilinear(K x,J i,Cast c) {
 bool b=true; int64_t in1=nj,in2=nj,out=nj; Pairs p; J n=xargc(x,i,p);
 for(J j=0;j<n;++j) {
   switch(j) {
    case 0: in1=int64(x,i+j,c,Setting::in1);   break;
    case 1: in2=int64(x,i+j,c,Setting::in2);   break;
    case 2: out=int64(x,i+j,c,Setting::out);  break;
    case 3:   b=mbool(x,i+j,c,Setting::bias); break;
    default: mpos(x,c,i+j); break;
   }
 }
 while(xpair(p))
  switch(mset(p.k,c)) {
   case Setting::in1:  in1=int64(p,c); break;
   case Setting::in2:  in2=int64(p,c); break;
   case Setting::out:  out=int64(p,c); break;
   case Setting::bias: b=mbool(p,c); break;
   default: mpair(c,p); break;
  }
 TORCH_CHECK(in1>0 && in2>0, msym(c), ": positive input sizes required");
 TORCH_CHECK(out>0,          msym(c), ": positive output size required");
 return torch::nn::BilinearOptions(in1,in2,out).bias(b);
}

K bilinear(bool a,const torch::nn::BilinearOptions& o) {
 K x=KDICT; torch::nn::BilinearOptions d(o.in1_features(),o.in2_features(),o.out_features());
 msetting(x, Setting::in1,  kj(o.in1_features()));
 msetting(x, Setting::in2,  kj(o.in2_features()));
 msetting(x, Setting::out, kj(o.out_features()));
 if(a || (o.bias() != d.bias())) msetting(x, Setting::bias, kb(o.bias()));
 return resolvedict(x);
}

} // namespace knn

