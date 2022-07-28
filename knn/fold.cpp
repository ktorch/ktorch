#include "../ktorch.h"
#include "fold.h"

namespace knn {

// --------------------------------------------------------------------------------------
// fold,unfold - set/get size,dilation,padding,stride
// --------------------------------------------------------------------------------------
torch::nn::FoldOptions fold(K x,J i,Cast c) {
 torch::nn::FoldOptions o(0,0);
 bool out=false,sz=false; Pairs p; J n=xargc(x,i,p);
 for(J j=0;j<n;++j)
   switch(j) {
    case 0: o.output_size(exarray<2>(x,i+j,c,Setting::out)); out=true; break;
    case 1: o.kernel_size(exarray<2>(x,i+j,c,Setting::size)); sz=true; break;
    case 2: o.dilation   (exarray<2>(x,i+j,c,Setting::dilate)); break;
    case 3: o.padding    (exarray<2>(x,i+j,c,Setting::pad));    break;
    case 4: o.stride     (exarray<2>(x,i+j,c,Setting::stride)); break;
    default: mpos(x,c,i+j); break;
   }
 while(xpair(p))
  switch(mset(p.k,c)) {
   case Setting::out:       o.output_size(exarray<2>(p,c));out=true; break;
   case Setting::size:      o.kernel_size(exarray<2>(p,c)); sz=true; break;
   case Setting::dilate:    o.dilation   (exarray<2>(p,c)); break;
   case Setting::pad:       o.padding    (exarray<2>(p,c)); break;
   case Setting::stride:    o.stride     (exarray<2>(p,c)); break;
   default: mpair(c,p); break;
  }
 TORCH_CHECK(out, msym(c),": no output size given");
 TORCH_CHECK(sz,  msym(c),": no kernel size given");
 return o;
}

K fold(bool a,const torch::nn::FoldOptions& o) {
 K x=KDICT; torch::nn::FoldOptions d(o.output_size(),o.kernel_size());
 msetting(x, Setting::out,  KEX(o.output_size()));
 msetting(x, Setting::size, KEX(o.kernel_size()));
 if(a || (*o.dilation() != *d.dilation())) msetting(x, Setting::dilate, KEX(o.dilation()));
 if(a || (*o.padding()  != *d.padding()))  msetting(x, Setting::pad,    KEX(o.padding()));
 if(a || (*o.stride()   != *d.stride()))   msetting(x, Setting::stride, KEX(o.stride()));
 return x;
}

torch::nn::UnfoldOptions unfold(K x,J i,Cast c) {
 torch::nn::UnfoldOptions o(0);
 bool sz=false; Pairs p; J n=xargc(x,i,p);
 for(J j=0;j<n;++j)
   switch(j) {
    case 0: o.kernel_size(exarray<2>(x,i+j,c,Setting::size)); sz=true; break;
    case 1: o.dilation   (exarray<2>(x,i+j,c,Setting::dilate)); break;
    case 2: o.padding    (exarray<2>(x,i+j,c,Setting::pad));    break;
    case 3: o.stride     (exarray<2>(x,i+j,c,Setting::stride)); break;
    default: mpos(x,c,i+j); break;
   }
 while(xpair(p))
  switch(mset(p.k,c)) {
   case Setting::size:      o.kernel_size(exarray<2>(p,c)); sz=true; break;
   case Setting::dilate:    o.dilation   (exarray<2>(p,c)); break;
   case Setting::pad:       o.padding    (exarray<2>(p,c)); break;
   case Setting::stride:    o.stride     (exarray<2>(p,c)); break;
   default: mpair(c,p); break;
  }
 TORCH_CHECK(sz, msym(c),": no kernel size given");
 return o;
}

K unfold(bool a,const torch::nn::UnfoldOptions& o) {
 K x=KDICT; torch::nn::UnfoldOptions d(o.kernel_size());
 msetting(x, Setting::size, KEX(o.kernel_size()));
 if(a || (*o.dilation() != *d.dilation())) msetting(x, Setting::dilate, KEX(o.dilation()));
 if(a || (*o.padding()  != *d.padding()))  msetting(x, Setting::pad,    KEX(o.padding()));
 if(a || (*o.stride()   != *d.stride()))   msetting(x, Setting::stride, KEX(o.stride()));
 return x;
}

} // knn namespace
