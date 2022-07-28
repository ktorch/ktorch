#pragma once
#include "util.h"
  
namespace knn {

// ---------------------------------------------------------------------------
// upsample & interpolate - similar options, interpolate has more modes, scale
// upsample implemented as a module in pytorch, interpolate only as a function
// ---------------------------------------------------------------------------
void upmode(torch::nn::UpsampleOptions&,S);
void upmode(torch::nn::functional::InterpolateFuncOptions&,S);
S upmode(const torch::nn::UpsampleOptions&);
S upmode(const torch::nn::functional::InterpolateFuncOptions& m);

// recompute_scale_factor only part of interpolate options, separate fns to handle setting
void rescale(K,J,Cast,Setting,torch::nn::UpsampleOptions&);
void rescale(K,J,Cast,Setting,torch::nn::functional::InterpolateFuncOptions&);
void rescale(const Pairs&,Cast,torch::nn::UpsampleOptions&);
void rescale(const Pairs&,Cast,torch::nn::functional::InterpolateFuncOptions&);

template<typename O>O upsample(K x,J i,Cast c) {
 O o; Pairs p; J n=xargc(x,i,p);
 for(J j=0;j<n;++j) {
  switch(j) {
   case 0: if(xempty(x,i+j)) o.size({}); else o.size(mlongs(x,i+j,c,Setting::size)); break;
   case 1: if(xempty(x,i+j)) o.scale_factor({}); else o.scale_factor(mdoubles(x,i+j,c,Setting::scale)); break;
   case 2: upmode(o,code(x,i+j,c,Setting::mode)); break;
   case 3: if(xempty(x,i+j)) o.align_corners({}); else o.align_corners(mbool(x,i+j,c,Setting::align)); break;
   case 4: rescale(x,i+j,c,Setting::rescale,o); break;
   default: TORCH_ERROR(msym(c),": up to ",(c==Cast::upsample ? 4 : 5)," positional arguments expected, ",n," given");
  }
 }
 while(xpair(p))
  switch(mset(p.k,c)) {
   case Setting::size:    if(pempty(p)) o.size({}); else o.size(mlongs(p,c)); break;
   case Setting::scale:   if(pempty(p)) o.scale_factor({}); else o.scale_factor(mdoubles(p,c)); break;
   case Setting::mode:    upmode(o,code(p,c)); break;
   case Setting::align:   if(pempty(p)) o.align_corners({}); else o.align_corners(mbool(p,c)); break;
   case Setting::rescale: rescale(p,c,o); break;
   default: mpair(c,p); break;
  }
 if(o.size()         && !(*o.size()).size())         o.size({});
 if(o.scale_factor() && !(*o.scale_factor()).size()) o.scale_factor({});
 TORCH_CHECK(o.size() || o.scale_factor(), msym(c),": no output size or scale factor given");
 TORCH_CHECK(!(o.size() && o.scale_factor()), msym(c),": both output size and scale factor given");
 return o;
}

template<typename O> K interp(bool a,const O& o) {
 K x=KDICT; O d;
 if(a || o.size())
  msetting(x, Setting::size, o.size() ? ((*o.size()).size()==1 ? kj((*o.size())[0]) : kget(*o.size())) : ktn(0,0));
 if(a || o.scale_factor())
  msetting(x, Setting::scale, o.scale_factor() ? ((*o.scale_factor()).size()==1 ? kf((*o.scale_factor())[0]) : kget(*o.scale_factor())) : ktn(0,0));
 if(a || o.mode().index() != d.mode().index()) msetting(x, Setting::mode,  ks(upmode(o)));
 if(a || (d.align_corners() != o.align_corners()) ||
         (d.align_corners() == o.align_corners() &&
          o.align_corners() && *o.align_corners() != *d.align_corners()))
  msetting(x, Setting::align, o.align_corners() ? kb(*o.align_corners()) : ktn(0,0));
 return x;
}

K    upsample(bool,const torch::nn::UpsampleOptions&);
K interpolate(bool,const torch::nn::functional::InterpolateFuncOptions&);

} // namespace knn
