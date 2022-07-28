#include "../ktorch.h"
#include "upsample.h"

namespace knn {

// ---------------------------------------------------------------------------
// upsample & interpolate - similar options, interpolate has more modes, scale
// upsample implemented as a module in pytorch, interpolate only as a function
// ---------------------------------------------------------------------------
void upmode(torch::nn::UpsampleOptions& o,S s) {
 switch(emap(s)) {
  case Enum::nearest:   o.mode(torch::kNearest); break;
  case Enum::linear:    o.mode(torch::kLinear); break;
  case Enum::bilinear:  o.mode(torch::kBilinear); break;
  case Enum::bicubic:   o.mode(torch::kBicubic); break;
  case Enum::trilinear: o.mode(torch::kTrilinear); break;
  default: TORCH_ERROR("unrecognized upsample mode: ",s); break;
 }
}

void upmode(torch::nn::functional::InterpolateFuncOptions& o,S s) {
 switch(emap(s)) {
  case Enum::nearest:      o.mode(torch::kNearest); break;
  case Enum::nearestexact: o.mode(torch::kNearestExact); break;
  case Enum::linear:       o.mode(torch::kLinear); break;
  case Enum::bilinear:     o.mode(torch::kBilinear); break;
  case Enum::bicubic:      o.mode(torch::kBicubic); break;
  case Enum::trilinear:    o.mode(torch::kTrilinear); break;
  case Enum::area:         o.mode(torch::kArea); break;
  default: TORCH_ERROR("unrecognized interpolate mode: ",s); break;
 }
}

S upmode(const torch::nn::UpsampleOptions& o)                    {return ESYM(o.mode());}
S upmode(const torch::nn::functional::InterpolateFuncOptions& o) {return ESYM(o.mode());}

// recompute_scale_factor only part of interpolate options, separate fns to handle setting
void rescale(K x,J i,Cast c,Setting s,torch::nn::UpsampleOptions& o) {
 TORCH_ERROR(msym(c),": up to 4 positional arguments expected, 5th argument unrecognized");
}
void rescale(K x,J i,Cast c,Setting s,torch::nn::functional::InterpolateFuncOptions& o) {
 if(xempty(x,i)) o.recompute_scale_factor({}); else o.recompute_scale_factor(mbool(x,i,c,s));
}
void rescale(const Pairs& p,Cast c,torch::nn::UpsampleOptions& o) {
 TORCH_ERROR(msym(c),": rescale is not a recognized option");
}
void rescale(const Pairs& p,Cast c,torch::nn::functional::InterpolateFuncOptions& o) {
 if(pempty(p)) o.recompute_scale_factor({}); else o.recompute_scale_factor(mbool(p,c));
}

K upsample(bool a,const torch::nn::UpsampleOptions& o) {return interp(a,o);}

K interpolate(bool a,const torch::nn::functional::InterpolateFuncOptions& o) {
 K x=interp(a,o);
 msetting(x, Setting::rescale, o.recompute_scale_factor() ? kb(*o.recompute_scale_factor()) : ktn(0,0));
 return x;
}

} // namespace knn
