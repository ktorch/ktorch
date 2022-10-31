#include "../ktorch.h"
#include "distance.h"

namespace knn {

// ----------------------------------------------------------------------------
// similar - cosine similarity distance, get/set optional dim & epsilon
// pairwise - pairwise distance, get/set optional power,eps,deep dimension flag
// ----------------------------------------------------------------------------
torch::nn::CosineSimilarityOptions similar(K x,J i,Cast c) {
 Pairs p; J n=xargc(x,i,p); torch::nn::CosineSimilarityOptions o;
 for(J j=0;j<n;++j)
  switch(j) {
   case 0: o.dim(int64(x,i+j,c,Setting::dim)); break;
   case 1: o.eps(mdouble(x,i+j,c,Setting::eps)); break;
   default: mpos(x,c,i+j); break;
  }
 while(xpair(p))
  switch(mset(p.k,c)) {
   case Setting::dim: o.dim(int64(p,c)); break;
   case Setting::eps: o.eps(mdouble(p,c)); break;
   default: mpair(c,p); break;
  }
 return o;
}

// also used w'loss functions in kloss.cpp
K similar(bool a,const torch::nn::CosineSimilarityOptions& o) {
 K x=KDICT; torch::nn::CosineSimilarityOptions d; 
 if(a || (o.dim() != o.dim())) msetting(x, Setting::dim, kj(o.dim()));
 if(a || (o.eps() != d.eps())) msetting(x, Setting::eps, kf(o.eps()));
 return resolvedict(x);
}

torch::nn::PairwiseDistanceOptions pairwise(K x,J i,Cast c) {
 Pairs p; J n=xargc(x,i,p); torch::nn::PairwiseDistanceOptions o;
 for(J j=0;j<n;++j)
  switch(j) {
   case 0: o.p(mdouble(x,i+j,c,Setting::p)); break;
   case 1: o.eps(mdouble(x,i+j,c,Setting::eps)); break;
   case 2: o.keepdim(mbool(x,i+j,c,Setting::keepdim)); break;
   default: mpos(x,c,i+j); break;
  }
 while(xpair(p))
  switch(mset(p.k,c)) {
   case Setting::p: o.p(mdouble(p,c)); break;
   case Setting::eps: o.eps(mdouble(p,c)); break;
   case Setting::keepdim: o.keepdim(mbool(p,c)); break;
   default: mpair(c,p); break;
  }
 return o;
}

// also used w'loss functions in kloss.cpp
K pairwise(bool a,const torch::nn::PairwiseDistanceOptions& o) {
 K x=KDICT; torch::nn::PairwiseDistanceOptions d; 
 if(a || (o.p()       != d.p()))       msetting(x, Setting::p,       kf(o.p()));
 if(a || (o.eps()     != d.eps()))     msetting(x, Setting::eps,     kf(o.eps()));
 if(a || (o.keepdim() != d.keepdim())) msetting(x, Setting::keepdim, kb(o.keepdim()));
 return resolvedict(x);
}

} // namespace knn
