#pragma once
#include "util.h"
  
namespace knn {

// ------------------------------------------------------------------------------------
// set/get single option of inplace flag for activation fns: relu,relu6,selu
// ------------------------------------------------------------------------------------
bool inplace(K,J,Cast);
K inplace(bool,bool);

// ---------------------------------------------------------------------------------------------
// set/get optional alpha & inplace flag for elu,celu (exponential, continuously differentiable)
// ---------------------------------------------------------------------------------------------
template<typename O> O alpha(K x,J i,Cast c) {
 O o; Pairs p; bool b; J n=xargc(x,i,p);
 if(n==1) {
  if(xbool(x,i,b))
    o.inplace(b);
  else
    o.alpha(mdouble(x,i,c,Setting::alpha));
 } else if(n==2) {
   o.alpha(mdouble(x,i,   c, Setting::alpha));
   o.inplace(mbool(x,i+1, c, Setting::inplace));
 } else if(n) {
  TORCH_ERROR(msym(c), ": unrecognized positional option(s), expecting alpha, inplace flag, or (alpha;inplace flag)");
 }
 while(xpair(p))
  switch(mset(p.k,c)) {
   case Setting::alpha:   o.alpha(mdouble(p,c)); break;
   case Setting::inplace: o.inplace(mbool(p,c)); break;
   default: mpair(c,p); break;
  }
 return o;
}

template<typename O> K alpha(bool a,const O& o) {
 K x=KDICT; O d;
 if(a || o.alpha()   != d.alpha())   msetting(x, Setting::alpha,   kf(o.alpha()));
 if(a || o.inplace() != d.inplace()) msetting(x, Setting::inplace, kb(o.inplace()));
 return resolvedict(x);
}

// --------------------------------------------------------------------------------------
// set/get slope & inplace flag for leakyrelu - a small positive gradient(slope) when x<0
// --------------------------------------------------------------------------------------
torch::nn::LeakyReLUOptions slope(K,J,Cast);
K slope(bool,Cast,const torch::nn::LeakyReLUOptions&);

// ------------------------------------------------------------
// get/set single option: lambda (for hardshrink, softshrink)
// ------------------------------------------------------------
double lambda(Cast);
double lambda(K,J,Cast);
K lambda(bool,Cast,double);

// ------------------------------------------------------------------------------
// set/get single dimension option (cat,glu & softmin,softmax,logsoftmax modules)
// ------------------------------------------------------------------------------
int64_t dim(Cast);
int64_t dim(K,J,Cast);
K dim(bool,Cast,int64_t);

// ----------------------------------------------------------------------------------
// softmax,softmin,logsoftmax: functional form requires dim & optional data type
// softdim: get default dimension from input tensor dimensions (deprecated)
// softargs: set options from k arg(s)
// ----------------------------------------------------------------------------------
J softdim(size_t);
void softargs(K,J,Cast,J&,c10::optional<Dtype>&);

// -----------------------------------------------------------------------------------
// rrelu - randomized leaky relu, functional form has an additional flag for training
// -----------------------------------------------------------------------------------
void rrelu(K x,J i,Cast c,bool fn,bool& tr,bool& in,double& lo,double& up);
torch::nn::RReLUOptions rrelu(K,J,Cast);       // return options for rrelu module
K rrelu(bool,const torch::nn::RReLUOptions&);  // retrieve options from rrelu module

// ----------------------------------------------------------------------------
// hardtanh - computationally cheaper version of tanh, straight line at min,max
//            functions here set/get module options from/to k values
// ----------------------------------------------------------------------------
torch::nn::HardtanhOptions hardtanh(K,J,Cast);
K hardtanh(bool,const torch::nn::HardtanhOptions&);

// ----------------------------------------------------------------------------
// softplus - smooth approximation to relu, can constrain to always be positive
// ----------------------------------------------------------------------------
torch::nn::SoftplusOptions softplus(K,J,Cast);
K softplus(bool,const torch::nn::SoftplusOptions&);

// -------------------------------------------------------------
// threshold - set/get threshold, replacement value,inplace flag
// -------------------------------------------------------------
torch::nn::ThresholdOptions threshold(K,J,Cast);
K threshold(bool,const torch::nn::ThresholdOptions&);

// -----------------------------------------------------------------------------------
// prelu: parameterized relu
//        module accepts 1 or number of input parms & optional initalization value
//        function requires weight directly rather than module's count & initial value
// -----------------------------------------------------------------------------------
torch::nn::PReLUOptions prelu(K,J,Cast);
K prelu(bool,const torch::nn::PReLUOptions&);

} // namespace knn
