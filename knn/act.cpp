#include "../ktorch.h"
#include "fns.h"
#include "act.h"

namespace knn {

// ------------------------------------------------------------------------------------
// set/get single option of inplace flag for activation fns: relu,relu6,selu
// ------------------------------------------------------------------------------------
bool inplace(K x,J i,Cast c) {
 bool b=false; Pairs p; J n=xargc(x,i,p);
 if(n)
  TORCH_CHECK(xbool(x,i,b) && n==1, msym(c),": unrecognized option(s), expecting single boolean flag");
 while(xpair(p)) {
  TORCH_CHECK(mset(p.k,c)==Setting::inplace, msym(c),": unrecognized option: ",p.k);
  b=mbool(p,c);
 }
 return b;
}

K inplace(bool a,bool b) {K x=KDICT; if(a || b) msetting(x, Setting::inplace, kb(b)); return x;}

// --------------------------------------------------------------------------------------
// set/get slope & inplace flag for leakyrelu - a small positive gradient(slope) when x<0
// --------------------------------------------------------------------------------------
torch::nn::LeakyReLUOptions slope(K x,J i,Cast c) {
 torch::nn::LeakyReLUOptions o; Pairs p; bool b; J n=xargc(x,i,p);
 if(n==1) {
  if(xbool(x,i,b))
    o.inplace(b);
  else
    o.negative_slope(mdouble(x,i,c,Setting::slope));
 } else if(n==2) {
   o.negative_slope(mdouble(x, i, c, Setting::slope));
   o.inplace(mbool(x,i+1, c, Setting::inplace));
 } else if(n) {
  TORCH_ERROR(msym(c), ": unrecognized positional option(s), expecting slope, inplace flag, or (slope;inplace flag)");
 }
 while(xpair(p))
  switch(mset(p.k,c)) {
   case Setting::slope:   o.negative_slope(mdouble(p,c)); break;
   case Setting::inplace: o.inplace(mbool(p,c)); break;
   default: mpair(c,p); break;
  }
 return o;
}

K slope(bool a,Cast c,const torch::nn::LeakyReLUOptions& o) {
 K x=KDICT; torch::nn::LeakyReLUOptions d;
 if(a || o.negative_slope()   != d.negative_slope()) msetting(x, Setting::slope,   kf(o.negative_slope()));
 if(a || o.inplace()          != d.inplace())        msetting(x, Setting::inplace, kb(o.inplace()));
 return x;
}
// ------------------------------------------------------------
// get/set single option: lambda (for hardshrink, softshrink)
// ------------------------------------------------------------
double lambda(Cast c) {
 return c==Cast::hardshrink ? torch::nn::HardshrinkOptions().lambda()
                            : torch::nn::SoftshrinkOptions().lambda();
}

double lambda(K x,J i,Cast c) {
 double l=lambda(c); Pairs p; J n=xargc(x,i,p);
 if(n==1) l=mdouble(x,i,c,Setting::lambda);
 TORCH_CHECK(n<2,msym(c),": unrecognized positional option(s), expecting lambda, e.g. 0.5");
 while(xpair(p)) {
  TORCH_CHECK(mset(p.k,c)==Setting::lambda,"unrecognized option: ",p.k); l=mdouble(p,c);
 }
 return l;
}

K lambda(bool a,Cast c,double l) {
 K x=KDICT;
 if(a || l != lambda(c)) msetting(x, Setting::lambda, kf(l));
 return x;
}

// ----------------------------------------------------------------------
// set/get single dimension option (cat,glu & softmin,softmax,logsoftmax)
// ----------------------------------------------------------------------
int64_t dim(Cast c) {
 switch(c) {
  case Cast::glu: return torch::nn::GLUOptions().dim();
  case Cast::cat: return knn::CatOptions().dim();
  default:        return nj;
 }
}

int64_t dim(K x,J i,Cast c) {
 int64_t d=dim(c); Pairs p; J n=xargc(x,i,p);
 if(n==1) d=int64(x,i,c,Setting::dim);
 TORCH_CHECK(n<2, msym(c),": unrecognized positional option(s), expecting single dimension");
 while(xpair(p)) {
  TORCH_CHECK(mset(p.k,c)==Setting::dim,"unrecognized option: ",p.k); d=int64(p,c);
 }
 TORCH_CHECK(d!=nj, msym(c),": no dimension given");
 return d;
}

K dim(bool a,Cast c,int64_t d) {
 K x=KDICT;
 if(a || d != dim(c)) msetting(x, Setting::dim, kj(d));
 return x;
}

// ----------------------------------------------------------------------------------
// softmax,softmin,logsoftmax: functional form requires dim & optional data type
// softdim: get default dimension from input tensor dimensions (deprecated)
// ----------------------------------------------------------------------------------
J softdim(size_t d) {return !(d==0 || d==1 || d==3);}

void softargs(K x,J i,Cast c,J &d,c10::optional<Dtype>& s) {
 s=c10::nullopt; Pairs p; J n=xargc(x,i,p);
 if(!((n==0 && p.n) || (xlong(x,i,d) && (n==1 || (n==2 && xtype(x,i+1,s))))))
  TORCH_ERROR(msym(c),": unrecognized arg(s), expecting dim or (dim;data type)");
 while(xpair(p))
  switch(mset(p.k,c)) {
   case Setting::dim:   d=int64(p,c); break;
   case Setting::dtype: s=knn::otype(p,c); break;
   default: mpair(c,p); break;
  }
 if(null(d))
  TORCH_ERROR("specify the dimension along which ",msym(c)," will be computed");
}

// -----------------------------------------------------------------------------------
// rrelu - randomized leaky relu, functional form has an additional flag for training
// -----------------------------------------------------------------------------------
void rrelu(K x,J i,Cast c,bool fn,bool& tr,bool& in,double& lo,double& up) {
 Pairs p; J n=xargc(x,i,p); torch::nn::functional::RReLUFuncOptions o;
 lo=o.lower(); up=o.upper(); in=o.inplace(); tr=o.training();
 if(n) {
  if(fn) {
   TORCH_CHECK((n==1 && (xnum(x,i,lo) || xbool(x,i,tr))) ||
               (n==2 &&  xnum(x,i,lo) && (xnum(x,i+1,up) || xbool(x,i+1,tr))) ||
               (n==3 &&  xnum(x,i,lo) &&  xnum(x,i+1,up) && xbool(x,i+2,tr))  ||
               (n==4 &&  xnum(x,i,lo) &&  xnum(x,i+1,up) && xbool(x,i+2,tr) && xbool(x,i+3,in)),
               "rrelu: unexpected positional arg(s), expects (lower;upper;train flag;inplace flag)");
  } else {
   TORCH_CHECK((n==1 && (xnum(x,i,lo) || xbool(x,i,in))) ||
               (n==2 &&  xnum(x,i,lo) && (xnum(x,i+1,up) || xbool(x,i+1,in))) ||
               (n==3 &&  xnum(x,i,lo) &&  xnum(x,i+1,up) && xbool(x,i+2,in)),
               "rrelu: unexpected positional arg(s), expects (lower;upper;inplace flag)");
  }
 }
 while(xpair(p))
  switch(mset(p.k,c)) {
   case Setting::lower:   lo=mdouble(p,c); break;
   case Setting::upper:   up=mdouble(p,c); break;
   case Setting::train:   TORCH_CHECK(fn,"rrelu: training flag not set for module"); tr=mbool(p,c);   break;
   case Setting::inplace: in=mbool(p,c);   break;
   default: mpair(c,p); break;
  }
}

// return options for rrelu module
torch::nn::RReLUOptions rrelu(K x,J i,Cast c) {
 double lo,up; bool in,tr; rrelu(x,i,c,false,tr,in,lo,up);
 return torch::nn::RReLUOptions().lower(lo).upper(up).inplace(in);
}

// retrieve options from rrelu module
K rrelu(bool a,const torch::nn::RReLUOptions& o) {
 K x=KDICT; torch::nn::RReLUOptions d;
 if(a || d.lower()   != o.lower())   msetting(x, Setting::lower,   kf(o.lower()));
 if(a || d.upper()   != o.upper())   msetting(x, Setting::upper,   kf(o.upper()));
 if(a || d.inplace() != o.inplace()) msetting(x, Setting::inplace, kb(o.inplace()));
 return x;
}

// ----------------------------------------------------------------------------
// hardtanh - computationally cheaper version of tanh, straight line at min,max
//            functions here set/get module options from/to k values
// ----------------------------------------------------------------------------
torch::nn::HardtanhOptions hardtanh(K x,J i,Cast c) {
 Pairs p; J n=xargc(x,i,p); torch::nn::HardtanhOptions o;
 bool b=o.inplace(); double v1=o.min_val(),v2=o.max_val();
 if(n) {
  TORCH_CHECK((n==1 && (xnum(x,i,v1) || xbool(x,i,b))) ||
              (n==2 &&  xnum(x,i,v1) && (xnum(x,i+1,v2) || xbool(x,i+1,b))) ||
              (n==3 &&  xnum(x,i,v1) &&  xnum(x,i+1,v2) && xbool(x,i+2,b)),
              "hardtanh: unexpected positional arg(s), expects (min;max;inplace flag)");
 }
 while(xpair(p))
  switch(mset(p.k,c)) {
   case Setting::min:     v1=mdouble(p,c); break;
   case Setting::max:     v2=mdouble(p,c); break;
   case Setting::inplace: b=mbool(p,c); break;
   default: mpair(c,p); break;
  }
 return o.min_val(v1).max_val(v2).inplace(b);
}

K hardtanh(bool a,const torch::nn::HardtanhOptions& o) {
 K x=KDICT; torch::nn::HardtanhOptions d;
 if(a || d.min_val() != o.min_val()) msetting(x, Setting::min,     kf(o.min_val()));
 if(a || d.max_val() != o.max_val()) msetting(x, Setting::max,     kf(o.max_val()));
 if(a || d.inplace() != o.inplace()) msetting(x, Setting::inplace, kb(o.inplace()));
 return x;
}

// ----------------------------------------------------------------------------
// softplus - smooth approximation to relu, can constrain to always be positive
// ----------------------------------------------------------------------------
torch::nn::SoftplusOptions softplus(K x,J i,Cast c) {
 Pairs p; J n=xargc(x,i,p); torch::nn::SoftplusOptions o; double v1=o.beta(),v2=o.threshold();
 if(n) {
  TORCH_CHECK(xnum(x,i,v1) && (n==1 || (n==2 && xnum(x,i+1,v2))),
              "softplus: unexpected positional arg(s), expects (beta;threshold)");
 }
 while(xpair(p))
  switch(mset(p.k,c)) {
   case Setting::beta:      v1=mdouble(p,c); break;
   case Setting::threshold: v2=mdouble(p,c); break;
   default: mpair(c,p); break;
  }
 return o.beta(v1).threshold(v2);
}

K softplus(bool a,const torch::nn::SoftplusOptions& o) {
 K x=KDICT; torch::nn::SoftplusOptions d;
 if(a || d.beta()      != o.beta())      msetting(x, Setting::beta,      kf(o.beta()));
 if(a || d.threshold() != o.threshold()) msetting(x, Setting::threshold, kf(o.threshold()));
 return x;
}

// -------------------------------------------------------------
// threshold - set/get threshold, replacement value,inplace flag
// -------------------------------------------------------------
torch::nn::ThresholdOptions threshold(K x,J i,Cast c) {
 Pairs p; J n=xargc(x,i,p); bool b=false; double v1=nf,v2=nf;
 if(n) {
  TORCH_CHECK((n==1 && (xnum(x,i,v1) || xbool(x,i,b))) ||
              (n==2 &&  xnum(x,i,v1) && (xnum(x,i+1,v2) || xbool(x,i+1,b))) ||
              (n==3 &&  xnum(x,i,v1) &&  xnum(x,i+1,v2) && xbool(x,i+2,b)),
              "threshold: unexpected positional arg(s), expects (threshold;value;inplace flag)");
 }
 while(xpair(p))
  switch(mset(p.k,c)) {
   case Setting::threshold: v1=mdouble(p,c); break;
   case Setting::value:     v2=mdouble(p,c); break;
   case Setting::inplace:   b=mbool(p,c); break;
   default: mpair(c,p); break;
  }
 TORCH_CHECK(v1 == v1 && v2 == v2, "threshold: both threshold level & replacement value must be given");
 return torch::nn::ThresholdOptions(v1,v2).inplace(b);
}

K threshold(bool a,const torch::nn::ThresholdOptions& o) {
 K x=KDICT;
 msetting(x, Setting::threshold, kf(o.threshold()));
 msetting(x, Setting::value,     kf(o.value()));
 if(a || o.inplace()) msetting(x, Setting::inplace, kb(o.inplace()));
 return x;
}

// -----------------------------------------------------------------------------------
// prelu: parameterized relu
//        module accepts 1 or number of input parms & optional initalization value
//        function requires weight directly rather than module's count & initial value
// -----------------------------------------------------------------------------------
torch::nn::PReLUOptions prelu(K x,J i,Cast c) {
 torch::nn::PReLUOptions o; auto m=o.num_parameters();auto w=o.init(); Pairs p; J n=xargc(x,i,p);
 if(n) TORCH_CHECK((n==1 && (xint64(x,i,m) || xdouble(x,i,w))) ||
                   (n==2 &&  xint64(x,i,m) && xdouble(x,i+1,w)),
                   "prelu: expecting 1-2 positional args in,init or (in;init)");
 while(xpair(p))
  switch(mset(p.k,c)) {
   case Setting::in:    m=int64(p,c); break;
   case Setting::init:  w=mdouble(p,c); break;
   default: mpair(c,p); break;
  }
 return o.num_parameters(m).init(w);
}

K prelu(bool a,const torch::nn::PReLUOptions& o) {
 K x=KDICT; torch::nn::PReLUOptions d;
 if(a || d.num_parameters() != o.num_parameters()) msetting(x, Setting::in,   kj(o.num_parameters()));
 if(a || d.init()           != o.init())           msetting(x, Setting::init, kf(o.init()));
 return x;
}

} // namespace knn
