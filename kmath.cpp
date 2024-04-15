#include "ktorch.h"

using optint=c10::optional<int64_t>;
using symint=c10::optional<c10::SymInt>;
using optstr=c10::optional<c10::string_view>;
using optdim =torch::OptionalIntArrayRef;
using optsize=torch::OptionalIntArrayRef;

// -----------------------------------------------------------------------------------------
// define function pointers, e.g. Ftt for function(tensor,tensor), G w'output
// -----------------------------------------------------------------------------------------
using Fts     = Tensor  (*)(const Tensor&, const Scalar&);
using Gts     = Tensor& (*)(Tensor&, const Tensor&, const Scalar&);
using Ftt     = Tensor  (*)(const Tensor&, const Tensor&);
using Gtt     = Tensor& (*)(Tensor&, const Tensor&, const Tensor&);

// -----------------------------------------------------------------------------------------
// tensor methods (used for in-place methods, e.g. abs_)
// -----------------------------------------------------------------------------------------
using Ts = Tensor& (Tensor::*)(const Scalar&) const;
using Tt = Tensor& (Tensor::*)(const Tensor&) const;

// -----------------------------------------------------------------------------------------
// point-wise & other math fns, returning new tensor, operating in place or -> output tensor
// -----------------------------------------------------------------------------------------
static K math1(K x,
               Tensor  (*f)(const Tensor&),           // f(x)
               Tensor& (*g)(Tensor&,const Tensor&),   // f_out(r,x)
               Tensor& (Tensor::*m)() const,          // x.inplace_()
               const char* s) {
 KTRY
  Tensor t,r;
  if(xten(x,t)) {                                     // operate on tensor, return new tensor
   return kten(f(t));
  } else if(xten(x,0,t) && xempty(x,1) && x->n==2) {  // operate via in-place method on tensor
   TORCH_CHECK(m, s,": no in-place method");
   return (t.*m)(), (K)0;
  } else if(xten(x,1,r) && x->n==2) {                 // array/tensor -> output tensor
   return g(r,xten(x,0,t) ? t : kput(x,0)), (K)0;
  } else {                                            // k array -> tensor -> fn(tensor) -> back to k array
   return kget(f(kput(x)));
  }
 KCATCH(s);
}

KAPI Abs(K x)        {return math1(x, torch::abs,         torch::abs_out,         &Tensor::abs_,         "absolute value");}
KAPI Acos(K x)       {return math1(x, torch::acos,        torch::acos_out,        &Tensor::acos_,        "arccosine");}
KAPI Angle(K x)      {return math1(x, torch::angle,       torch::angle_out,       nullptr,               "angle");}
KAPI Asin(K x)       {return math1(x, torch::asin,        torch::asin_out,        &Tensor::asin_,        "arcsine");}
KAPI Atan(K x)       {return math1(x, torch::atan,        torch::atan_out,        &Tensor::atan_,        "arctangent");}
KAPI Bitwisenot(K x) {return math1(x, torch::bitwise_not, torch::bitwise_not_out, &Tensor::bitwise_not_, "bitwise not");}
KAPI Ceil(K x)       {return math1(x, torch::ceil,        torch::ceil_out,        &Tensor::ceil_,        "ceiling");}
KAPI Cos(K x)        {return math1(x, torch::cos,         torch::cos_out,         &Tensor::cos_,         "cosine");}
KAPI Cosh(K x)       {return math1(x, torch::cosh,        torch::cosh_out,        &Tensor::cosh_,        "hyperbolic cosine");}
KAPI Digamma(K x)    {return math1(x, torch::digamma,     torch::digamma_out,     &Tensor::digamma_,     "log derivative of gamma");}
KAPI Erf(K x)        {return math1(x, torch::erf,         torch::erf_out,         &Tensor::erf_,         "error function");}
KAPI Erfc(K x)       {return math1(x, torch::erfc,        torch::erfc_out,        &Tensor::erfc_,        "complimentary error function");}
KAPI Erfinv(K x)     {return math1(x, torch::erfinv,      torch::erfinv_out,      &Tensor::erfinv_,      "inverse error function");}
KAPI Exp(K x)        {return math1(x, torch::exp,         torch::exp_out,         &Tensor::exp_,         "exponential");}
KAPI Expm1(K x)      {return math1(x, torch::expm1,       torch::expm1_out,       &Tensor::expm1_,       "exponential minus 1");}
KAPI Floor(K x)      {return math1(x, torch::floor,       torch::floor_out,       &Tensor::floor_,       "floor");}
KAPI Frac(K x)       {return math1(x, torch::frac,        torch::frac_out,        &Tensor::frac_,        "fractional");}
KAPI Inverse(K x)    {return math1(x, torch::inverse,     torch::inverse_out,     nullptr,               "matrix inverse");}
KAPI Lgamma(K x)     {return math1(x, torch::lgamma,      torch::lgamma_out,      &Tensor::lgamma_,      "lgamma");}
KAPI Log(K x)        {return math1(x, torch::log,         torch::log_out,         &Tensor::log_,         "log");}
KAPI Log10(K x)      {return math1(x, torch::log10,       torch::log10_out,       &Tensor::log10_,       "log10");}
KAPI Log1p(K x)      {return math1(x, torch::log1p,       torch::log1p_out,       &Tensor::log1p_,       "log1p");}
KAPI Log2(K x)       {return math1(x, torch::log2,        torch::log2_out,        &Tensor::log2_,        "log2");}
KAPI Msort(K x)      {return math1(x, torch::msort,       torch::msort_out,       nullptr,               "msort");}
KAPI Neg(K x)        {return math1(x, torch::neg,         torch::neg_out,         &Tensor::neg_,         "negative");}
KAPI Not(K x)        {return math1(x, torch::logical_not, torch::logical_not_out, &Tensor::logical_not_, "logical not");}
KAPI Reciprocal(K x) {return math1(x, torch::reciprocal,  torch::reciprocal_out,  &Tensor::reciprocal_,  "reciprocal");}
KAPI Round(K x)      {return math1(x, torch::round,       torch::round_out,       &Tensor::round_,       "round");}
KAPI Rsqrt(K x)      {return math1(x, torch::rsqrt,       torch::rsqrt_out,       &Tensor::rsqrt_,       "reciprocal square root");}
KAPI Ksigmoid(K x)   {return math1(x, torch::sigmoid,     torch::sigmoid_out,     &Tensor::sigmoid_,     "sigmoid");}
KAPI Sgn(K x)        {return math1(x, torch::sgn,         torch::sgn_out,         &Tensor::sgn_,         "sgn");}
KAPI Sign(K x)       {return math1(x, torch::sign,        torch::sign_out,        &Tensor::sign_,        "sign");}
KAPI Sin(K x)        {return math1(x, torch::sin,         torch::sin_out,         &Tensor::sin_,         "sine");}
KAPI Sinh(K x)       {return math1(x, torch::sinh,        torch::sinh_out,        &Tensor::sinh_,        "hyperbolic sine");}
KAPI Sqrt(K x)       {return math1(x, torch::sqrt,        torch::sqrt_out,        &Tensor::sqrt_,        "square root");}
KAPI Tan(K x)        {return math1(x, torch::tan,         torch::tan_out,         &Tensor::tan_,         "tangent");}
KAPI Ktanh(K x)      {return math1(x, torch::tanh,        torch::tanh_out,        &Tensor::tanh_,        "hyperbolic tangent");}
KAPI Trunc(K x)      {return math1(x, torch::trunc,       torch::trunc_out,       &Tensor::trunc_,       "truncate");}

// -------------------------------------------------------------------------
// point-wise functions w'args (x;y;optional output tensor),y may be scalar
// -------------------------------------------------------------------------
static K math2(K x,Ftt f,Fts fs,Gtt g,Tt m,Ts ms,const char* c) {
 KTRY
  bool e=false,p=false; Scalar s; Tensor a,b,r;
  if(x->t) {
   TORCH_CHECK(x->t>0, c,": not implemented for ",kname(x->t));
   a=kput(x);
   TORCH_CHECK(a.size(0)==2, c, ": unable to derive a pair of values from inputs of size ", a.sizes());
   b=a[1]; a=a[0];
  } else {
   TORCH_CHECK(x->n==2 || x->n==3, c,": expecting 2-3 args, (x;y) or (x;y;output), but given ",x->n);
   TORCH_CHECK(x->n==2 || (xten(x,2,r) || (e=xempty(x,2))), c,": need an output tensor for 3rd argument, given ",kname(x,2));
   if(xscalar(x,1,s)) {
    if(!(p=xten(x,0,a)))
     a=kput(x,0);
    if(!fs || !ms || r.defined())   // convert scalar to tensor if no scalar fn or method or if using output tensor
     b=torch::scalar_to_tensor(s,a.device());
   } else {
    p=xtenarg(x,a,b);
   }
  }
  if(e) {
   if(b.defined()) {
    TORCH_CHECK(m, c,": no in-place call implemented");
    (a.*m)(b);
   } else {
    TORCH_CHECK(ms, c,": no in-place call implemented");
    (a.*ms)(s);
   }
   return xptr(x,0) ? (K)0 : kget(a);
  } else if(r.defined()) {
   return g(r,a,b), (K)0;
  } else {
   return kresult(p, b.defined() ? f(a,b) : fs(a,s));
  }
 KCATCH(c);
}

KAPI Atan2(K x)     {return math2(x, torch::atan2,       nullptr,          torch::atan2_out,       &Tensor::atan2_,       nullptr,             "atan2");}
KAPI Div(K x)       {return math2(x, torch::div,         torch::div,       torch::div_out,         &Tensor::div_,         &Tensor::div_,       "divide");}
KAPI Fmax(K x)      {return math2(x, torch::fmax,        nullptr,          torch::fmax_out,        nullptr,               nullptr,             "fmax");}
KAPI Fmin(K x)      {return math2(x, torch::fmin,        nullptr,          torch::fmin_out,        nullptr,               nullptr,             "fmin");}
KAPI Fmod(K x)      {return math2(x, torch::fmod,        torch::fmod,      torch::fmod_out,        &Tensor::fmod_,        &Tensor::fmod_,      "fmod");}
KAPI Maximum(K x)   {return math2(x, torch::maximum,     nullptr,          torch::maximum_out,     nullptr,               nullptr,             "maximum");}
KAPI Minimum(K x)   {return math2(x, torch::minimum,     nullptr,          torch::minimum_out,     nullptr,               nullptr,             "minimum");}
KAPI Mul(K x)       {return math2(x, torch::mul,         torch::mul,       torch::mul_out,         &Tensor::mul_,         &Tensor::mul_,       "multiply");}
KAPI Remainder(K x) {return math2(x, torch::remainder,   torch::remainder, torch::remainder_out,   &Tensor::remainder_,   &Tensor::remainder_, "remainder");}
KAPI Xor(K x)       {return math2(x, torch::logical_xor, nullptr,          torch::logical_xor_out, &Tensor::logical_xor_, nullptr,             "xor");}

// --------------------------------------------------------------------------------------------
// add - x + multiplier * y
// addc - handle args for (x;y;z;multiplier;output)
// addcmul/addcdiv - add tensor to product or quotient of two other tensors: t+v*a*b or t+v*a/b
// --------------------------------------------------------------------------------------------
KAPI Add(K x) {
 KTRY
  TORCH_CHECK(x->t>=0, "add: not implemented for ",kname(x));
  bool p=false; Scalar m=1; Tensor a,b,r;
  if(x->t) {
   a=kput(x); auto n=a.size(0);
   TORCH_CHECK(1<n && n<4, "add: 2-3 inputs expected, (x;y;multiplier), but ",n," given");
   if(n==3) m=a[2].item();
   b=a[1]; a=a[0];
  } else {
   J n=x->n;
   TORCH_CHECK(1<n && n<5, "add: 2-4 args expected, (x;y;multiplier;output), but ",n," given");
   if(n>2 && xten(x,n-1,r)) n--;
   if(n>2 && xnum(x,n-1,m)) n--;
   TORCH_CHECK(n==2, "add: unrecognized arg(s) expecting 2-4 args, (x;y;multiplier;output)");
   p=xtenarg(x,a,b);
  }
  return r.defined() ? (torch::add_out(r,a,b,m), (K)0) : kresult(p, torch::add(a,b,m));
 KCATCH("add");
}

static K addc(K x,
              Tensor  (*f)(const Tensor&,const Tensor&,const Tensor&,const Scalar&),
              Tensor& (*g)(Tensor&,const Tensor&,const Tensor&,const Tensor&,const Scalar&),
              const char* e) {
 KTRY
  TORCH_CHECK(x->t>=0, e,": not implemented for ",kname(x));
  bool p=false; Scalar m=1; Tensor a,b,c,r;
  if(x->t) {
   a=kput(x); auto n=a.size(0);
   TORCH_CHECK(2<n && n<5, e,": 3-4 inputs expected, (x;y;z;multiplier), but ",n," given");
   if(n==4) m=a[3].item();
   b=a[1]; c=a[2]; a=a[0];
  } else {
   J n=x->n;
   TORCH_CHECK(2<n && n<6, e,": 3-5 args expected, (x;y;z;multiplier;output), but ",n," given");
   if(n>3 && xten(x,n-1,r)) n--;
   if(n>3 && xnum(x,n-1,m)) n--;
   TORCH_CHECK(n==3, e,": unrecognized arg(s) expecting 3-5 args, (x;y;z;multiplier;output)");
   p=xtenarg(x,a,b,c);
  }
  return r.defined() ? (g(r,a,b,c,m), (K)0) : kresult(p, f(a,b,c,m));
 KCATCH(e);
}

KAPI Addcmul(K x) {return addc(x, torch::addcmul, torch::addcmul_out, "addcmul");}
KAPI Addcdiv(K x) {return addc(x, torch::addcdiv, torch::addcdiv_out, "addcdiv");}

// ----------------------------------------------------------------------------
// kcum - handle k args for cumulative prod/sum: (x;dim;type;output)
// cumprod/cumsum - cumulative prod/sum with optional dim,dtye & output
// ----------------------------------------------------------------------------
static K kcum(K x,
              Tensor  (*f)(const Tensor&,int64_t,c10::optional<Dtype>),
              Tensor& (*g)(Tensor&,const Tensor&,int64_t,c10::optional<Dtype>),
              const char *c) {
 KTRY
  bool p=false; int64_t d=-1; c10::optional<Dtype> t=c10::nullopt; Tensor a,r;
  if(xten(x,a)) {
   p=true;
  } else if(xarray(x,5)) {
   a=kput(x);
  } else {
   J n=x->n;
   TORCH_CHECK(n>0 && n<5, c,": expecting up to 4 args, (input;dim;dtype;output), given ",x->n);
   if(n>1 && (xten(x,n-1,r))) n--;
   if(n>1 && xtype(x,n-1,t)) n--;
   if(n>1 && xint64(x,n-1,d)) n--;
   TORCH_CHECK(n==1, c,": unrecognized args, expecting up to 4, (input;dim;dtype;output)");
   if(!(p=xten(x,0,a))) a=kput(x,0);
  }
  if(r.defined())
   return g(r,a,d,t), (K)0;
  else
   return kresult(p, f(a,d,t));
 KCATCH(c);
}

KAPI Cumprod(K x) {return kcum(x, torch::cumprod, torch::cumprod_out, "cumprod");}
KAPI  Cumsum(K x) {return kcum(x, torch::cumsum,  torch::cumsum_out,  "cumsum");}

// ----------------------------------------------------------------------
// clamp - clamp with min & max, null for min or max does one-sided clamp
// lerp - linear interpolation
// mvlgamma - multivariate log gamma
// pow - power function with scalar or tensor exponent
// ----------------------------------------------------------------------
KAPI Clamp(K x) {
 KTRY
  bool p; c10::optional<Scalar> lo,hi; Tensor t,r;
  TORCH_CHECK(!x->t, "clamp: not implemented for ",kname(x));
  TORCH_CHECK(xnumn(x,1,lo) && xnumn(x,2,hi), "clamp: expecting 2nd & 3rd arg to be numeric low & high limits");
  TORCH_CHECK(x->n==3 || (x->n==4 && xten(x,3,r)), "clamp: unexpected arg(s), expecting (input;lo;hi) or (input;lo;hi;output)");
  if(!(p=xten(x,0,t))) t=kput(x,0);
  if(r.defined())
   return torch::clamp_out(r,t,lo,hi), (K)0;
  else
   return kresult(p,torch::clamp(t,lo,hi));
 KCATCH("clamp");
}

KAPI Lerp(K x) {
 KTRY
  TORCH_CHECK(x->t>=0, "lerp: not implemented for ",kname(x));
  bool p=false; J n=xlen(x); Scalar w=0; Tensor a,b,wt,r;
  TORCH_CHECK(2<n && n<5, "lerp: 3-4 arguments expected, (x;y;wt;output), but ",n," supplied");
  TORCH_CHECK(n==3 || (n==4 && xten(x,n-1,r)), "lerp: optional 4th arg is an output tensor, but given ",kname(x,n-1));
  if(x->t) {
   a=kput(x); b=a[1], w=a[2].item(); a=a[0];
  } else {
   p=xnum(x,2,w) ? xtenarg(x,a,b) : xtenarg(x,a,b,wt);
  }
  if(r.defined())
   return (wt.defined() ? torch::lerp_out(r,a,b,wt) : torch::lerp_out(r,a,b,w)), (K)0;
  else
   return kresult(p, wt.defined() ? torch::lerp(a,b,wt) : torch::lerp(a,b,w));
 KCATCH("linear interpolation");
}

KAPI Mvlgamma(K x) {
 KTRY
  TORCH_CHECK(x->t>=0, "mvlgamma: not implemented for ",kname(x));
  bool p=false; J n=xlen(x); int64_t d; Tensor a,r;
  TORCH_CHECK(xint64(x,1,d), "mvlgamma: 2nd argument of number of dimensions expected, given ",kname(x,1));
  TORCH_CHECK(1<n && n<4, "mvlgammma: 2-3 arguments expected, (x;p;output), but ",n," supplied");
  TORCH_CHECK(n==2 || (n==3 && xten(x,n-1,r)), "mvlgamma: optional 3rd arg is an output tensor, but given ",kname(x,n-1));
  if(!(p=xten(x,0,a))) a=x->t ? kput(x)[0] : kput(x,0);
  return r.defined() ? (torch::mvlgamma_out(r,a,d), (K)0) : kresult(p,torch::mvlgamma(a,d));
 KCATCH("mvlgamma");
}

// ----------------------------------------------------
// kpow - handle args and various forms of power calls
// pow - power function with scalar or tensor exponent
// fpow - power function using double precision
// ----------------------------------------------------
static K kpow(K x,
              Tensor& (Tensor::*m0)(const Scalar&) const,
              Tensor& (Tensor::*m1)(const Tensor&) const,
              Tensor  (*f0)(const Scalar&,const Tensor&),
              Tensor  (*f1)(const Tensor&,const Scalar&),
              Tensor  (*f2)(const Tensor&,const Tensor&),
              Tensor& (*g0)(Tensor&,const Scalar&,const Tensor&),
              Tensor& (*g1)(Tensor&,const Tensor&,const Scalar&),
              Tensor& (*g2)(Tensor&,const Tensor&,const Tensor&),
              const char *c) {
 KTRY
  Scalar s; Tensor a,b,r;
  bool p=false,e=false; J m,n=((e=xempty(x,2)) || xten(x,2,r)) ? x->n-1 : xlen(x);
  if(n != 2) {             m=-1;
  } else if(xnum(x,0,s)) { m=0; if(!(p=xten(x,1,b))) b=kput(x,1);
  } else if(xnum(x,1,s)) { m=1; if(!(p=xten(x,0,a))) a=kput(x,0);
  } else if(x->t) {        m=2; p=false; a=kput(x);  b=a[1]; a=a[0];
  } else {                 m=2; p=xtenarg(x,a,b);
  }
  TORCH_CHECK(m>-1 && m<3, c,": expects inputs x,y with optional output tensor or null as 3rd arg");
  if(e) {
   TORCH_CHECK(a.defined(), c,": cannot be called in-place on a scalar");
   if     (m==1) (a.*m0)(s);
   else if(m==2) (a.*m1)(b);
   return p ? (K)0 : kget(a);
  } else if(r.defined()) {
   if     (m==0) g0(r,s,b);
   else if(m==1) g1(r,a,s);
   else if(m==2) g2(r,a,b);
   return (K)0;
  } else {
   if     (m==0) r=f0(s,b);
   else if(m==1) r=f1(a,s);
   else if(m==2) r=f2(a,b);
   return kresult(p,r);
  }
 KCATCH(c);
}

KAPI  Pow(K x) {return kpow(x, &Tensor::pow_,  &Tensor::pow_,
                               torch::pow,     torch::pow,     torch::pow,
                               torch::pow_out, torch::pow_out, torch::pow_out, "pow");}

KAPI Fpow(K x) {return kpow(x, &Tensor::float_power_,  &Tensor::float_power_,
                               torch::float_power,     torch::float_power,     torch::float_power,
                               torch::float_power_out, torch::float_power_out, torch::float_power_out, "fpow");}

// ---------------------------------------------------------------------
// dist - p-norm of  a - b, with optional exponent p (default p=2)
// ---------------------------------------------------------------------
KAPI Dist(K x) {
 KTRY
  bool p; Scalar n=2; Tensor a,b;
  TORCH_CHECK(x->t>=0, "dist: not implemented for ",kname(x));
  if(x->t) {
   return a=kput(x), kget(torch::dist(a[0],a[1],(x->n==2) ? n : a[2].item()));
  } else {
   TORCH_CHECK(x->n==2 || (x->n==3 && xnum(x,2,n)), "dist: expecting 2-3 args, (x;y;p), ",x->n," arg(s) given");
   return p=xtenarg(x,a,b), kresult(p, torch::dist(a,b,n));
  }
 KCATCH("dist");
}

// ---------------------------------------------------------------------------
// https://github.com/pytorch/pytorch/pull/76547
// newer matrix norm routines, torch::frobenius_norm(a,d,k) deprecated
// matrixnorm: get 1-5 args, input,dim,keepdim,datatype,output
// fnorm,nnorm: frobenius & nuclear norms, call with ord set to 'fro' & 'nuc'
// ---------------------------------------------------------------------------
static K matrixnorm(K x,const char *s,c10::string_view v,IntArrayRef d={-2,-1}) {
 KTRY
  bool k=false,p=false; Tensor a,r; c10::optional<Dtype> t=c10::nullopt;
  if(xten(x,a)) {
   p=true;
  } else if(x->t<0 || xarray(x,5)) {
   a=kput(x);
  } else {
   J n=xlen(x);
   TORCH_CHECK(n>0 && n<6, s,": expecting 1-5 args, (input;dim;keepdim;dtype;output), given ",x->n);
   if(n>1 &&  xten(x,n-1,r)) n--;
   if(n>1 && xtype(x,n-1,t)) n--;
   if(n>1 && xbool(x,n-1,k)) n--;
   if(n>1 && xsize(x,n-1,d)) n--;
   TORCH_CHECK(n==1, s,": unrecognized args, expecting up to 5, (input;dim;keepdim;dtype;output)");
   if(!(p=xten(x,0,a))) a=kput(x,0);
  }
  if(r.defined())
   return torch::linalg_matrix_norm_out(r,a,v,d,k,t), (K)0;
  else
   return kresult(p, torch::linalg_matrix_norm(a,v,d,k,t));
 KCATCH(s);
}

KAPI Fnorm(K x) {return matrixnorm(x, "fnorm", "fro");}
KAPI Nnorm(K x) {return matrixnorm(x, "nnorm", "nuc");}

// ---------------------------------------------------------------------------
// normargs: get 1-6 args, input,ord (norm type),dim,keepdim,datatype & output
// mnorm,vnorm: call matrix & vector norm routines with numeric 'ord' argument
// https://github.com/pytorch/pytorch/pull/76547
// ---------------------------------------------------------------------------
static K normargs(K x,bool v,const char *s,IntArrayRef d={-2,-1}) {
 KTRY
  bool k=false,o=false,p=false,z=false; double f=2; Tensor a,r; c10::optional<Dtype> t=c10::nullopt;
  if(xten(x,a)) {
   p=true;
  } else if(x->t<0 || xarray(x,6)) {
   a=kput(x);
  } else {
   J n=xlen(x);
   TORCH_CHECK(n>0 && n<7, s,": expecting 1-6 args, (input;ord;dim;keepdim;dtype;output), given ",x->n);
   if(n>1 &&  xten(x,n-1,r)) n--;
   if(n>1 && xtype(x,n-1,t)) n--;
   if(n>1 && xbool(x,n-1,k)) n--;
   if(n>1 && xsize(x,n-1,d)) n--,z=true;
   if(n>1 && xdouble(x,n-1,f)) o=true,n--;  // ord arg (2nd positional arg) defined
   TORCH_CHECK(n==1, s,": unrecognized args, expecting up to 6, (input;ord;dim;keepdim;dtype;output)");
   if(!(p=xten(x,0,a))) a=kput(x,0);
  }
  if(v) {
   c10::OptionalIntArrayRef dm=c10::nullopt; if(z && d.size()) dm=d;
   if(r.defined())
    return torch::linalg_vector_norm_out(r,a,f,dm,k,t), (K)0;
   else
    return kresult(p, torch::linalg_vector_norm(a,f,dm,k,t));
  } else {  // if no ord defined, use frobenius norm
   if(r.defined())
    return (o ? torch::linalg_matrix_norm_out(r,a,f,d,k,t) : torch::linalg_matrix_norm_out(r,a,"fro",d,k,t)), (K)0;
   else
    return kresult(p, o ? torch::linalg_matrix_norm(a,f,d,k,t) : torch::linalg_matrix_norm(a,"fro",d,k,t));
  }
 KCATCH(s);
}

KAPI Mnorm(K x) {return normargs(x, false, "mnorm");}
KAPI Vnorm(K x) {return normargs(x, true,  "vnorm");}

// --------------------------------------------------------------------------
// kvar - process args for std dev/variance: (x;dims;unbiased;keepdim;output)
// std - return std deviation given k array/tensor and optional dims,flags
// var - return variance given k array/tensor and optional dims,flags
// --------------------------------------------------------------------------
static K kvar(K x,
              Tensor  (*f1)(const Tensor&,bool),
              Tensor  (*f2)(const Tensor&,optdim,bool,bool),
              Tensor& (*g)(Tensor&,const Tensor&,optdim,bool,bool),
              const char *c) {
 KTRY
  IntArrayRef d; bool p=false,m=false,u=true,k=false; Tensor a,r;
  if(xten(x,a)) {
   p=true;
  } else if(xarray(x,5)) {
   a=kput(x);
  } else {
   J n=x->n;
   TORCH_CHECK(n>0 && n<6, c,": expecting 1-5 args, (input;dim;unbiased;keepdim;output), given ",x->n);
   if(n>1 && (xten(x,n-1,r))) n--;
   if(n>2 && xbool(x,n-2,u) && xbool(x,n-1,k)) n-=2;
   if(n>1 && xbool(x,n-1,u)) n--;
   if(n>1 && xsize(x,n-1,d)) m=true,n--;
   TORCH_CHECK(n==1, c,": unrecognized args, expecting up to 5, (input;dim;unbiased;keepdim;output)");
   if(!(p=xten(x,0,a))) a=kput(x,0);
  }
  if(!(a.is_floating_point() || a.is_complex())) a=a.to(torch::kFloat);
  if(r.defined())
   return g(r,a,d,u,k), (K)0;
  else
   return kresult(p, m ? f2(a,d,u,k) : f1(a,u));
 KCATCH(c);
}

KAPI Std(K x) {return kvar(x, torch::std, torch::std, torch::std_out, "std");}
KAPI Var(K x) {return kvar(x, torch::var, torch::var, torch::var_out, "variance");}

// --------------------------------------------------------------------------
// kvar - args for mean & stddev/variance: (x;dim;unbiased;keepdim)
// meanstd - return mean & stddev given input and optional dims,flags
// meanvar - return mean & var given input and optional dims,flags
// --------------------------------------------------------------------------
static K kvar(K x,
              Tuple (*f1)(const Tensor&,bool),
              Tuple (*f2)(const Tensor&,optdim,bool,bool),
              const char *c) {
 KTRY
  IntArrayRef d; bool p=false,m=false,u=true,k=false; Tensor a;
  if(xten(x,a)) {
   p=true;
  } else if(xarray(x,4)) {
   a=kput(x);
  } else {
   J n=x->n;
   TORCH_CHECK(n>0 && n<5, c,": expecting 1-4 args, (input;dim;unbiased;keepdim), given ",x->n);
   if(n>2 && xbool(x,n-2,u) && xbool(x,n-1,k)) n-=2;
   if(n>1 && xbool(x,n-1,u)) n--;
   if(n>1 && xsize(x,n-1,d)) m=true,n--;
   TORCH_CHECK(n==1, c,": unrecognized args, expecting up to 4, (input;dim;unbiased;keepdim)");
   if(!(p=xten(x,0,a))) a=kput(x,0);
  }
  if(!(a.is_floating_point() || a.is_complex())) a=a.to(torch::kFloat);
  auto t=m ? f2(a,d,u,k) : f1(a,u);
  return kresult(p, torch::stack({std::get<1>(t),std::get<0>(t)}, 0));
 KCATCH(c);
}

KAPI Meanstd(K x) {return kvar(x, torch::std_mean, torch::std_mean, "meanstd");}
KAPI Meanvar(K x) {return kvar(x, torch::var_mean, torch::var_mean, "meanvar");}

// ---------------------------------------------------------------------------
// kmean - handle args for mean, prod & sum calls
// kprod - cover functions to call for single dimension from dimension array
// mean - return mean, optional dimension(s), w'convert to optional data type
// prod - return product, optional single dim, optional data type
// sum - return sum, optional dim(s), optional data type
// ---------------------------------------------------------------------------
static K kmean(K x,
               Tensor  (*f)(        const Tensor&,optdim,bool,c10::optional<Dtype>),
               Tensor& (*g)(Tensor&,const Tensor&,optdim,bool,c10::optional<Dtype>),
               const char *c) {
 KTRY
  bool p=false,k=false; IntArrayRef i; c10::optional<Dtype> t=c10::nullopt; Tensor a,r;
  if(xten(x,a)) {
   p=true;
  } else if(xarray(x,5)) {
   a=kput(x);
  } else {
   J n=x->n;
   TORCH_CHECK(n>0 && n<6, c,": expecting up to 5 args, (input;dim;keepdim;dtype;output), given ",x->n);
   if(n>1 && (xten(x,n-1,r))) n--;
   if(n>1 && xtype(x,n-1,t)) n--;
   if(n>1 && xbool(x,n-1,k)) n--;
   if(n>1 && xsize(x,n-1,i)) n--;
   TORCH_CHECK(n==1, c,": unrecognized args, expecting up to 4, (input;dim;keepdim;output)");
   if(!(p=xten(x,0,a))) a=kput(x,0);
  }
  optdim d=c10::nullopt; if(i.size()) d=i;
  if(r.defined())
   return g(r,a,d,k,t), (K)0;
  else
   return kresult(p,f(a,d,k,t));
 KCATCH(c);
}

static Tensor kprod(const Tensor& x,optdim d,bool k,c10::optional<Dtype> t) {
 auto n=d ? d.value().size() : 0;
 TORCH_CHECK(n==0 || n==1, "prod: product can only be calculated along a single dimension, given ",n);
 return d ? torch::prod(x,d.value()[0],k,t) : torch::prod(x,t);
}

static Tensor& kprod_out(Tensor& r,const Tensor& x,optdim d,bool k,c10::optional<Dtype> t) {
 auto n=d ? d.value().size() : 0;
 TORCH_CHECK(n==0 || n==1, "prod: product can only be calculated along a single dimension, given ",n);
 return torch::prod_out(r,x,n ? d.value()[0] : -1,k,t);
}

KAPI    Mean(K x) {return kmean(x, torch::mean,    torch::mean_out,    "mean");}
KAPI Nanmean(K x) {return kmean(x, torch::nanmean, torch::nanmean_out, "nanmean");}
KAPI    Prod(K x) {return kmean(x, kprod,          kprod_out,          "prod");}
KAPI     Sum(K x) {return kmean(x, torch::sum,     torch::sum_out,     "sum");}
KAPI  Nansum(K x) {return kmean(x, torch::nansum,  torch::nansum_out,  "nansum");}

// --------------------------------------------------------------------------
// kmed - handle args for median/nanmedian/mode
// median - return median if no dimension/output args, else values & indices
// mode - mode uses same args as median, but always returns values & indices
// --------------------------------------------------------------------------
static K kmed(K x,
             Tensor (*f1)(const Tensor&),
             Tuple  (*f2)(const Tensor&,int64_t,bool),
             const char *c) {
 KTRY
  int64_t d=-1; bool b=false,p=false,k=false; Tensor a; TensorVector *v=nullptr;
  if(xten(x,a)) {
   b=true, p=true;
  } else if(xarray(x,4)) {
   b=true, a=kput(x);
  } else {
   J n=x->n;
   TORCH_CHECK(n>0 && n<5, c,": expecting up to 4 args, (input;dim;keepdim;output), given ",x->n);
   if(n>1 && (v=xvec(x,n-1))) n--;
   if(n>1 && xbool(x,n-1,k)) n--;
   if(n>1 && xint64(x,n-1,d)) n--;
   TORCH_CHECK(n==1, c,": unrecognized args, expecting up to 4, (input;dim;keepdim;output)");
   if(!(p=xten(x,0,a))) a=kput(x,0);
  }
  return v ? koutput(*v,f2(a,d,k)) : ((b && f1) ? kresult(p,f1(a)) : kresult(p,f2(a,d,k)));
 KCATCH(c);
}

KAPI    Median(K x) {return kmed(x, torch::median,    torch::median,    "median");}
KAPI Nanmedian(K x) {return kmed(x, torch::nanmedian, torch::nanmedian, "nanmedian");}
KAPI      Mode(K x) {return kmed(x, nullptr,          torch::mode,      "mode");}

// ----------------------------------------------------------------------------------------
// comparison fns with arg of (input1;input2;optional output tensor), input2 may be scalar
// if output tensor is empty, e.g. ge(t;0;[]), operates in-place, like t.ge_(0)
// ----------------------------------------------------------------------------------------
static K compare2(K x,Ftt f,Fts fn,Gtt g,Gts gn,Tt m,Ts mn,const char* s) {
 KTRY
  bool p,e=xempty(x,2); Scalar n; Tensor a,b,r;
  if(2 == ((e || xten(x,2,r)) ? x->n-1 : xlen(x))) {
   if(xnum(x,1,n)) {
    if(!(p=xten(x,0,a))) a=kput(x,0);
   } else {
    if(x->t)
     a=kput(x), b=a[1],a=a[0],p=false;
    else
     p=xtenarg(x,a,b);
   }
   if(e)
    return b.defined() ? (&a->*m)(b) : (&a->*mn)(n), (K)0;
   else if(r.defined())
    return (b.defined() ? g(r,a,b) : gn(r,a,n)), (K)0;
   else
    return kresult(p, b.defined() ? f(a,b) : fn(a,n));
  } else {
   TORCH_ERROR(s,": expects args of(x;y;optional output tensor), x is array or tensor, y may also be a scalar");
  }
 KCATCH(s);
}

KAPI Eq(K x)  {return compare2(x, torch::eq, torch::eq, torch::eq_out, torch::eq_out, &Tensor::eq_, &Tensor::eq_, "eq");}
KAPI Ge(K x)  {return compare2(x, torch::ge, torch::ge, torch::ge_out, torch::ge_out, &Tensor::ge_, &Tensor::ge_, "ge");}
KAPI GT(K x)  {return compare2(x, torch::gt, torch::gt, torch::gt_out, torch::gt_out, &Tensor::gt_, &Tensor::gt_, "gt");}
KAPI Le(K x)  {return compare2(x, torch::le, torch::le, torch::le_out, torch::le_out, &Tensor::le_, &Tensor::le_, "le");}
KAPI Lt(K x)  {return compare2(x, torch::lt, torch::lt, torch::lt_out, torch::lt_out, &Tensor::lt_, &Tensor::lt_, "lt");}
KAPI Ne(K x)  {return compare2(x, torch::ne, torch::ne, torch::ne_out, torch::ne_out, &Tensor::ne_, &Tensor::ne_, "ne");}

// -----------------------------------------------------------------------------------------
// kclose - handle args for close & allclose functions
// close - boolean true for each element where 2 input elements within tolerance, false else
// allclose - true if 1st input values are all within tolerance of 2nd input
// equal - true if 1st input has same size and elements as 2nd input
// -----------------------------------------------------------------------------------------
static K kclose(K x,bool s,const char *c) {
 KTRY
  bool na=false, p=false; double rt=1e-05,at=1e-08; Tensor a,b;
  if(x->t<0) {
   TORCH_ERROR(c,": not implemented for ",kname(x->t));
  } else if(x->t) {
   a=kput(x);
   TORCH_CHECK(a.dim() && (a.size(0)==2 || (a.size(0)<5 && a.dtype()==torch::kDouble)),
               c, ": unable to derive a pair of values to compare from inputs of size ", a.sizes());
   if(a.size(0)>2) rt=a[2].item<double>();
   if(a.size(0)>3) at=a[3].item<double>();
   b=a[1]; a=a[0];
  } else {
   J n=xbool(x,x->n-1,na) ? x->n-1 : x->n;
   TORCH_CHECK(n==2 || (xdouble(x,2,rt) && (n==3 || (n==4 && xdouble(x,3,at)))),
               c, ": expects (x;y), (x;y;nan equal), (x;y;rel tol), (x;y;rel tol;abs tol), or (x;y;rel;abs;nan equal)");
   p=xtenarg(x,a,b);
  }
  if(s)
   return kb(torch::allclose(a,b,rt,at,na));
  else
   return kresult(p, torch::isclose(a,b,rt,at,na));
 KCATCH(c);
}

KAPI Allclose(K x) {return kclose(x, true,  "allclose");}
KAPI    Close(K x) {return kclose(x, false, "close");}

KAPI Equal(K x) {
 KTRY
  Tensor a,b;
  if(xlen(x)==2) {
   if(x->t) a=kput(x), b=a[1], a=a[0];
   else     xtenarg(x,a,b);
   return kb(torch::equal(a,b));
  } else {
   return KERR("equal expects two input arrays/tensors to compare");
  }
 KCATCH("equal");
}

// ----------------------------------------------------------------------------------------------
// comparison functions that check for special values (nan, +/- inf) on floating point tensors
// ----------------------------------------------------------------------------------------------
static K special(K x,Tensor (*f)(const Tensor&)) {
 KTRY
  Tensor *t=xten(x);
  return kresult(t, f(t ? *t : kput(x)));
 KCATCH("special");
}

KAPI Finite(K x) {return special(x, torch::isfinite);}
KAPI Inf(K x)    {return special(x, torch::isinf);}
KAPI Nan(K x)    {return special(x, torch::isnan);}
KAPI Neginf(K x) {return special(x, torch::isneginf);}
KAPI Posinf(K x) {return special(x, torch::isposinf);}

// -------------------------------------------------------------------------
// karg - handle args for any/all
// any/all - true/false if any/all true w'optional dimension & output tensor
// -------------------------------------------------------------------------
static K karg(K x, 
              Tensor   (*f1)(const Tensor&),
              Tensor   (*f2)(const Tensor&, int64_t, bool),
              Tensor&  (*g1)(Tensor&, const Tensor&),
              Tensor&  (*g2)(Tensor&, const Tensor&, int64_t, bool),
              const char *c) {
 KTRY
  int64_t d=-1; bool b=false,p=false,k=false; Tensor a,r;
  if(xten(x,a)) {
   b=true, p=true;
  } else if(xarray(x,4)) {
   b=true, a=kput(x);
  } else {
   J n=x->n;
   TORCH_CHECK(n>0 && n<5, c,": expecting up to 4 args, (input;dim;keepdim;output), given ",x->n);
   if(n>1 && (xten(x,n-1,r))) n--;
   if(n>1 && xbool(x,n-1,k)) n--;
   if(n>1 && xint64(x,n-1,d)) n--;
   TORCH_CHECK(n==1, c,": unrecognized args, expecting up to 4, (input;dim;keepdim;output)");
   if(!(p=xten(x,0,a))) a=kput(x,0);
  }
  if(b) 
   return kresult(p,f1(a));
  else if(r.defined())
   return x->n==2 ? g1(r,a) : g2(r,a,d,k), (K)0;
  else
   return kresult(p,f2(a,d,k));
 KCATCH(c);
}

KAPI All(K x) {return karg(x, torch::all, torch::all, torch::all_out, torch::all_out, "all");}
KAPI Any(K x) {return karg(x, torch::any, torch::any, torch::any_out, torch::any_out, "any");}

// ----------------------------------------------------------------
// kmax - handle args for (input; dim; keepdim; output vector)
// Min/Max - api functions for min/max returning values and indices
// aminmax - returns minimum & maximum values
// ----------------------------------------------------------------
static K kmax(K x,int m,const char *c) {
 // m:mode, 0-max, 1-min, 2-aminmax
 KTRY
  int64_t d=-1; bool b=false,p=false,k=false; Tensor a; TensorVector *v=nullptr;
  if(xten(x,a)) {
   b=true, p=true;
  } else if(xarray(x,4)) {
   b=true, a=kput(x);
  } else {
   J n=x->n;
   TORCH_CHECK(n>0 && n<5, c,": expecting up to 4 args, (input;dim;keepdim;output), given ",x->n);
   if(n>1 && (v=xvec(x,n-1))) n--;
   if(n>1 && xbool(x,n-1,k)) n--;
   if(n>1 && xint64(x,n-1,d)) n--;
   TORCH_CHECK(n==1, c,": unrecognized args, expecting up to 4, (input;dim;keepdim;output)");
   if(!(p=xten(x,0,a))) a=kput(x,0);
  }
  if(b) {  // single tensor or input
   if(m<2) {
    return kresult(p, m ? torch::min(a) : torch::max(a));
   } else {
    const auto& t=torch::aminmax(a);
    return kresult(p, torch::stack({std::get<0>(t), std::get<1>(t)}).reshape(2));
   }
  } else {
   const auto& t=m<2 ? (m ? torch::min(a,d,k) : torch::max(a,d,k)) : torch::aminmax(a,d,k);
   return v ? koutput(*v,t) : kresult(p,t);
  }
 KCATCH(c);
}

KAPI     Max(K x) {return kmax(x, 0, "Max");}
KAPI     Min(K x) {return kmax(x, 1, "Min");}
KAPI Aminmax(K x) {return kmax(x, 2, "aminmax");}

// --------------------------------------------------------------------
// karg - handle args for amax/amin/logsumexp: (x;dim;keepdim;output)
// amax,amin - max,min, w'optional dim(s), keepdim flag & output tensor
// logsumexp - log of summed exponentials, w'dim(s),keepdim & output
// --------------------------------------------------------------------
static K karg(K x,
              bool b,
              Tensor  (*f)(const Tensor&,IntArrayRef,bool),
              Tensor& (*g)(Tensor&,const Tensor&,IntArrayRef,bool),
              const char *c) {
 KTRY
  IntArrayRef d; bool p=false,k=false; Tensor a,r;
  if(xten(x,a)) {
   p=true;
  } else if(xarray(x,4)) {
   a=kput(x);
  } else {
   J n=x->n;
   TORCH_CHECK(n>0 && n<5, c,": expecting up to 4 args, (input;dim;keepdim;output), given ",x->n);
   if(n>1 &&  xten(x,n-1,r)) n--;
   if(n>1 && xbool(x,n-1,k)) n--;
   if(n>1 && xsize(x,n-1,d)) n--;
   TORCH_CHECK(n==1, c,": unrecognized args, expecting up to 4, (input;dim;keepdim;output)");
   if(!(p=xten(x,0,a))) a=kput(x,0);
  }
  TORCH_CHECK(!b || d.size(), c,": needs explicit dimension(s) for log of summed exponentials");
  if(r.defined())
   return g(r,a,d,k), (K)0;
  else
   return kresult(p, f(a,d,k));
 KCATCH(c);
}

KAPI      Amax(K x) {return karg(x, false, torch::amax,      torch::amax_out,      "amax");}
KAPI      Amin(K x) {return karg(x, false, torch::amin,      torch::amin_out,      "amin");}
KAPI Logsumexp(K x) {return karg(x, true,  torch::logsumexp, torch::logsumexp_out, "logsumexp");}

// -----------------------------------------------------------------------
// karg - args for argmax,argmin (x;dim;keepdim;output)
// argmax,argmin - index of max/min, w'optional dim, keepdim flag & output
// -----------------------------------------------------------------------
static K karg(K x,
              Tensor  (*f)(const Tensor&,c10::optional<int64_t>,bool),
              Tensor& (*g)(Tensor&,const Tensor&,c10::optional<int64_t>,bool),
              const char *c) {
 KTRY
  c10::optional<int64_t> d; bool p=false,k=false; Tensor a,r;
  if(xten(x,a)) {
   p=true;
  } else if(xarray(x,4)) {
   a=kput(x);
  } else {
   J n=x->n;
   TORCH_CHECK(n>0 && n<5, c,": expecting up to 4 args, (input;dim;keepdim;output), given ",x->n);
   if(n>1 &&  xten(x,n-1,r)) n--;
   if(n>1 && xbool(x,n-1,k)) n--;
   if(n>1 && xint64(x,n-1,d)) n--;
   TORCH_CHECK(n==1, c,": unrecognized args, expecting up to 4, (input;dim;keepdim;output)");
   if(!(p=xten(x,0,a))) a=kput(x,0);
  }
  if(r.defined())
   return g(r,a,d,k), (K)0;
  else
   return kresult(p, f(a,d,k));
 KCATCH(c);
}

KAPI Argmax(K x) {return karg(x, torch::argmax, torch::argmax_out, "argmax");}
KAPI Argmin(K x) {return karg(x, torch::argmin, torch::argmin_out, "argmin");}

// --------------------------------------------------------------------------
// sort - sort by dimension,optional descending,stable flag and output vector
// --------------------------------------------------------------------------
KAPI Sort(K x) {
 KTRY
  int64_t d=-1; bool p=false,dn=false,s=false,st=false; Tensor a; TensorVector *v=nullptr;
  if(xten(x,a)) {
   p=true;
  } else if(xarray(x,5)) {
   a=kput(x);
  } else {
   J n=x->n;
   TORCH_CHECK(n>0 && n<6, "sort: expecting up to 5 args, (input;dim;descend;stable;output), given ",x->n);
   if(n>1 && (v=xvec(x,n-1))) n--;
   if(n>2 && xbool(x,n-2,dn) && xbool(x,n-1,st)) s=true, n-=2;
   if(n>1 && xbool(x,n-1,dn)) n--;
   if(n>1 && xint64(x,n-1,d)) n--;
   TORCH_CHECK(n==1, "sort: unrecognized args, expecting up to 5, (input;dim;descend;stable;output)");
   if(!(p=xten(x,0,a))) a=kput(x,0);
  }
  if(v)   // output vector supplied
   return koutput(*v, s ? torch::sort(a,st,d,dn) : torch::sort(a,d,dn));
  else
   return kresult(p, s ? torch::sort(a,st,d,dn) : torch::sort(a,d,dn));
 KCATCH("sort");
}

// ------------------------------------------------------------------
// argsort - sort call, but only return indices, args (x;dim;descend)
// ------------------------------------------------------------------
KAPI Argsort(K x) {
 KTRY
  int64_t d=-1; bool p=false,dn=false; Tensor a;
  if(xten(x,a)) {
   p=true;
  } else if(xarray(x,5)) {
   a=kput(x);
  } else {
   J n=x->n;
   TORCH_CHECK(n>0 && n<4, "argsort: expecting up to 3 args, (input;dim;descend), given ",x->n);
   if(n>1 && xbool(x,n-1,dn)) n--;
   if(n>1 && xint64(x,n-1,d)) n--;
   TORCH_CHECK(n==1, "argsort: unrecognized args, expecting up to 3, (input;dim;descend)");
   if(!(p=xten(x,0,a))) a=kput(x,0);
  }
  return kresult(p, torch::argsort(a,d,dn));
 KCATCH("argsort");
}

// ----------------------------------------------------------------------------------
// topk - largest/smallest k values by dimension & sort flag, return values & indices
// ----------------------------------------------------------------------------------
KAPI Topk(K x) {
 KTRY
  int64_t d=-1,k; bool p=false,l=true,s=true; Tensor a; TensorVector *v=nullptr;
  TORCH_CHECK(!x->t, "topk: not implemented for ",kname(x));
  J n=x->n;
  TORCH_CHECK(n>1 && n<7, "topk: expecting 2-6 args, (input;k;dim;largest;sort;output), given ",x->n);
  if(n>2 && (v=xvec(x,n-1))) n--;
  TORCH_CHECK(n<6, "topk: unrecognized 6th argument, expecting outout vector, given ",kname(x,5));
  TORCH_CHECK(xint64(x,1,k), "topk: expecting number of values as 2nd arg, given ",kname(x,1));
  if(n>2) {
    if(xint64(x,2,d)) {
     TORCH_CHECK(n<4 || xbool(x,3,l), "topk: unrecognized 4th arg, expecting largest flag, given ",kname(x,3));
     TORCH_CHECK(n<5 || xbool(x,4,l), "topk: unrecognized 5th arg, expecting sort flag, given ",kname(x,4));
    } else {
     TORCH_CHECK(xbool(x,2,l),        "topk: unrecognized 3rd arg, expecting dimension or largest flag, given ",kname(x,2));
     TORCH_CHECK(n<4 || xbool(x,3,s), "topk: unrecognized 4th arg, expecting sort flag, given ",kname(x,3));
     TORCH_CHECK(n<5,                 "topk: unexpected 5th arg, ",kname(x,4));
    }
  }
  if(!(p=xten(x,0,a))) a=kput(x,0);
  return v ? koutput(*v,torch::topk(a,k,d,l,s)) : kresult(p,torch::topk(a,k,d,l,s));
 KCATCH("topk");
}

// ---------------------------------------------------------------------
// kthvalue - return the kth smallest value,index by optional dimension
// ---------------------------------------------------------------------
KAPI Kthvalue(K x) {
 KTRY
  int64_t d=-1,k; bool p=false,kd=false; Tensor a; TensorVector *v=nullptr;
  TORCH_CHECK(!x->t, "kthvalue: not implemented for ",kname(x));
  J n=x->n;
  TORCH_CHECK(n>1 && n<6, "kthvalue: expecting 2-5 args, (input;k;dim;keepdim;output), given ",x->n);
  if(n>2 && (v=xvec(x,n-1))) n--;
  TORCH_CHECK(n<5, "kthvalue: unrecognized 5th argument, expecting outout vector, given ",kname(x,5));
  TORCH_CHECK(xint64(x,1,k), "kthvalue: expecting k-th value as 2nd arg, given ",kname(x,1));
  if(n>2) {
    if(xint64(x,2,d)) {
     TORCH_CHECK(n<4 || xbool(x,3,kd), "kthvalue: unrecognized 4th arg, expecting keepdim flag, given ",kname(x,3));
    } else {
     TORCH_CHECK(xbool(x,2,kd), "kthvalue: unrecognized 3rd arg, expecting dim or keepdim flag, given ",kname(x,2));
     TORCH_CHECK(n<4,           "kthvalue: unexpected 4th arg, ",kname(x,3));
    }
  }
  if(!(p=xten(x,0,a))) a=kput(x,0);
  return v ? koutput(*v, torch::kthvalue(a,k,d,kd)) : kresult(p, torch::kthvalue(a,k,d,kd));
 KCATCH("kthvalue");
}

// ---------------------------------------------------------------------
// isin - return x in y, where x & y can be scalar/array/tensor
// ---------------------------------------------------------------------
KAPI In(K x) {
 KTRY
  bool p=false,u=false,i=false; Scalar sa,sb; Tensor a,b,r;
  TORCH_CHECK(!x->t, "In: not implemented for ",kname(x));
  J n=x->n;
  TORCH_CHECK(n>1 && n<6, "In: expecting 2-5 args, (x;y;unique;invert;output), given ",x->n);
  if(n>2 && xten(x,n-1,r)) n--;
  if(xten(x,0,a)) p=true; else if(!xscalar(x,0,sa)) a=kput(x,0);
  if(xten(x,1,b)) p=true; else if(!xscalar(x,1,sb)) b=kput(x,1);
  TORCH_CHECK(a.defined() || b.defined(), "In: not implemented for inputs that are both scalars");
  TORCH_CHECK(n<3 || xbool(x,2,u), "In: expecting 3rd arg to be unique flag, given ",kname(x,2));
  TORCH_CHECK(n<4 || xbool(x,3,i), "In: expecting 4th arg to be invert flag, given ",kname(x,3));
  if(r.defined()) {
   return a.defined() && b.defined() ? torch::isin_out(r,a,b,u,i)
                      : (a.defined() ? torch::isin_out(r,a,sb,u,i) 
                                     : torch::isin_out(r,sa,b,u,i)),
          (K)0;
  } else {
   return kresult(p, a.defined() && b.defined() ? torch::isin(a,b,u,i)
                                 : (a.defined() ? torch::isin(a,sb,u,i) 
                                                : torch::isin(sa,b,u,i)));
  }
 KCATCH("In");
}

// ------------------------------------------------------------------------
//  windowing functions: bartlett, blackman, hann, hamming & kaiser window
// ------------------------------------------------------------------------
static K kwindow(K x,I m,const char* c) { // m: 0-bartlett, 1-blackman, 2-hann, 3-hamming, 4-kaiser
 KTRY
  bool p; J w; double a,b; Tensor t; TensorOptions o;
  J n=xopt(x,x->n-1,o) ? x->n-1 : xlen(x);
  if(xlong(x,w) ||
    (n==1 && xlong(x,0,w))||
    (n==2 && xlong(x,0,w) && xbool(x,1,p)) ||
    (n==3 && xlong(x,0,w) && xbool(x,1,p) && xnum(x,2,a) && m>=3) ||
    (n==4 && xlong(x,0,w) && xbool(x,1,p) && xnum(x,2,a) && xnum(x,3,b) && m==3)) {
   if(!o.has_dtype()) o=o.dtype(torch::get_default_dtype());
   switch(m) {
    case 0: t=(n==1) ? torch::bartlett_window(w,o) : torch::bartlett_window(w,p,o); break;
    case 1: t=(n==1) ? torch::blackman_window(w,o) : torch::blackman_window(w,p,o); break;
    case 2: t=(n==1) ? torch::hann_window(w,o)     : torch::hann_window(w,p,o);     break;
    case 3: if(n==1) t=torch::hamming_window(w,o);
       else if(n==2) t=torch::hamming_window(w,p,o);
       else if(n==3) t=torch::hamming_window(w,p,a,o);
       else if(n==4) t=torch::hamming_window(w,p,a,b,o);
       break;
    case 4: if(n==1) t=torch::kaiser_window(w,o);
       else if(n==2) t=torch::kaiser_window(w,p,o);
       else if(n==3) t=torch::kaiser_window(w,p,a,o);
       break;
    default: TORCH_ERROR("unrecognized windowing mode, expecting 0-4, received: ",m); break;
   }
  } else {
   if(m<3) {
    TORCH_ERROR(c," expects arg(s) of length, (length;tensor attributes), (length;periodic;tensor attributes)");
   } else if(m==3) {
    TORCH_ERROR(c," expects arg(s) of length, (length;periodic), (length;periodic;alpha) or (length;periodic;alpha;beta), along w'optional tensor attributes");
   } else {
    TORCH_ERROR(c," expects arg(s) of length, (length;periodic) or (length;periodic;beta), along w'optional tensor attributes");
   }
  }
  return kten(t);
 KCATCH(c);
}

KAPI Bartlett(K x) {return kwindow(x, 0, "bartlett");}
KAPI Blackman(K x) {return kwindow(x, 1, "blackman");}
KAPI     Hann(K x) {return kwindow(x, 2, "hann");}
KAPI  Hamming(K x) {return kwindow(x, 3, "hamming");}
KAPI   Kaiser(K x) {return kwindow(x, 4, "kaiser");}

// -------------------------------------------------------------------------------
// Fast Fourier Transform - utilities
// fftnorm - check k sym for valid FFT norm string, error else
// -------------------------------------------------------------------------------
static optstr fftnorm(K x) {
 static std::array<S,3> s={{cs("forward"),cs("backward"),cs("ortho")}};
 if(nullsym(x)) {
  return c10::nullopt;
 } else {
  for(const auto c:s)
   if(c == x->s) return c10::string_view((const char*)c); // return c with string_view(?)
  TORCH_ERROR("unrecognized fft norm: ",x->s);
 }
}

static bool fftnorm(K x,optstr& s) {
 if(x->t == -KS)
  return s=fftnorm(x), true;
 else
  return false;
}

static bool fftnorm(K x,J i,optstr& s) {
 return xind(x,i) && fftnorm(kK(x)[i],s);
}

// ----------------------------------------------------------------------------------------------------
// 1-d Fast Fourier transforms
// ----------------------------------------------------------------------------------------------------
// ffd1 - process arg(s) (input;n;dim;norm)
// fft/ifft   - 1-d fft & inverse over a single given dimension
// rfft/irfft - 1-dimensional fft & inverse of real input with onesided Hermitian output.
// hfft/ihfft - 1-d fft of a onesided Hermitian signal and inverse of real-valued fourier domain signal
// ----------------------------------------------------------------------------------------------------
static K ffd1(K x,
              Tensor  (*f)(const Tensor&,symint,int64_t,optstr),
              Tensor& (*g)(Tensor&,const Tensor&,optint,int64_t,optstr),
              const char* c) {
 KTRY
  bool p=false; optint n=c10::nullopt; int64_t d=-1; optstr s=c10::nullopt; Tensor a,r;
  if(xten(x,a)) {
   p=true;
  } else if(xarray(x,5)) {
   a=kput(x);
  } else {
   J j=x->n;
   TORCH_CHECK(j>0 && j<6, c,": expecting up to 5 args, (input;size;dim;norm;output), given ",x->j);
   if(j>1 && (xten(x,j-1,r))) j--;
   if(j>1 && fftnorm(x,j-1,s)) j--;
   if(j>2 && xint64(x,j-2,n) && xint64(x,j-1,d)) j-=2;
   if(j>1 && xint64(x,j-1,n)) j--;
   TORCH_CHECK(j==1, c,": unrecognized args, expecting up to 5, (input;size;dim;norm;output)");
   if(!(p=xten(x,0,a))) a=kput(x,0);
  }
  if(r.defined())
   return g(r,a,n,d,s), (K)0;
  else
   return kresult(p, f(a,n,d,s));
 KCATCH(c);
}

KAPI   fft(K x) {return ffd1(x, torch::fft::fft,   torch::fft_fft_out,   "fft");}
KAPI  ifft(K x) {return ffd1(x, torch::fft::ifft,  torch::fft_ifft_out,  "ifft");}
KAPI  rfft(K x) {return ffd1(x, torch::fft::rfft,  torch::fft_rfft_out,  "rfft");}
KAPI irfft(K x) {return ffd1(x, torch::fft::irfft, torch::fft_irfft_out, "irfft");}
KAPI  hfft(K x) {return ffd1(x, torch::fft::hfft,  torch::fft_hfft_out,  "hfft");}
KAPI ihfft(K x) {return ffd1(x, torch::fft::ihfft, torch::fft_ihfft_out, "ihfft");}

// ----------------------------------------------------------------------------------------------------
// 2-d Fast Fourier transforms
// ----------------------------------------------------------------------------------------------------
// ffd2 - process args for 2-d Fourier transforms (input;size;dim;norm;output)
// fft2/ifft2 - 2-d fft & inverse over two dimensions
// rfft2/irfft2 - 2-d fft and inverse of real input (returns a onesided Hermitian output)
// hfft2/ihfft2 - 2-d fft of a onesided Hermitian signal & inverse of real-valued fourier domain signal
// ----------------------------------------------------------------------------------------------------
static K ffd2(K x,
              Tensor  (*f)(const Tensor&,optsize,IntArrayRef,optstr),
              Tensor& (*g1)(Tensor&,const Tensor&,optsize,IntArrayRef,optstr),
              const Tensor& (*g2)(const Tensor&,const Tensor&,optsize,IntArrayRef,optstr),
              const char* c) {
 KTRY
  bool p=false; IntArrayRef n,d; optstr s=c10::nullopt; Tensor a,r;
  if(xten(x,a)) {
   p=true;
  } else if(xarray(x,5)) {
   a=kput(x);
  } else {
   J j=x->n;
   TORCH_CHECK(j>0 && j<6, c,": expecting up to 5 args, (input;size;dim;norm;output), given ",x->j);
   if(j>1 && (xten(x,j-1,r))) j--;
   if(j>1 && fftnorm(x,j-1,s)) j--;
   if(j>2 && xsize(x,j-2,n) && xsize(x,j-1,d)) j-=2;
   if(j>1 && xsize(x,j-1,n)) j--;
   TORCH_CHECK(j==1, c,": unrecognized args, expecting up to 5, (input;size;dim;norm;output)");
   if(!(p=xten(x,0,a))) a=kput(x,0);
  }
  optsize sz=c10::nullopt; if(n.size()) sz=n;
  std::array<int64_t,2> dm={-2,-1}; if(!d.size()) d=dm;
  if(r.defined())
   return (g1 ? g1(r,a,sz,d,s) : g2(r,a,sz,d,s)), (K)0;
  else
   return kresult(p, f(a,sz,d,s));
 KCATCH(c);
}

KAPI   fft2(K x) {return ffd2(x, torch::fft::fft2,   torch::fft_fft2_out,   nullptr, "fft2");}
KAPI  ifft2(K x) {return ffd2(x, torch::fft::ifft2,  torch::fft_ifft2_out,  nullptr, "ifft2");}
KAPI  rfft2(K x) {return ffd2(x, torch::fft::rfft2,  torch::fft_rfft2_out,  nullptr, "rfft2");}
KAPI irfft2(K x) {return ffd2(x, torch::fft::irfft2, torch::fft_irfft2_out, nullptr, "irfft2");}
KAPI  hfft2(K x) {return ffd2(x, torch::fft::hfft2,  nullptr, torch::fft_hfft2_out,  "hfft2");}
KAPI ihfft2(K x) {return ffd2(x, torch::fft::ihfft2, nullptr, torch::fft_ihfft2_out, "ihfft2");}

// ----------------------------------------------------------------------------------------------------
// n-dimensional Fast Fourier transforms
// ----------------------------------------------------------------------------------------------------
// fftn/ifftn - n-dimensional fft & inverse transform over n given dimensions
// rfftn/irfftn - n-dimensional FFT/inverse of real input with onesided Hermitian output
// hfftn/ihfftn - n-dim fft of onesided Hermitian signal & inverse of real-valued fourier domain signal
// ----------------------------------------------------------------------------------------------------
static K ffdn(K x,
              Tensor  (*f)(const Tensor&,optsize,optdim,optstr),
              Tensor& (*g1)(Tensor&,const Tensor&,optsize,optdim,optstr),
              const Tensor& (*g2)(const Tensor&,const Tensor&,optsize,optdim,optstr),
              const char* c) {
 KTRY
  bool p=false; IntArrayRef n,d; optstr s=c10::nullopt; Tensor a,r;
  if(xten(x,a)) {
   p=true;
  } else if(xarray(x,5)) {
   a=kput(x);
  } else {
   J j=x->n;
   TORCH_CHECK(j>0 && j<6, c,": expecting up to 5 args, (input;size;dim;norm;output), given ",x->j);
   if(j>1 && (xten(x,j-1,r))) j--;
   if(j>1 && fftnorm(x,j-1,s)) j--;
   if(j>2 && xsize(x,j-2,n) && xsize(x,j-1,d)) j-=2;
   if(j>1 && xsize(x,j-1,n)) j--;
   TORCH_CHECK(j==1, c,": unrecognized args, expecting up to 5, (input;size;dim;norm;output)");
   if(!(p=xten(x,0,a))) a=kput(x,0);
  }
  optsize sz=c10::nullopt; optdim dm=c10::nullopt; Ksize v;
  if(n.size()) sz=n;
  if(d.size()) dm=d;
  if(r.defined())
   return (g1 ? g1(r,a,sz,dm,s) : g2(r,a,sz,dm,s)), (K)0;
  else
   return kresult(p, f(a,sz,dm,s));
 KCATCH(c);
}

// anomalies in the declarations for hfftn/ihfftn
// definitions in libtorch/include/ATen/ops/fft_hfftn.h/ihfftn.h have optional sizes & dims
// but definitions in libtorch/include/torch/csrc/api/include/torch/fft.h use non-optional dim ??
// similar problems for hfttn_out/ihfftn_out: const Tensor& result vs Tensor&
KAPI   fftn(K x) {return ffdn(x, torch::fft::fftn,   torch::fft_fftn_out,   nullptr, "fftn");}
KAPI  ifftn(K x) {return ffdn(x, torch::fft::ifftn,  torch::fft_ifftn_out,  nullptr, "ifftn");}
KAPI  rfftn(K x) {return ffdn(x, torch::fft::rfftn,  torch::fft_rfftn_out,  nullptr, "rfftn");}
KAPI irfftn(K x) {return ffdn(x, torch::fft::irfftn, torch::fft_irfftn_out, nullptr, "irfftn");}
KAPI  hfftn(K x) {return ffdn(x, torch::fft_hfftn,   nullptr, torch::fft_hfftn_out,  "hfftn");}
KAPI ihfftn(K x) {return ffdn(x, torch::fft_ihfftn,  nullptr, torch::fft_ihfftn_out, "ihfftn");}

// ---------------------------------------------------------------------------------------------------
// fshift - process arg(s) for fft shift utilities
// fftshift - reorders n-dimensional FFT output to have negative frequency terms first, via torch.roll
// ifftshift -  inverse transform, from centered Fourier space back to centered spatial data
// ---------------------------------------------------------------------------------------------------
static K fshift(K x,Tensor (*f)(const Tensor&,optdim),const char* c) {
 KTRY
  bool p=false; optdim d=c10::nullopt; Tensor a;
  if(xten(x,a)) {
   p=true;
  } else if(xarray(x,2)) {
   a=kput(x);
  } else {
   IntArrayRef z;
   TORCH_CHECK(!x->t && x->n==2, c,": unexpected arg(s), expecting (input;dim)");
   TORCH_CHECK(xsize(x,1,z), c,": expected 2nd arg of dimension(s), given ",kname(x,1));
   if(!(p=xten(x,0,a))) a=kput(x,0);
   if(z.size()) d=z;
  }
  return kresult(p, f(a,d));
 KCATCH(c);
}

KAPI  fftshift(K x) {return fshift(x, torch::fft::fftshift,  "fftshift");}
KAPI ifftshift(K x) {return fshift(x, torch::fft::ifftshift, "ifftshift");}

// ----------------------------------------------------------------------------------------------------
// ftfreq - process args (n;d;options/output)
// fftfreq - discrete Fourier Transform sample frequencies for a signal of size n
// rfftfreq - computes the sample frequencies for rfft with a signal of size n.
// ----------------------------------------------------------------------------------------------------
static K ftfreq(K x,
                Tensor  (*f)(int64_t,double,const TensorOptions&),
                Tensor& (*g)(Tensor&,int64_t,double),
                const char* c) {
 KTRY
  int64_t n=nj; double d=1.0; Tensor r; TensorOptions o;
  if(x->t == -KJ) {
   n=x->j;
  } else if(x->t == KJ) {
   TORCH_CHECK(x->n==2, c,": expecting FFT length and spacing but given ",x->n," element(s)");
   n=kJ(x)[0]; d=kJ(x)[1];
  } else {
   TORCH_CHECK(!x->t, c,": unrecognized arg(s), expecting (length;scale;options/output), given ",kname(x));
   TORCH_CHECK(xint64(x,0,n), c,": expecting FFT length as first argument, given ",kname(x,0));
   J j=x->n;
   if(j>1 && xopt(x,j-1,o)) j--;
   if(j>1 && xten(x,j-1,r)) j--;
   if(j>1 && xnum(x,j-1,d)) j--;
   TORCH_CHECK(j==1, c,": unrecognized arg(s), expecting (length;scale;options/output)");
  }
  TORCH_CHECK(n>=0, c,": length cannot be negative, given ",n);
  return r.defined() ? (g(r,n,d), (K)0) : kten(f(n,d,o));
 KCATCH(c);
}

KAPI  fftfreq(K x) {return ftfreq(x, torch::fft::fftfreq,  torch::fft_fftfreq_out,  "fftfreq");}
KAPI rfftfreq(K x) {return ftfreq(x, torch::fft::rfftfreq, torch::fft_rfftfreq_out, "rfftfreq");}

/*
Tensor stft(const Tensor & self,J n_fft,J hop_length,J win_length,const Tensor& window={},bool normalized=false,bool onesided=true);
torch.stft(input, n_fft, hop_length=None, win_length=None, window=None, center=True, pad_mode='reflect', normalized=False, onesided=True)
      stft(input, n_fft, hop_length, win_length, window, normalized, onesided)
*/

/* OTHER
broadcast_tensors(*tensors) -> List of tensors
einsum(equation, *operands) -> tensor
meshgrid(*tensors, **kwargs)
*/

// -------------------------------------------------------------------------------------
// multidot - multiply series of matrices
// bincount - frequency of each value in an array of non-negative integers
// flip - reverse the order of a tensor along given dim(s)
// trace - return sum of elements of diagonal of 2-d matrix
// -------------------------------------------------------------------------------------
KAPI Multidot(K x) {
 KTRY
  bool p; const TensorVector& v=xtensors(x,p,"multidot");
  return kresult(p, torch::linalg_multi_dot(v));
 KCATCH("multidot");
}

KAPI Bincount(K x) {
 KTRY
  bool p=false; J m=0; Tensor t,w,r;
  TORCH_CHECK(x->t>=0,"bincount: not implemented for ",kname(x));
  if(x->t) {
   t=kput(x);
  } else if(xten(x,t)) {
   p=true;
  } else if(x->n==2 && xlong(x,1,m)) {
   if(!(p=xten(x,0,t))) t=kput(x,0);
  } else if(x->n==2 || (x->n==3 && xlong(x,2,m))) {
   p=xtenarg(x,t,w);
  } else {
   TORCH_ERROR("bincount expects 1-3 args, input, (input;bins), (input;weight) or (input;weight;bins)");
  }
  return kresult(p, torch::bincount(t,w,m));
 KCATCH("bincount");
}

KAPI Flip(K x) {
 KTRY
  Tensor t; IntArrayRef s;
  if(xsize(x,1,s)) {
   TORCH_CHECK(x->n==2, "Flip: expecting 2 arguments, (input;dims), given ", x->n);
   auto *t=xten(x,0);
   return kresult(t, torch::flip(t ? *t : kput(x,0), s));
  } else {
   TORCH_CHECK(x->t==KJ && x->n==2, "Flip: unrecognized arg(s), expecting 2 arguments, (input;dims)");
   auto t=kput(x);
   return kget(torch::flip(t[0], t[1].item<int64_t>()));
  }
 KCATCH("Flip");
}

KAPI Trace(K x) {
 KTRY
  Tensor t;
  return xten(x,t) ? kten(torch::trace(t)) : kget(torch::trace(kput(x)));
 KCATCH("trace");
}

// -------------------------------------------------------------------------------------
// diag1: process k args (input;offset;output)
// diag/diagflat - extract diagonal or create matrix from diagonal
// diagonal - extract diagonal from matrix or higher dimensional input
// -------------------------------------------------------------------------------------
static K diag1(K x,
               Tensor  (*f)(const Tensor&, int64_t),          // function call
               Tensor& (*g)(Tensor&,const Tensor&,int64_t),   // function w'output
               const char* c) {
 KTRY
  TORCH_CHECK(-1<x->t && x->t<11, c,": not implemented for ",kname(x));

  bool p=false; int64_t d=0; Tensor a,r;
  if(xten(x,a)) {
   p=true;
  } else if(xarray(x,3)) {
   a=kput(x);
  } else {
   J n=x->n;
   TORCH_CHECK(n>0 && n<4, c,": expecting up to 3 args, (input;offset;output), given ",x->n);
   if(n>1 && (xten(x,n-1,r))) n--;
   if(n>1 && xint64(x,n-1,d)) n--;
   TORCH_CHECK(n==1, c,": unrecognized arg(s), expecting up to 3, (input;offset;output)");
   if(!(p=xten(x,0,a))) a=kput(x,0);
  }
  if(r.defined()) {
   TORCH_CHECK(g, c,": output tensor not implemented");
   return g(r,a,d), (K)0;
  } else {
   return kresult(p, f(a,d));
  }
 KCATCH(c);
}

KAPI Diag(K x)     {return diag1(x, torch::diag,     torch::diag_out,  "diag");}
KAPI Diagflat(K x) {return diag1(x, torch::diagflat, nullptr,          "diagflat");}

KAPI Diagonal(K x) {  //extract diagonal elements, optional offset & dimensions i,j
 KTRY
  bool p; J o=0,i=0,j=1; Tensor t;
  if(x->t) {
   TORCH_ERROR("diagonal: not implemented for ",kname(x->t));
  } else if(xlong(x,1,o) && (x->n==2 || (xlong(x,2,i) && (x->n==3 || (x->n==4 && xlong(x,3,j)))))) {
   if(!(p=xten(x,0,t))) t=kput(x,0);
  } else {
   if(!(p=xten(x,t))) t=kput(x);
  }
  return kresult(p, torch::diagonal(t,o,i,j));
 KCATCH("diagonal");
}

// -------------------------------------------------
// triarg - process args for (input;offset;output)
// tril/triu - api fns for lower & upper triangles
// ------------------------------------------------
static K triarg(K x,
                Tensor  (*f)(const Tensor&, int64_t),          // function call
                Tensor& (*g)(Tensor&,const Tensor&,int64_t),   // function w'output
                Tensor& (Tensor::*m)(int64_t) const,           // method
                const char* c) {
 KTRY
  TORCH_CHECK(-1<x->t && x->t<11, c,": not implemented for ",kname(x));

  bool e=false,p=false; int64_t d=0; Tensor a,r;
  if(xten(x,a)) {
   p=true;
  } else if(xarray(x,3)) {
   a=kput(x);
  } else {
   J n=x->n;
   TORCH_CHECK(n>0 && n<4, c,": expecting up to 3 args, (input;dim;output), given ",x->n);
   if(n>1 && (xten(x,n-1,r))) n--;
   if(n>1 && (xempty(x,n-1))) e=true, n--;
   if(n>1 && xint64(x,n-1,d)) n--;
   TORCH_CHECK(n==1, c,": unrecognized arg(s), expecting up to 3, (input;dim;output)");
   if(!(p=xten(x,0,a))) a=kput(x,0);
  }
  if(e) {
   TORCH_CHECK(m, c,": no in-place method");
   return (a.*m)(d), (K)0;
  } else if(r.defined()) {
   TORCH_CHECK(g, c,": output tensor not implemented");
   return g(r,a,d), (K)0;
  } else {
   return kresult(p, f(a,d));
  }
 KCATCH(c);
}

KAPI Tril(K x)     {return triarg(x, torch::tril,     torch::tril_out,  &Tensor::tril_, "tril");}
KAPI Triu(K x)     {return triarg(x, torch::triu,     torch::triu_out,  &Tensor::triu_, "triu");}

// -------------------------------------------------------------------------------------------
// histc - histogram of inputs with optional number of bins, min & max bin, output tensor
// cross - cross product of two tensors/arrays, with optional dimension & output tensor
// renorm - renormalize along given dimension using given p-norm exponent and max norm
// roll - specify single or multiple shifts & dimensions to rotate tensor
// tensordot - returns a contraction of a and b over multiple dimensions (new for version 1.0)
// unique - return unique elements in input along with optional indicies
// -------------------------------------------------------------------------------------------
KAPI Histc(K x) {
 KTRY
  J b=100,p; Scalar lo=0,hi=0; Tensor r,t;
  if(x->t)
   return kget(torch::histc(kput(x)));
  else if(xten(x,t))
   return kten(torch::histc(t));
  J n=(x->n>1 && xten(x,x->n-1,r)) ? x->n-1 : x->n;  //arg count excluding output tensor if supplied
  TORCH_CHECK(n==1 || (xlong(x,1,b) && (n==2 || (xnum(x,2,lo) && (n==3 || (xnum(x,3,hi) && n==4))))), 
             "histc: unrecognized arg(s), expecting 1-5 args, (input;bins;lo;hi;output)");
  if( !(p=xten(x,0,t)) )
   t=kput(x,0);
  if(r.defined())
   return torch::histc_out(r,t,b,lo,hi), (K)0;
  else
   return kresult(p, torch::histc(t,b,lo,hi));
 KCATCH("histc");
}

KAPI Cross(K x) {
 KTRY
  TORCH_CHECK(!x->t, "cross: not implemented for ",kname(x));
  Tensor a,b,r; J n=x->n; int64_t d=-1;
  TORCH_CHECK(1<n && n<5, "cross: expecting 2-4 args, (x;y;dim;output), given ",n);
  if(n>2 &&   xten(x,n-1,r)) n--;
  if(n>2 && xint64(x,n-1,d)) n--;
  TORCH_CHECK(n==2, "cross: unexpected args(), expecting 2-4 args, (x;y;dim;output)");
  bool p=xtenarg(x,a,b);
  return r.defined() ? (torch::cross_out(r,a,b,d), (K)0) : kresult(p, torch::cross(a,b,d));
 KCATCH("cross");
}

KAPI Renorm(K x) {
 KTRY
  bool b; J d; Scalar p,m; Tensor r,t;
  TORCH_CHECK(xnum(x,1,p) && xlong(x,2,d) && xnum(x,3,m) && (x->n==4 || (x->n==5 && xten(x,4,r))),
              "renorm: unexpected arg(s), expecting 4-5 args, (input;power;dim;maxnorm;output)");
  if(!(b=xten(x,0,t))) t=kput(x,0);
  if(r.defined()) 
   return torch::renorm_out(r,t,p,d,m), (K)0;
  else
   return kresult(b, torch::renorm(t,p,d,m));
 KCATCH("renorm");
}

KAPI Roll(K x) {
 KTRY
  bool p; IntArrayRef s,d; Tensor t;
  TORCH_CHECK(xsize(x,1,s) && (x->n==2 || (xsize(x,2,d) && x->n==3)),
             "roll: unexpected arg(s), expects 2-3 args, (input;shifts;dims)");
  if(!(p=xten(x,0,t))) t=kput(x,0);
  return kresult(p, x->n==2 ? torch::roll(t,s) : torch::roll(t,s,d));
 KCATCH("roll");
}

KAPI Tensordot(K x) {
 KTRY
  J d=2; IntArrayRef i,j; Tensor a,b;
  TORCH_CHECK(x->t>=0, "tensordot: not implemented for ",kname(x));
  if(x->t && (x->n==2 || x->n==3)) {
   a=kput(x);
   if(x->n==3 && a[2].item().toDouble()!=0)
    TORCH_ERROR("tensordot: non-zero dimension specified for scalars");
   return kget(torch::tensordot(a[0],a[1],i,j));
  } else if(x->n==2 || (x->n==3 && xlong(x,2,d)) || (x->n==4 && xsize(x,2,i) && xsize(x,3,j))) {
   bool p=xtenarg(x,a,b); Ksize s1,s2;
   if(x->n<4) {
    for(I k=0;k<d;++k) s1.push_back(k-d), s2.push_back(k);
    i=s1; j=s2;
   }
   return kresult(p, torch::tensordot(a,b,i,j));
  } else {
   TORCH_ERROR("tensordot: unexpected arg(s), expecting inputs (x;y), (x;y;dim) or (x;y;dim1;dim2)");
  }
 KCATCH("tensordot");
}

// -----------------------------------------------------------------------------------
// uresult - return tensor or 2-3 element vector of tensors depending on input & flags
// unique - return unique elements in input along with optional indicies & counts
// uniquec - return first occurrence from consecutive group of like elements
// -----------------------------------------------------------------------------------
static K uresult(bool p,bool bi,bool bc,const Tuple3& t) {
 if(bi && bc)
  return kresult(p, t); //values,indices,counts
 else if(bi)
  return kresult(p, TensorVector{std::get<0>(t),std::get<1>(t)}); // unique values & indices
 else if(bc)
  return kresult(p, TensorVector{std::get<0>(t),std::get<2>(t)}); // unique values & counts
 else
  return kresult(p, std::get<0>(t));
}

KAPI Unique(K x) {
 KTRY
  bool p=false,bs=true,bi=false,bc=false; c10::optional<int64_t> d=c10::nullopt; J n=xlen(x); Tensor t;
  if(xten(x,t)) {
   p=true;
  } else if(xarray(x,5)) {
   t=kput(x);
  } else {
   if (xint64(x,1,d)){
    TORCH_CHECK( n==2 ||
                (n==3 && xbool(x,2,bs)) ||
                (n==4 && xbool(x,2,bs) && xbool(x,3,bi)) ||
                (n==5 && xbool(x,2,bs) && xbool(x,3,bi) && xbool(x,4,bc)),
                "unique: unrecognized arg(s), given input & dim, up to 3 additional flags expected for sorted,indices & counts");
   } else {
    TORCH_CHECK((n==2 && xbool(x,1,bs)) ||
                (n==3 && xbool(x,1,bs) && xbool(x,2,bi)) ||
                (n==4 && xbool(x,1,bs) && xbool(x,2,bi) && xbool(x,3,bc)),
                "unique: unrecognized arg(s), expecting input, optional dim and flag(s) for sorted,indices & counts");
   }
  if(!(p=xten(x,0,t))) t=kput(x,0);
  }
  return uresult(p,bi,bc,d ? torch::unique_dim(t,*d,bs,bi,bc) : torch::_unique2(t,bs,bi,bc));
 KCATCH("unique");
}

KAPI Uniquec(K x) {
 KTRY
  bool p=false,bi=false,bc=false; c10::optional<int64_t> d=c10::nullopt; J n=xlen(x); Tensor t;
  if(xten(x,t)) {
   p=true;
  } else if(xarray(x,4)) {
   t=kput(x);
  } else {
   if (xint64(x,1,d)){
    TORCH_CHECK( n==2 ||
                (n==3 && xbool(x,2,bi)) ||
                (n==4 && xbool(x,2,bi) && xbool(x,3,bc)),
                "uniquec: unrecognized arg(s), given input & dim, up to 2 additional flags expected for indices & counts");
   } else {
    TORCH_CHECK((n==2 && xbool(x,1,bi)) ||
                (n==3 && xbool(x,1,bi) && xbool(x,2,bc)),
                "uniquec: unrecognized arg(s), expecting input, optional dim and flag(s) for indices & counts");
   }
   if(!(p=xten(x,0,t))) t=kput(x,0);
  }
  return uresult(p,bi,bc,torch::unique_consecutive(t,bi,bc,d));
 KCATCH("uniquec");
}

// --------------------------------------------------------------------------
// addbmm - beta * mat + alpha * sum of batch1 * batch2
// addmm - beta * mat + alpha * mat1 * mat2
// addmv - beta * vector + alpha * mat1 * vec1
// addr - beta * mat + alpha * outter product of vec1,vec2
// baddbmm - beta * batchmat + alpha * sum of batch1 * batch2
// --------------------------------------------------------------------------
static K kadd(K X,
              Tensor  (*f)(const Tensor&,const Tensor&,const Tensor&,const Scalar&,const Scalar&),
              Tensor& (*g)(Tensor&,const Tensor&,const Tensor&,const Tensor&,const Scalar&,const Scalar&),
              const char* c) {
 KTRY
  TORCH_CHECK(!X->t, c,": not implemented for ",kname(X));
  J n=X->n; Scalar a=1,b=1; Tensor x,y,z,r;
  TORCH_CHECK(2<n && n<7, c,": 3-6 args expected, (x;y;z;beta;alpha;output), ",n," given");
  if(n>3 && xten(X,n-1,r)) n--;
  if(n>4 && xnum(X,n-2,b) && xnum(X,n-1,a)) n-=2;
  if(n>3 && xnum(X,n-1,b)) n--;
  TORCH_CHECK(n, c,": unrecognized arg(s), 3-6 args expected, (x;y;z;beta;alpha;output)");
  bool p=xtenarg(X,x,y,z);
  return r.defined() ? (g(r,x,y,z,b,a), (K)0) : kresult(p, f(x,y,z,b,a));
 KCATCH(c);
}

KAPI  Addbmm(K x) {return kadd(x, torch::addbmm,  torch::addbmm_out,  "addbmm");}
KAPI   Addmm(K x) {return kadd(x, torch::addmm,   torch::addmm_out,   "addmm");}
KAPI   Addmv(K x) {return kadd(x, torch::addmv,   torch::addmv_out,   "addmv");}
KAPI    Addr(K x) {return kadd(x, torch::addr,    torch::addr_out,    "addr");}
KAPI Baddbmm(K x) {return kadd(x, torch::baddbmm, torch::baddbmm_out, "baddbmm");}

// -----------------------------------------------------------------------------
// lu  - LU decomposition w'pivot flag
// lux - LU decomposition w'pivot & error check flags, return error codes
// -----------------------------------------------------------------------------
KAPI Lu(K x) {
 KTRY
  bool p=false,pv=true; Tensor a; TensorVector *v=nullptr;
  if(xten(x,a)) {
   p=true;
  } else if(xarray(x,3)) {
   a=kput(x);
  } else {
   J n=x->n;
   TORCH_CHECK(n>0 && n<4, "lufactor: expecting 1-3 args, (x;pivot;output), given ",x->n);
   if(n>1 && (v=xvec(x,n-1))) n--;
   if(n>1 && xbool(x,n-1,pv)) n--;
   TORCH_CHECK(n==1, "lufactor: unrecognized args, expecting 1-3 args, (x;pivot;output)");
   if(!(p=xten(x,0,a))) a=kput(x,0);
  }
  return v ? koutput(*v,torch::linalg_lu_factor(a,pv)) : kresult(p,torch::linalg_lu_factor(a,pv));
 KCATCH("lufactor");
}

KAPI Lux(K x) {
 KTRY
  TORCH_CHECK(!x->t, "lux: not implemented for ",kname(x));
  bool p=false,pv=true,e=false; J n=x->n; Tensor a; TensorVector *v=nullptr;
  if(xten(x,a)) {
   p=true;
  } else if(xarray(x,4)) {
   a=kput(x);
  } else {
   TORCH_CHECK(0<n && n<5, "lux: expecting 1-4 arg(s), (x;pivot;check;output), ",n," given");
   if(n>1 && (v=xvec(x,n-1))) n--;
   if(n>2 && xbool(x,n-2,pv) && xbool(x,n-1,e)) n-=2;
   if(n>1 && xbool(x,n-1,pv)) n--;
   TORCH_CHECK(n==1, "lux: unexpected arg(s), expecting 1-4 arg(s), (x;pivot;check;output)");
   if(!(p=xten(x,0,a))) a=kput(x,0);
  }
  return v ? koutput(*v, torch::linalg_lu_factor_ex(a,pv,e))
           : kresult(p,  torch::linalg_lu_factor_ex(a,pv,e));
 KCATCH("lux");
}

// --------------------------------------------------------------
// lun - unpack data & pivots from batched LU factorization
// --------------------------------------------------------------
KAPI Lun(K x) {
 KTRY
  TORCH_CHECK(!x->t, "lun: not implemented for ",kname(x));
  bool p=false,b1=true,b2=true; J n=x->n; Tensor lu,pv; TensorVector *u=nullptr,*v=nullptr;
  TORCH_CHECK(0<n && n<6, "lun: expecting 1-5 args, (lu vector;dataflag;pivotflag;output) or (lu matrix;pivots;dataflag;pivotflag;output), ",n," arg(s) given");
  if(n>1 && ((v=xvec(x,n-1)))) n--;
  if(n>2 && xbool(x,n-2,b1) && xbool(x,n-1,b2)) n-=2;
  if(n>1 && xbool(x,n-1,b1)) n--;
  if(n==1) {
   if(!(u=xvec(x))) u=xvec(x,1);
   if(u) {
    TORCH_CHECK(u->size()>=2, "lun: first arg is ",u->size(),"-element vector, but 2 elements required (LU matrix & pivots)");
    p=true, lu=u->at(0), pv=u->at(1);
   } else {
    p=xtenarg(x,lu,pv);
   }
  } else {
   TORCH_CHECK(n==2, "lun: unrecognized inputs, expecting a single input vector or 2 arrays/tensors, but ",n," inputs given");
   p=xtenarg(x,lu,pv);
  }
  const auto& r=torch::lu_unpack(lu,pv,b1,b2);
  return v ? koutput(*v,r) : kresult(p,r);
 KCATCH("lun");
}

// --------------------------------------------------------------------------
// lusolve - batch LU solve of the linear system Ax = b from LU factorization
// --------------------------------------------------------------------------
KAPI Lusolve(K x) {
 KTRY
  TORCH_CHECK(!x->t, "lusolve: not implemented for ",kname(x));
  bool p=false; J n=x->n; Tensor b,m,pv,r; TensorVector *v;
  TORCH_CHECK(1<n && n<5, "lusolve: 2-4 arguments expected, (b;lu;output) or (b;matrix;pivot;output), but ",n," given");
  K y=kK(x)[1];
  if(xten(y) || xarray(y,5)) {
   TORCH_CHECK(n>2, "lusolve: expecting (b;matrix;pivot) or (b;matrix;pivot;output), but given ",n," args");
   p=xtenarg(x,b,m,pv);
   TORCH_CHECK(n==3 || xten(x,3,r), "lusolve: 4th arg of output tensor expected, given ",kname(x,3));
  } else {
   TORCH_CHECK(n<4, "lusolve: expecting (b;lu) or (b;lu;output), but given ",n," args");
   if((v=xvec(y))) {
    TORCH_CHECK(v->size()>=2, "lusolve: 2nd arg of tensor vector of LU output should have at least 2 elements, ",v->size()," found");
    p=true; m=v->at(0); pv=v->at(1);
   } else {
    p=xtenarg(y,m,pv);
   }
   if(xten(x,0,b))
    p=true;
   else
    b=kput(x,0);
   TORCH_CHECK(n==2 || xten(x,2,r), "lusolve: 3rd arg of output tensor expected, given ",kname(x,2));
  }
  return r.defined() ? (torch::lu_solve_out(r,b,m,pv), (K)0) : kresult(p, torch::lu_solve(b,m,pv));
 KCATCH("lusolve");
}

// ------------------------------------------------------------------------------------------
// matrix_power - raise matrix or batch of matrices to given integer power (may be negative)
// ktol - handle arg(s) (input;atol;rtol;hermitian;output)
// matrix_rank - return rank of 2-d tensor, specify optional tolerance and symmetric flag
// pinverse - pseudo inverse, with same tolerance and flag args as matrix rank
// ------------------------------------------------------------------------------------------
KAPI Matrix_power(K x) {
 KTRY
  bool p; J n; Tensor r,t;
  TORCH_CHECK(xlong(x,1,n), "power: 2nd argument of long integer n-th power expected, given ",kname(x,1));
  if(!(p=xten(x,0,t))) t=kput(x,0);
  TORCH_CHECK(x->n==2 || (x->n==3 && xten(x,2,r)), "power: unrecognized arg(s), expecting (matrix;n) or (matrix;n;output)");
  return r.defined() ? (torch::matrix_power_out(r,t,n),(K) 0) : kresult(p, torch::matrix_power(t,n));
 KCATCH("power");
}

static K ktol(K x,
              Tensor  (*f)(        const Tensor&,const c10::optional<double>,const c10::optional<double>,bool),
              Tensor& (*g)(Tensor&,const Tensor&,const c10::optional<double>,const c10::optional<double>,bool),
              const char *c) {
 KTRY
  bool h=false,p=false; double at=nf,rt=nf; Tensor a,r;
  if(xten(x,a)) {
   p=true;
  } else if(xarray(x,5)) {
   a=kput(x);
  } else {
   J n=x->n;
   TORCH_CHECK(n>0 && n<6, c,": expecting up to 5 args, (input;atol;rtol;hermitian;output), given ",x->n);
   if(n>1 && (xten(x,n-1,r))) n--;
   if(n>1 && xbool(x,n-1,h)) n--;
   if(n>2 && xdouble(x,n-2,at) && xdouble(x,n-1,rt)) n-=2;
   if(n>1 && xdouble(x,n-1,at)) n--;
   TORCH_CHECK(n==1, c,": unrecognized arg(s), expecting up to 5, (input;atol;rtol;hermitian;output)");
   if(!(p=xten(x,0,a))) a=kput(x,0);
  }
  c10::optional<double> atol=c10::nullopt,rtol=c10::nullopt;
  if(at==at) atol=at;
  if(rt==rt) rtol=rt;
  if(r.defined())
   return g(r,a,atol,rtol,h), (K)0;
  else
   return kresult(p, f(a,atol,rtol,h));
 KCATCH(c);
}

KAPI    Mrank(K x) {return ktol(x, torch::linalg_matrix_rank, torch::linalg_matrix_rank_out, "mrank");}
KAPI Pinverse(K x) {return ktol(x, torch::linalg_pinv,        torch::linalg_pinv_out,        "pinverse");}

// ----------------------------------------------------------------------------------------
// det,logdet,slogdet - determinant, log determinant & log determinant w'sign
// ----------------------------------------------------------------------------------------
static K kdet(K x,I m,const char* c) { //x:arg, m:mode 0-det,1-logdet,2-slogdet, c:name
 KTRY
  Tensor a; bool p=xten(x,a); if(!p) a=kput(x);
  return m==2 ? kresult(p, torch::slogdet(a)) : kresult(p, m ? torch::logdet(a) : torch::det(a));
 KCATCH(c);
}

KAPI Det(K x)     {return kdet(x, 0, "det");}
KAPI Logdet(K x)  {return kdet(x, 1, "logdet");}
KAPI Slogdet(K x) {return kdet(x, 2, "slogdet");}

// --------------------------------------------------------------------------------------------------
// blas2 - BLAS fns with 2 input tensors/arrays & optional output tensor, return tensor or set output
// --------------------------------------------------------------------------------------------------
static K blas2(K x,
               Tensor  (*f)(const Tensor&,const Tensor&),
               Tensor& (*g)(Tensor&,const Tensor&,const Tensor&),
               const char* c) {
 KTRY
  TORCH_CHECK(!x->t, c,": not implemented for ",kname(x));
  bool p; Tensor a,b,r; J n=x->n;
  TORCH_CHECK(1<n && n<4, c,": expecting 2-3 args, (x;y;output), given ",n);
  TORCH_CHECK(n==2 || (n==3 && xten(x,2,r)), c,": expecting 3rd arg of output tensor, given ",kname(x,2));
  p=xtenarg(x,a,b);
  return r.defined() ? (g(r,a,b), (K)0) : kresult(p, f(a,b));
 KCATCH(c);
}


static Tensor  mtm(              const Tensor& a,const Tensor&b) {return torch::mm(      a.t(),b);}
static Tensor& mtm_out(Tensor& r,const Tensor& a,const Tensor&b) {return torch::mm_out(r,a.t(),b);}
static Tensor  mmt(              const Tensor& a,const Tensor&b) {return torch::mm(      a,b.t());}
static Tensor& mmt_out(Tensor& r,const Tensor& a,const Tensor&b) {return torch::mm_out(r,a,b.t());}

KAPI Bmm(K x)    {return blas2(x, torch::bmm,     torch::bmm_out,    "bmm");}
KAPI Dot(K x)    {return blas2(x, torch::dot,     torch::dot_out,    "dot");}
KAPI Outer(K x)  {return blas2(x, torch::outer,   torch::outer_out,  "outer");}
KAPI Matmul(K x) {return blas2(x, torch::matmul,  torch::matmul_out, "matmul");}
KAPI Mm(K x)     {return blas2(x, torch::mm,      torch::mm_out,     "mm");}
KAPI Mmt(K x)    {return blas2(x, mmt,            mmt_out,           "mmt");}
KAPI Mtm(K x)    {return blas2(x, mtm,            mtm_out,           "mtm");}
KAPI Mv(K x)     {return blas2(x, torch::mv,      torch::mv_out,     "mv");}
KAPI Householder(K x)  {return blas2(x, torch::linalg_householder_product,   torch::linalg_householder_product_out,  "householder");}

// --------------------------------------------------------------------------------------
// qr - qr decomposition, returns orthoganal and upper triangular matrix
// geqrf - qr decomposition using lower level BLAS routine, returns "reflector" matrices
// ormqr - multiply mat by orthogonal Q matrix of the QR factorization formed by geqrf
// --------------------------------------------------------------------------------------
static c10::string_view qrmode(K x) {
 static std::array<S,3> s={{cs("reduced"),cs("complete"),cs("r")}};
 if(!x)
  return s[0];
 for(const auto c:s)
  if(c == x->s) return c10::string_view((const char*)c);
 TORCH_ERROR("unrecognized qr mode: ",x->s);
}

KAPI Qr(K x) {
 KTRY
  TORCH_CHECK(!x->t, "qr: not implemented for ",kname(x));
  bool p=false; c10::string_view m=qrmode(nullptr); J n=x->n; Tensor a; TensorVector *v=nullptr;
  if(xten(x,a)) {
   p=true;
  } else if(xarray(x,3)) {
   a=kput(x);
  } else {
   TORCH_CHECK(0<n && n<4, "qr: 1-3 args expected, (x;mode;output), ",n," given");
   if(n>1 && (v=xvec(x,n-1))) n--;
   if(n>1 && xsym(x,n-1)) n--, m=qrmode(kK(x)[n]);
   TORCH_CHECK(n==1, "qr: unexpected arg(s), 1-3 args expected, (x;mode;output)");
   if(!(p=xten(x,0))) a=kput(x,0);
  }
  const auto& t=torch::linalg_qr(a,m);
  return v ? koutput(*v,t) : kresult(p,t);
 KCATCH("qr");
}

// --------------------------------------------------------------------------------------
// svdsym - check symbol input for valid driver name
// svd - singular value decomposition of a real matrix, returns/writes U,S,V(h)
// svdvals - singular value decomposition, return values only
// --------------------------------------------------------------------------------------
static optstr svdsym(S s,const char *c) {
 static std::array<S,3> d={{cs("gesvd"), cs("gesvda"), cs("gesvdj")}};
 if(nullsym(s)) {
  return c10::nullopt;
 } else {
  for(const auto a:d)
   if(a == s) return c10::string_view((const char*)a);
  TORCH_ERROR(c,": unrecognized driver `",s);
 }
}

KAPI Svd(K x) {
 KTRY
  bool p=false,f=true; S s; Tensor a; TensorVector *v=nullptr; optstr d=c10::nullopt;
  TORCH_CHECK(!x->t, "svd: not implemented for ",kname(x));
  if(xten(x,a)) {
   p=true;
  } else if(xarray(x,4)) {
   a=kput(x);
  } else {
   J n=x->n;
   TORCH_CHECK(n>0 && n<5, "svd: expecting 1-4 args, (x;full;driver;out vector), given ",x->n);
   if(n>1 && (v=xvec(x,n-1))) n--;
   if(n>1 && (xsym(x,n-1,s))) d=svdsym(s,"svd"), n--;
   if(n>1 && xbool(x,n-1,f)) n--;
   TORCH_CHECK(n==1, "svd: unrecognized args, expecting 1-4 args, (a;full;driver;out vector)");
   if(!(p=xten(x,0,a))) a=kput(x,0);
  }
  return v ? koutput(*v,torch::linalg::svd(a,f,d)) : kresult(p,torch::linalg::svd(a,f,d));
 KCATCH("svd");
}

KAPI Svdvals(K x) {
 KTRY
  bool p=false; S s; Tensor a,r; optstr d=c10::nullopt;
  TORCH_CHECK(!x->t, "svdvals: not implemented for ",kname(x));
  if(xten(x,a)) {
   p=true;
  } else if(xarray(x,3)) {
   a=kput(x);
  } else {
   J n=x->n;
   TORCH_CHECK(n>0 && n<4, "svdvals: expecting 1-3 args, (x;driver;out), given ",x->n);
   if(n>1 && (xten(x,n-1,r))) n--;
   if(n>1 && (xsym(x,n-1,s))) d=svdsym(s,"svdvals"), n--;
   TORCH_CHECK(n==1, "svdvals: unrecognized args, expecting 1-3 args, (a;driver;out)");
   if(!(p=xten(x,0,a))) a=kput(x,0);
  }
  return r.defined() ? (torch::linalg::svdvals_out(r,a,d), (K)0) : kresult(p,torch::linalg::svdvals(a,d));
 KCATCH("svdvals");
}

// --------------------------------------------------------------------------------------
// linalg - solvers
// solve - solution to least squares for ax=b, given a,b returns x
// trisolve - solves ax=b w'triangular matrix a and multiple right-hand sides b
// method - BLAS/LAPACK method to use, translate from symbol to string
// lstsq - solution to least squares for ax=b, returns x,residuals,rank,singular values
// --------------------------------------------------------------------------------------
KAPI Solve(K x) {
 KTRY
  bool l=true; Tensor a,b,r;
  TORCH_CHECK(!x->t, "solve: not implemented for ",kname(x));
  TORCH_CHECK(1<x->n && x->n<5, "solve: expects 2-4 args, (a;b;left;out), given ",x->n);
  TORCH_CHECK(x->n==2 || 
             (x->n==3 && (xbool(x,2,l) || xten(x,2,r))) ||
             (x->n==4 &&  xbool(x,2,l) && xten(x,3,r)),
             "solve: unrecognized arg(s), expecting 2-4 args, (a;b), (a;b;left), (a;b;out) or (a;b;left;out)");
  auto p=xtenarg(x,a,b);
  return r.defined() ? (torch::linalg::solve_out(r,a,b,l), (K)0) : kresult(p, torch::linalg::solve(a,b,l));
 KCATCH("solve");
}

KAPI Trisolve(K x) {
 KTRY
  TORCH_CHECK(!x->t, "solve: not implemented for ",kname(x));
  J n=x->n; bool u,l=true,t=false; Tensor a,b,r;
  TORCH_CHECK(2<n && n<7, "solve: expects 3-6 args, (a;b;upper;left;unitriangular;output), given ",n);
  TORCH_CHECK(xbool(x,2,u), "solve: 3rd arg of upper triangular flag expected, given ",kname(x,2));
  if(n>3 && xten(x,n-1,r)) n--;
  if(n>4 && xbool(x,n-2,l) && xbool(x,n-1,t)) n-=2;
  if(n>3 && xbool(x,n-1,l)) n--;
  TORCH_CHECK(n==3, "solve: unrecognized args, expecting 3-6 args, (a;b;upper;left;unitriangular;output)");
  bool p=xtenarg(x,a,b);
  return r.defined() ? (torch::linalg::solve_triangular_out(r,a,b,u,l,t), (K)0) 
                     :  kresult(p, torch::linalg::solve_triangular(a,b,u,l,t));
 KCATCH("trisolve");
}

static optstr method(K x) {
 static std::array<S,4> s={{cs("gels"), cs("gelsd"), cs("gelss"), cs("gelsy")}};
 if(nullsym(x)) {
  return c10::nullopt;
 } else {
  for(const auto c:s)
   if(c == x->s) return c10::string_view((const char*)c); // return c with string_view(?)
  TORCH_ERROR("unrecognized driver: ",x->s);
 }
}

KAPI Lstsq(K x) {
 KTRY
  TORCH_CHECK(!x->t, "lstsq: not implemented for ",kname(x));
  J n=x->n; double r=nf; Tensor a,b; optstr m=c10::nullopt; TensorVector *v=nullptr;
  TORCH_CHECK(1<n && n<6, "lstsq: expects 2-5 args, (a;b;rcond;method;output), given ",n);
  if(n>2 && (v=xvec(x,n-1))) n--;
  if(n>2 && xsym(x,n-1)) n--, m=method(kK(x)[n]);
  if(n>2 && xdouble(x,n-1,r)) n--; 
  TORCH_CHECK(n==2, "lstsq: unexpected args, expects 2-5 args, (a;b;rcond;method;output)");
  bool p=xtenarg(x,a,b); c10::optional<double> c=c10::nullopt; if(r==r) c=r;
  return v ? koutput(*v, torch::linalg::lstsq(a,b,c,m)) : kresult(p, torch::linalg::lstsq(a,b,c,m));
 KCATCH("lstsq");
}

// --------------------------------------------------------------------
// chol - cholesky decomposition
// cholx - cholesky decomposition with additional flags and debug info
// --------------------------------------------------------------------
KAPI Chol(K x) {
 KTRY
  bool p=false,u=false; Tensor a,r;
  TORCH_CHECK(!x->t, "solve: not implemented for ",kname(x));
  if(xten(x,a)) {
   p=true;
  } else if(xarray(x,2)) {
   a=kput(x);
  } else {
   TORCH_CHECK((x->n==2 && (xbool(x,1,u) || xten(x,1,r))) ||
               (x->n==3 &&  xbool(x,1,u) && xten(x,2,r)),
               "chol: unrecognized arg(s), expecting 1-3 args, a, (a;upper), (a;out) or (a;upper;out)");
   p=xten(x,0,a);
   if(!p) a=kput(x,0);
  }
  return r.defined() ? (torch::linalg_cholesky_out(r,a,u), (K)0) : kresult(p, torch::linalg_cholesky(a,u));
 KCATCH("chol");
}

KAPI Cholx(K x) {
 KTRY
  TORCH_CHECK(!x->t, "cholx: not implemented for ",kname(x));
  bool p=false,u=false,e=false; J n=x->n; Tensor a; TensorVector *v=nullptr;
  if(xten(x,a)) {
   p=true;
  } else if(xarray(x,4)) {
   a=kput(x);
  } else {
   TORCH_CHECK(0<n && n<5, "cholx: expecting 1-4 arg(s), (x;upper;check;output), ",n," given");
   if(n>1 && (v=xvec(x,n-1))) n--;
   if(n>2 && xbool(x,n-2,u) && xbool(x,n-1,e)) n-=2;
   if(n>1 && xbool(x,n-1,u)) n--;
   TORCH_CHECK(n==1, "cholx: unexpected arg(s), expecting 1-4 arg(s), (x;upper;check;output)");
   if(!(p=xten(x,0,a))) a=kput(x,0);
  }
  return v ? koutput(*v, torch::linalg_cholesky_ex(a,u,e))
           : kresult(p,  torch::linalg_cholesky_ex(a,u,e));
 KCATCH("cholx");
}

// ----------------------------------------------------------------------------------
// cholinverse - inverse of symmetric positive-definite matrix using cholesky factors
// ----------------------------------------------------------------------------------
KAPI Cholinverse(K x) {
 KTRY
  TORCH_CHECK(!x->t, "cholinverse: not implemented for ",kname(x));
  bool p=false,u=false; J n=x->n; Tensor r,a;
  if(xten(x,a)) {
   p=true;
  } else if(xarray(x,3)) {
   a=kput(x);
  } else {
   TORCH_CHECK(0<n && n<4, "cholinverse: 1-3 args expected, (x;upper;output), ",n," given");
   if(n>1 &&  xten(x,n-1,r)) n--;
   if(n>1 && xbool(x,n-1,u)) n--;
   TORCH_CHECK(n==1, "cholinverse: unexpected arg(s), 1-3 args expected, (x;upper;output)");
   if(!(p=xten(x,0))) a=kput(x,0);
  }
  return r.defined() ? (torch::cholesky_inverse_out(r,a,u), (K)0)
                     : kresult(p, torch::cholesky_inverse(a,u));
 KCATCH("cholinverse");
}

// --------------------------------------------------------------------------------
// cholsolve - solves equations w'positive semidefinite matrix and cholesky factors
// --------------------------------------------------------------------------------
KAPI Cholsolve(K x) {
 KTRY
  TORCH_CHECK(!x->t, "choleskysolve: not implemented for ",kname(x));
  bool p,u=false; J n=xlen(x); Tensor a,b,r;
  TORCH_CHECK(1<n && n<5, "choleskysolve: 2-4 args expected, (b;u;upper;output), but ",n," given");
  if(n>2 &&  xten(x,n-1,r)) n--;
  if(n>2 && xbool(x,n-1,u)) n--;
  TORCH_CHECK(n==2,"choleskysolve: unrecognized arg(s), 2-4 args expected, (b;u;upper;output");
  p=xtenarg(x,a,b);
  return r.defined() ? (torch::cholesky_solve_out(r,a,b,u), (K)0)
                     : kresult(p, torch::cholesky_solve(a,b,u));
 KCATCH("choleskysolve");
}

// ----------------------------------------------------------
// eigenvalues and eigenvectors of matrix & symmetric matrix
// keig - parse args (x;upper;output)
// eig/eigh - eigen values and vctors, real & symmetric
// eigvals/eigvalsh eigen values only, real & symmetric
// ----------------------------------------------------------
static K keig(K x,bool s,bool t,const char *c) {  // s-true for symmetric, t-true for tuple result
 KTRY
  TORCH_CHECK(!x->t, c,": not implemented for ",kname(x));
  bool p=false,u=false; J m=s ? 3 : 2,n=x->n; Tensor a,r; TensorVector *v=nullptr;
  if(xten(x,a)) {
   p=true;
  } else if(xarray(x,m)) {
   a=kput(x);
  } else {
   TORCH_CHECK(0<n && n<=m, c,": expecting 1-",m," args, (x;", (m==3 ? "upper;" : ""),";output), ",n," given");
   if(n>1 && !t && xten(x,n-1,r)) n--;
   if(n>1 &&  t && (v=xvec(x,n-1))) n--;
   if(n>1 &&  s && xbool(x,n-1,u)) n--;
   TORCH_CHECK(n==1, c,": unrecognized arg(s), expecting 1-",m," args, (x;", (m==3 ? "upper;" : ""),";output), ",n," given");
   if(!(p=xten(x,0,a))) a=kput(x,0);
  }
  if(r.defined()) {
   return (s ? torch::linalg_eigvalsh_out(r, a, u ? "U" : "L") : torch::linalg_eigvals_out(r, a)), (K)0;
  } else if(v) {
   return koutput(*v, s ? torch::linalg_eigh(a, u ? "U" : "L") : torch::linalg_eig(a));
  } else if(t) {
   return kresult(p, s ? torch::linalg_eigh(a, u ? "U" : "L") : torch::linalg_eig(a));
  } else {
   return kresult(p, s ? torch::linalg_eigvalsh(a, u ? "U" : "L") : torch::linalg_eigvals(a));
  }
 KCATCH(c);
}

KAPI Eig(K x)       {return keig(x, false, true,  "eig");}
KAPI Eigh(K x)      {return keig(x, true,  true,  "eigh");}
KAPI Eigvals(K x)   {return keig(x, false, false, "eigvals");}
KAPI Eigvalsh(K x)  {return keig(x, true,  false, "eigvalsh");}

// ------------------------------------------------------------------
// prob - call probability function based on enumeration and 0-2 args
// ------------------------------------------------------------------
enum class Prob:char {cauchy,exponential,geometric,lognormal,normal,random,uniform};

static void prob(Prob p,Tensor& t) {
 switch(p) { 
  case Prob::cauchy:      t.cauchy_(); break;
  case Prob::exponential: t.exponential_(); break;
  case Prob::geometric:   TORCH_ERROR("geometric: requires a probability argument"); break;
  case Prob::lognormal:   t.log_normal_(); break;
  case Prob::normal:      t.normal_(); break;
  case Prob::random:      t.random_(); break;
  case Prob::uniform:     t.uniform_(); break;
 } 
}

static void prob(Prob p,Tensor& t,Scalar a) {
 switch(p) { 
  case Prob::cauchy:      t.cauchy_(a.toDouble()); break;
  case Prob::exponential: t.exponential_(a.toDouble()); break;
  case Prob::geometric:   t.geometric_(a.toDouble()); break;
  case Prob::lognormal:   t.log_normal_(a.toDouble()); break;
  case Prob::normal:      t.normal_(a.toDouble()); break;
  case Prob::random:      t.random_(a.toLong()); break;
  case Prob::uniform:     t.uniform_(a.toDouble()); break;
 } 
}

static void prob(Prob p,Tensor& t,Scalar a,Scalar b) {
 switch(p) { 
  case Prob::cauchy:      t.cauchy_(a.toDouble(),b.toDouble()); break;
  case Prob::exponential:
  case Prob::geometric:   TORCH_ERROR("geometric: expects single probality arg, given 2 args");
  case Prob::lognormal:   t.log_normal_(a.toDouble(),b.toDouble()); break;
  case Prob::normal:      t.normal_(a.toDouble(),b.toDouble()); break;
  case Prob::random:      t.random_(a.toLong(),b.toLong()); break;
  case Prob::uniform:     t.uniform_(a.toDouble(),b.toDouble()); break;
 } 
}

// ----------------------------------------------------------------------
// probarg - define & check 0-2 scalar args, return arg count
// probcall - call probability function on tensor with 0-2 args
// probpick - tensors picked via index/key, prob fn called for each pick
// ----------------------------------------------------------------------
static J probarg(K x,J i,Prob p,const char *s,Scalar& a,Scalar& b) {
 J n=x->n-i;
 if(n) {
  TORCH_CHECK(n==0 || xnum(x,i,  a), s,": expecting long/double for ",(i==1 ? "2nd" : "3rd")," argument, given ",kname(x,i));
  TORCH_CHECK(n==1 || xnum(x,i+1,b), s,": expecting long/double for ",(i==1 ? "3rd" : "4th")," argument, given ",kname(x,i+1));
  if(p==Prob::random) {
   TORCH_CHECK(n<2 || (a.isIntegral(false) && b.isIntegral(false)), s,": requires integers for low & high limits");
   TORCH_CHECK(n<1 ||  a.isIntegral(false), s,": requires integer for high limit");
  }          
 } 
 TORCH_CHECK(p != Prob::geometric || n==1, s,": requires a single probability argument, ",n," args supplied");
 return n;
}

static void probcall(Prob p,const char *s,Tensor& t,J n,const Scalar& a,const Scalar& b) {
 switch(n) {
  case 0: prob(p,t); break;
  case 1: prob(p,t,a); break;
  case 2: prob(p,t,a,b); break;
  default: TORCH_ERROR(s,": expecting 1-2 args, ",n," given"); break;
 }
}

static bool probpick(K x) {
 if(!x->t && x->n>1) { 
  auto t=kK(x)[1]->t;
  return t==-KS || t==KS || t==KJ || x->n>3;
 } else {
  return false;
 }
}

static K probpick(Ktag *g,K x,Prob p,const char *s) {
 K y=kK(x)[1]; Scalar a,b; auto n=probarg(x,2,p,s,a,b);
 for(auto& t:tensorpick(g,y,false,Cast::tensor,s))
  probcall(p,s,t,n,a,b);
 return (K)0;
} 

// ---------------------------------------------------------------------------
// kprob - parse k args, call relevant prob fn with given args or defaults
// ---------------------------------------------------------------------------
static K kprob(Ktag *g,K x,Prob p,const char *s) {
 Scalar a,b; auto n=probarg(x,1,p,s,a,b);
 switch(g->a) {
  case Class::tensor: probcall(p,s,g->tensor(),n,a,b); break;
  case Class::vector: for(auto& t:g->vector()) probcall(p,s,t,n,a,b); break;
  case Class::dict:   for(auto& t:g->dict().values()) probcall(p,s,t,n,a,b); break;
  default: TORCH_ERROR(s,": not implemented for ",mapclass(g->a),", without parameter/buffer names"); break;
 }
 return (K)0;
}

static K kprob(K x,bool b,Prob p,const char *s) {
 Tensor t;
 if(b) {   // k array without distinguishable scalar arg(s)
  t=kput(x); prob(p,t);
 } else { // k array & 1-2 scalar args
  Scalar a,b; t=kput(x,0);  auto n=probarg(x,1,p,s,a,b); probcall(p,s,t,n,a,b);
 }
 return kget(t);
}
 
static K kprob(K x,Prob p,const char *s) {
 KTRY
  torch::NoGradGuard nograd;
  if(auto *g=xtag(x)) {
   return kprob(g,x,p,s);
  } else if(auto *g=xtag(x,0)) {
   return probpick(x) ? probpick(g,x,p,s) : kprob(g,x,p,s);
  } else {
   return kprob(x,xarray(x,3),p,s);
  }
 KCATCH("probability");
}

KAPI Cauchy(K x)      {return kprob(x, Prob::cauchy,      "cauchy");}
KAPI Exponential(K x) {return kprob(x, Prob::exponential, "exponential");}
KAPI Geometric(K x)   {return kprob(x, Prob::geometric,   "geometric");}
KAPI Lognormal(K x)   {return kprob(x, Prob::lognormal,   "lognormal");}
KAPI Normal(K x)      {return kprob(x, Prob::normal,      "normal");}
KAPI Random(K x)      {return kprob(x, Prob::random,      "random");}
KAPI Uniform(K x)     {return kprob(x, Prob::uniform,     "uniform");}

// -------------------------------------------------------------------------------------
// bernoulli - function or in-place method, taking args of input a, prob p, output o
//             args: a, (a;p), (a;o), (a;p;[])  a can be k scalar/array or tensor
// -------------------------------------------------------------------------------------
KAPI Bernoulli(K x) {
 KTRY
  bool p=false; Tensor a,r;
  if(xten(x,a)) {
   p=true;
  } else if(!x->t && xten(x,1,r)) {
   TORCH_CHECK(x->n==2, "bernoulli: expecting args of (x;output), but ",x->n," args given");
   if(!(p=xten(x,0,a))) a=kput(x,0);
  } else {
   a=kput(x);
  }
  // v1.12.0 introduced new signatures, need null generator to bernoulli_out to resolve ambiguity
  return r.defined() ? (torch::bernoulli_out(r,a,c10::nullopt), (K)0) : kresult(p,torch::bernoulli(a));
 KCATCH("bernoulli");
}

// --------------------------------------------------------------------------------------
// multinomial - given vector or matrix of non-negative probabilities, pick n
//               args: (array/tensor; number of samples; replacement flag; output tensor)
// --------------------------------------------------------------------------------------
KAPI Multinomial(K x) {
 KTRY
  TORCH_CHECK(x->t>=0, "multinomial: not implemented for ",kname(x));
  bool b=false,p=false; int64_t m=-1; Tensor a,r;
  if(xten(x,a)) {
   p=true;
  } else if(xarray(x,4)) {
   a=kput(x);
  } else {
   J n=x->n;
   TORCH_CHECK(n>0 && n<5, "multinomial: expecting 1-4 args, (input;n;replaceflag;output), given ",x->n);
   if(n>1 && (xten(x,n-1,r))) n--;
   if(n>1 && xbool(x,n-1,b)) n--;
   if(n>1 && xint64(x,n-1,m)) n--;
   TORCH_CHECK(n==1, "multinomial: unrecognized args, expecting (input;n;replaceflag;output)");
   if(!(p=xten(x,0,a))) a=kput(x,0);
  }
  if(r.defined()) {
   torch::multinomial_out(r,a,m<0 ? 1 : m,b);
   if(m<0) r.squeeze_();
   return (K)0;
  } else {
   r=torch::multinomial(a,m<0 ? 1 : m,b);
   return kresult(p, m<0 ? r.squeeze() : r);
  }
 KCATCH("multinomial");
}

// -------------------------------------------------------------------------------------
// map api function to name in q session, upper case for 1st letter if reserved in k
// -------------------------------------------------------------------------------------
void mathfn(K x) {
 fn(x, "Abs",                KFN(Abs),             1);
 fn(x, "Acos",               KFN(Acos),            1);
 fn(x, "add",                KFN(Add),             1);
 fn(x, "addbmm",             KFN(Addbmm),          1);
 fn(x, "addmm",              KFN(Addmm),           1);
 fn(x, "addmv",              KFN(Addmv),           1);
 fn(x, "addr",               KFN(Addr),            1);
 fn(x, "addcdiv",            KFN(Addcdiv),         1);
 fn(x, "addcmul",            KFN(Addcmul),         1);
 fn(x, "All",                KFN(All),             1);
 fn(x, "allclose",           KFN(Allclose),        1);
 fn(x, "amax",               KFN(Amax),            1);
 fn(x, "amin",               KFN(Amin),            1);
 fn(x, "aminmax",            KFN(Aminmax),         1);
 fn(x, "angle",              KFN(Angle),           1);
 fn(x, "Any",                KFN(Any),             1);
 fn(x, "argmax",             KFN(Argmax),          1);
 fn(x, "argmin",             KFN(Argmin),          1);
 fn(x, "argsort",            KFN(Argsort),         1);
 fn(x, "Asin",               KFN(Asin),            1);
 fn(x, "Atan",               KFN(Atan),            1);
 fn(x, "atan2",              KFN(Atan2),           1);
 fn(x, "baddbmm",            KFN(Baddbmm),         1);
 fn(x, "bartlett",           KFN(Bartlett),        1);
 fn(x, "bernoulli",          KFN(Bernoulli),       1);
 fn(x, "blackman",           KFN(Blackman),        1);
 fn(x, "bincount",           KFN(Bincount),        1);
 fn(x, "bitwisenot",         KFN(Bitwisenot),      1);
 fn(x, "bmm",                KFN(Bmm),             1);
 fn(x, "cauchy",             KFN(Cauchy),          1);
 fn(x, "ceil",               KFN(Ceil),            1);
 fn(x, "chol",               KFN(Chol),            1);
 fn(x, "cholx",              KFN(Cholx),           1);
 fn(x, "cholinverse",        KFN(Cholinverse),     1);
 fn(x, "cholsolve",          KFN(Cholsolve),       1);
 fn(x, "clamp",              KFN(Clamp),           1);
 fn(x, "close",              KFN(Close),           1);
 fn(x, "Cos",                KFN(Cos),             1);
 fn(x, "cosh",               KFN(Cosh),            1);
 fn(x, "Cross",              KFN(Cross),           1);
 fn(x, "cumprod",            KFN(Cumprod),         1);
 fn(x, "cumsum",             KFN(Cumsum),          1);
 fn(x, "det",                KFN(Det),             1);
 fn(x, "diag",               KFN(Diag),            1);
 fn(x, "diagflat",           KFN(Diagflat),        1);
 fn(x, "diagonal",           KFN(Diagonal),        1);
 fn(x, "digamma",            KFN(Digamma),         1);
 fn(x, "dist",               KFN(Dist),            1);
 fn(x, "Div",                KFN(Div),             1);
 fn(x, "dot",                KFN(Dot),             1);
 fn(x, "eig",                KFN(Eig),             1);
 fn(x, "eigh",               KFN(Eigh),            1);
 fn(x, "eigvals",            KFN(Eigvals),         1);
 fn(x, "eigvalsh",           KFN(Eigvalsh),        1);
 fn(x, "eq",                 KFN(Eq),              1);
 fn(x, "equal",              KFN(Equal),           1);
 fn(x, "erf",                KFN(Erf),             1);
 fn(x, "erfc",               KFN(Erfc),            1);
 fn(x, "erfinv",             KFN(Erfinv),          1);
 fn(x, "Exp",                KFN(Exp),             1);
 fn(x, "expm1",              KFN(Expm1),           1);
 fn(x, "exponential",        KFN(Exponential),     1);
 fn(x, "fft",                KFN(fft),             1);
 fn(x, "fft2",               KFN(fft2),            1);
 fn(x, "fftfreq",            KFN(fftfreq),         1);
 fn(x, "fftn",               KFN(fftn),            1);
 fn(x, "fftshift",           KFN(fftshift),        1);
 fn(x, "finite",             KFN(Finite),          1);
 fn(x, "Flip",               KFN(Flip),            1);
 fn(x, "Floor",              KFN(Floor),           1);
 fn(x, "fmax",               KFN(Fmax),            1);
 fn(x, "fmin",               KFN(Fmin),            1);
 fn(x, "fmod",               KFN(Fmod),            1);
 fn(x, "fnorm",              KFN(Fnorm),           1);
 fn(x, "fpow",               KFN(Fpow),            1);
 fn(x, "frac",               KFN(Frac),            1);
 fn(x, "ge",                 KFN(Ge),              1);
 fn(x, "geometric",          KFN(Geometric),       1);
 fn(x, "gt",                 KFN(GT),              1);
 fn(x, "hann",               KFN(Hann),            1);
 fn(x, "hamming",            KFN(Hamming),         1);
 fn(x, "hfft",               KFN(hfft),            1);
 fn(x, "hfft2",              KFN(hfft2),           1);
 fn(x, "hfftn",              KFN(hfftn),           1);
 fn(x, "histc",              KFN(Histc),           1);
 fn(x, "inverse",            KFN(Inverse),         1);
 fn(x, "ifft",               KFN(ifft),            1);
 fn(x, "ifft2",              KFN(ifft2),           1);
 fn(x, "ifftn",              KFN(ifftn),           1);
 fn(x, "ifftshift",          KFN(ifftshift),       1);
 fn(x, "ihfft",              KFN(ihfft),           1);
 fn(x, "ihfft2",             KFN(ihfft2),          1);
 fn(x, "ihfftn",             KFN(ihfftn),          1);
 fn(x, "In",                 KFN(In),              1);
 fn(x, "inf",                KFN(Inf),             1);
 fn(x, "irfft",              KFN(irfft),           1);
 fn(x, "irfft2",             KFN(irfft2),          1);
 fn(x, "irfftn",             KFN(irfftn),          1);
 fn(x, "kaiser",             KFN(Kaiser),          1);
 fn(x, "kthvalue",           KFN(Kthvalue),        1);
 fn(x, "le",                 KFN(Le),              1);
 fn(x, "lerp",               KFN(Lerp),            1);
 fn(x, "lgamma",             KFN(Lgamma),          1);
 fn(x, "Log",                KFN(Log),             1);
 fn(x, "log10",              KFN(Log10),           1);
 fn(x, "log1p",              KFN(Log1p),           1);
 fn(x, "log2",               KFN(Log2),            1);
 fn(x, "logdet",             KFN(Logdet),          1);
 fn(x, "lognormal",          KFN(Lognormal),       1);
 fn(x, "logsumexp",          KFN(Logsumexp),       1);
 fn(x, "lstsq",              KFN(Lstsq),           1);
 fn(x, "lt",                 KFN(Lt),              1);
 fn(x, "lu",                 KFN(Lu),              1);
 fn(x, "lun",                KFN(Lun),             1);
 fn(x, "lux",                KFN(Lux),             1);
 fn(x, "lusolve",            KFN(Lusolve),         1);
 fn(x, "matmul",             KFN(Matmul),          1);
 fn(x, "multinomial",        KFN(Multinomial),     1);
 fn(x, "outer",              KFN(Outer),           1);
 fn(x, "power",              KFN(Matrix_power),    1);
 fn(x, "mrank",              KFN(Mrank),           1);
 fn(x, "Max",                KFN(Max),             1);
 fn(x, "maximum",            KFN(Maximum),         1);
 fn(x, "mean",               KFN(Mean),            1);
 fn(x, "meanstd",            KFN(Meanstd),         1);
 fn(x, "meanvar",            KFN(Meanvar),         1);
 fn(x, "median",             KFN(Median),          1);
 fn(x, "Min",                KFN(Min),             1);
 fn(x, "minimum",            KFN(Minimum),         1);
 fn(x, "mm",                 KFN(Mm),              1);
 fn(x, "mmt",                KFN(Mmt),             1);
 fn(x, "mnorm",              KFN(Mnorm),           1);
 fn(x, "mode",               KFN(Mode),            1);
 fn(x, "msort",              KFN(Msort),           1);
 fn(x, "mtm",                KFN(Mtm),             1);
 fn(x, "mul",                KFN(Mul),             1);
 fn(x, "multidot",           KFN(Multidot),        1);
 fn(x, "mv",                 KFN(Mv),              1);
 fn(x, "mvlgamma",           KFN(Mvlgamma),        1);
 fn(x, "nan",                KFN(Nan),             1);
 fn(x, "nanmean",            KFN(Nanmean),         1);
 fn(x, "nanmedian",          KFN(Nanmedian),       1);
 fn(x, "nansum",             KFN(Nansum),          1);
 fn(x, "ne",                 KFN(Ne),              1);
 fn(x, "Neg",                KFN(Neg),             1);
 fn(x, "neginf",             KFN(Neginf),          1);
 fn(x, "Not",                KFN(Not),             1);
 fn(x, "nnorm",              KFN(Nnorm),           1);
 fn(x, "normal",             KFN(Normal),          1);
 fn(x, "householder",        KFN(Householder),     1);
 fn(x, "pinverse",           KFN(Pinverse),        1);
 fn(x, "pow",                KFN(Pow),             1);
 fn(x, "posinf",             KFN(Posinf),          1);
 fn(x, "prod",               KFN(Prod),            1);
 fn(x, "qr",                 KFN(Qr),              1);
 fn(x, "random",             KFN(Random),          1);
 fn(x, "Reciprocal",         KFN(Reciprocal),      1);
 fn(x, "remainder",          KFN(Remainder),       1);
 fn(x, "roll",               KFN(Roll),            1);
 fn(x, "renorm",             KFN(Renorm),          1);
 fn(x, "rfft",               KFN(rfft),            1);
 fn(x, "rfft2",              KFN(rfft2),           1);
 fn(x, "rfftfreq",           KFN(rfftfreq),        1);
 fn(x, "rfftn",              KFN(rfftn),           1);
 fn(x, "round",              KFN(Round),           1);
 fn(x, "rsqrt",              KFN(Rsqrt),           1);
 fn(x, "sigmoid",            KFN(Ksigmoid),        1);
 fn(x, "sgn",                KFN(Sgn),             1);
 fn(x, "sign",               KFN(Sign),            1);
 fn(x, "Sin",                KFN(Sin),             1);
 fn(x, "sinh",               KFN(Sinh),            1);
 fn(x, "slogdet",            KFN(Slogdet),         1);
 fn(x, "solve",              KFN(Solve),           1);
 fn(x, "sort",               KFN(Sort),            1);
 fn(x, "Sqrt",               KFN(Sqrt),            1);
 fn(x, "std",                KFN(Std),             1);
 fn(x, "Sum",                KFN(Sum),             1);
 fn(x, "svd",                KFN(Svd),             1);
 fn(x, "svdvals",            KFN(Svdvals),         1);
 fn(x, "Tan",                KFN(Tan),             1);
 fn(x, "tanh",               KFN(Ktanh),           1);
 fn(x, "tensordot",          KFN(Tensordot),       1);
 fn(x, "topk",               KFN(Topk),            1);
 fn(x, "trace",              KFN(Trace),           1);
 fn(x, "tril",               KFN(Tril),            1);
 fn(x, "triu",               KFN(Triu),            1);
 fn(x, "trisolve",           KFN(Trisolve),        1);
 fn(x, "trunc",              KFN(Trunc),           1);
 fn(x, "uniform",            KFN(Uniform),         1);
 fn(x, "unique",             KFN(Unique),          1);
 fn(x, "uniquec",            KFN(Uniquec),         1);
 fn(x, "variance",           KFN(Var),             1);
 fn(x, "vnorm",              KFN(Vnorm),           1);
 fn(x, "xor",                KFN(Xor),             1);
}
