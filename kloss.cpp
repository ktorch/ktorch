#include "ktorch.h"
#include "kloss.h"

// append a loss option to a k dictionary given dict,name & value
#define OPTION(x,k,v) dictadd(x, lset(Setting::k), v)

// ------------------------------------------------------------------------------------------------------
// kloss - given loss type & shared pointer to newly created loss module, return kptr
// lmap - map to/from sym to loss function name, e.g. `mse <-> Cast::mse
// lset - map to/from sym to loss setting enum, e.g. `reduce <-> Setting::reduce
// ------------------------------------------------------------------------------------------------------
K kloss(Cast c,const AnyModule& m) {return kptr(new Kmodule(Class::loss,c,m));}

static Cast lmap(S s) {
 for(auto&m:env().loss)
  if(std::get<0>(m)==s) return std::get<1>(m);
 AT_ERROR("Unrecognized loss function: ",s);
}

static S lmap(Cast c) {
 for(auto&m:env().loss)
  if(std::get<1>(m)==c) return std::get<0>(m);
 AT_ERROR("Unrecognized loss function: ",(I)c);
}

static S lset(Setting s) {
 for(auto&m:env().lset)
  if(std::get<1>(m)==s) return std::get<0>(m);
 AT_ERROR("Unrecognized loss setting: ",(I)s);
}

static Setting lset(S s) {
 for(auto&m:env().lset)
  if(std::get<0>(m)==s) return std::get<1>(m);
 AT_ERROR("Unrecognized loss setting: ",s);
}

// ----------------------------------------------------------------------------------------------------
// input checking fns with error msg specific to loss module name and setting
// check positional or name-value pairs for lbool->boolean, lsym->sym, int64-integer, ldouble..
// ----------------------------------------------------------------------------------------------------
static bool lbool(K x,J i,Cast c,Setting s) {
 bool b;
 TORCH_CHECK(xbool(x,i,b), lmap(c)," ",lset(s),": expected boolean scalar, given ",kname(x,i));
 return b;
}

static bool lbool(const Pairs& p,Cast c) {
 TORCH_CHECK(p.t==-KB, lmap(c)," ",p.k,": expected boolean scalar, given ",kname(p.t));
 return p.b;
}

static S lsym(K x,J i,Cast c,Setting s) {
 S sy;
 TORCH_CHECK(xsym(x,i,sy), lmap(c)," ",lset(s),": expected symbol, given ",kname(x,i));
 return sy;
}

static S lsym(const Pairs& p,Cast c) {
 TORCH_CHECK(p.t==-KS, lmap(c)," ",p.k,": expected symbol, given ",kname(p.t));
 return p.s;
}

static int64_t int64(K x,J i,Cast c,Setting s) {
 int64_t n;
 TORCH_CHECK(xint64(x,i,n), lmap(c)," ",lset(s),": expected long scalar, given ",kname(x,i));
 return n;
}

static int64_t int64(const Pairs& p,Cast c) {
 TORCH_CHECK(p.t==-KJ, lmap(c)," ",p.k,": expected long scalar, given ",kname(p.t));
 return p.j;
}

static double ldouble(K x,J i,Cast c,Setting s) {
 double f;
 TORCH_CHECK(xnum(x,i,f), lmap(c)," ",lset(s),": expected double, given ",kname(x,i));
 return f;
}

static double ldouble(const Pairs& p,Cast c) {
 TORCH_CHECK(p.t==-KJ || p.t==-KF, lmap(c)," ",p.k,": expected double, given ",kname(p.t));
 return pdouble(p);
}

// -----------------------------------------------------------------------------------------------
//  reduction arg uses variant, using functions below to translate sym -> variant value
//  (can simplify once KL loss removes "batchmean" reduction)
// -----------------------------------------------------------------------------------------------
using Reduce1=c10::variant<torch::enumtype::kNone, torch::enumtype::kMean, torch::enumtype::kSum>;
using Reduce2=c10::variant<torch::enumtype::kNone, torch::enumtype::kBatchMean, torch::enumtype::kSum, torch::enumtype::kMean>;

static void reduce(Reduce1& r,Cast c,S s) {
 switch(emap(s)) {
  case Enum::none: r=torch::kNone; break;
  case Enum::mean: r=torch::kMean; break;
  case Enum::sum:  r=torch::kSum; break;
  default: AT_ERROR(lmap(c)," reduce:",s," is not one of none,mean,sum");
 }
}

static void reduce(Reduce2& r,Cast c,S s) {
 switch(emap(s)) {
  case Enum::none:      r=torch::kNone; break;
  case Enum::batchmean: r=torch::kBatchMean; break;
  case Enum::mean:      r=torch::kMean; break;
  case Enum::sum:       r=torch::kSum; break;
  default: AT_ERROR(lmap(c)," reduce:",s," is not one of none,batchmean,mean,sum");
 }
}

// ----------------------------------------------------------------
// reduce - get/set reduction mode for various loss module options
// ----------------------------------------------------------------
template<typename O> static O reduce(K x,J i,Cast c) {
 O o; Pairs p; J n=xargc(x,i,p); S s=nullptr;
 TORCH_CHECK(n<2, lmap(c),": only 1 positional argument(reduce) expected, ",n," given");
 if(n==1) s=lsym(x,i,c,Setting::reduce);
 while(xpair(p)) {
  TORCH_CHECK(lset(p.k)==Setting::reduce, "Unrecognized option: ",p.k,", ",lmap(c)," loss expects single option: reduce");
  s=lsym(p,c);
 }
 if(s) reduce(o.reduction(),c,s);
 return o;
}

template<typename O> static void reduce(bool a,K x,const O& o,const O d=O());
template<typename O> static void reduce(bool a,K x,const O& o,const O d) {
 if(a || d.reduction().index() != o.reduction().index())
  OPTION(x, reduce, ks(ESYM(o.reduction())));
}

// ------------------------------------------------------------------------------------------------------
// lossfunc - call loss function with x,y tensors/arrays and optional reduction mode
// bce - binary cross entropy has option of batch weights, so function parses (x;y) or (x;y;wt)
// ------------------------------------------------------------------------------------------------------
static K lossfunc(K a,Cast c) {
 KTRY
  namespace nn=torch::nn; namespace f=nn::functional; bool b,p; Tensor r,x,y;
  TORCH_CHECK(a->t>=0, lmap(c),": not implemented for ",kname(a));
  b=a->n==2;
  if(a->t) {
   TORCH_CHECK(b, lmap(c),": loss expects 2-element arg of input & target, ",a->n," value(s) given");
   x=kput(a); y=x[1]; x=x[0]; p=false;
  } else {
   p=xtenarg(a,x,y);
  }
  switch(c) {
   case Cast::kl: r=b ? f::kl_div(x,y) : f::kl_div(x,y,reduce<nn::KLDivLossOptions>(a,2,c)); break;
   case Cast::l1: r=b ? f::l1_loss(x,y) : f::l1_loss(x,y,reduce<nn::L1LossOptions>(a,2,c)); break;
   case Cast::mse: r=b ? f::mse_loss(x,y) : f::mse_loss(x,y,reduce<nn::MSELossOptions>(a,2,c)); break;
   case Cast::multilabel:
    r=b ? f::multilabel_margin_loss(x,y)
        : f::multilabel_margin_loss(x,y,reduce<nn::MultiLabelMarginLossOptions>(a,2,c));
    break;
   case Cast::smoothl1:
    r=b ? f::smooth_l1_loss(x,y) : f::smooth_l1_loss(x,y,reduce<nn::SmoothL1LossOptions>(a,2,c));
    break;
   case Cast::softmargin:
    r=b ? f::soft_margin_loss(x,y) : f::soft_margin_loss(x,y,reduce<nn::SoftMarginLossOptions>(a,2,c));
    break;
   default: AT_ERROR("Unrecognized loss function"); break;
  }
 return kresult(p,r);
 KCATCH("loss");
}

KAPI kl(K x)          {return lossfunc(x, Cast::kl);}
KAPI l1(K x)          {return lossfunc(x, Cast::l1);}
KAPI mse(K x)         {return lossfunc(x, Cast::mse);}
KAPI multilabel(K x)  {return lossfunc(x, Cast::multilabel);}
KAPI smoothl1(K x)    {return lossfunc(x, Cast::smoothl1);}
KAPI softmargin(K x)  {return lossfunc(x, Cast::softmargin);}

// ----------------------------------------------------------------------------
// binary cross entropy: optional 3rd input of batch weights
// bcearg - evaluate arg to see if weight input or reduction option
// ----------------------------------------------------------------------------
static bool bcearg(K x) {return x->t==-KS || x->t==KS || xempty(x) || xdict(x);}

KAPI bce(K a) {
 KTRY
  auto r=BCELossOptions().reduction(); bool p; Tensor x,y,w; Cast c=Cast::bce;
  TORCH_CHECK(0<=a->t && a->t<11, lmap(c),": not implemented for ",kname(a));
  TORCH_CHECK(a->n>1, lmap(c),": expects input,target,optional batch weight,optional reduce mode");
  if(a->t) {
   TORCH_CHECK(a->n<4,lmap(c),": unrecognized args, expecting 2-3 elements for input,target,optional weight");
   p=false; x=kput(a); if(a->n==3) w=x[2]; y=x[1]; x=x[0];
  } else {
   bool b=a->n==2 ? true : bcearg(kK(a)[2]);
   p = b ? xtenarg(a,x,y) : xtenarg(a,x,y,w);
   r=reduce<BCELossOptions>(a,3-b,c).reduction();
  }
  return kresult(p, torch::nn::functional::detail::binary_cross_entropy(x,y,w,r));
 KCATCH("bce");
}

// ------------------------------------------------------------------------------------------------------
// classwt - set optional class weights & reduction mode, also index to ignore for some losses
//           classes with optional index use the same elements, so a templated fn is used,
//           but others use "weight" vs "pos_weight", requiring class-specific overloads
// ------------------------------------------------------------------------------------------------------
static void classwt(K x,J i,Cast c,S& s,Tensor& w) {
 Pairs p; J n=xargc(x,i,p); s=nullptr;
 if(n && xsym(x,i+n-1,s)) n--;
 if(n) {n--; if(!xempty(x,i+n) && !xten(x,i+n,w)) w=kput(x,i+n);}
 TORCH_CHECK(!n, lmap(c),": unrecognized positional arg(s), expected weights, reduce mode or (weights;reduce mode)");
 while(xpair(p))
  switch(lset(p.k)) {
   case Setting::weight: if(!pempty(p)) pten(p,w); break;
   case Setting::reduce: s=lsym(p,c); break;
   default: AT_ERROR("Unrecognized ",lmap(c)," option: ",p.k); break;
  }
}

static auto& classwt(K x,J i,Cast c,BCEWithLogitsLossOptions&& o) {
 S s; Tensor w; classwt(x,i,c,s,w);
 if(s) reduce(o.reduction(),c,s);
 if(w.defined()) o.pos_weight(w);
 return o;
}

static auto& classwt(K x,J i,Cast c,torch::nn::MultiLabelSoftMarginLossOptions&& o) {
 S s; Tensor w; classwt(x,i,c,s,w);
 if(s) reduce(o.reduction(),c,s);
 if(w.defined()) o.weight(w);
 return o;
}

template<typename O> O classwt(K x,J i,Cast c) {
 O o; Pairs p; J n=xargc(x,i,p); S s=nullptr; Tensor w;
 if(n && xsym(x,i+n-1,s)) n--; // allow last arg of symbol regardless
 for(J j=0;j<n;++j)
  switch(j) {
   case 0: if(!xempty(x,i+j) && !xten(x,i+j,w)) w=kput(x,i+j); break;
   case 1: o.ignore_index(int64(x,i+j,c,Setting::ignore)); break;
   default: AT_ERROR(lmap(c),": up to 3 positional args expected(class weight;index to ignore;reduce mode), ",n," given");
  }
 while(xpair(p))
  switch(lset(p.k)) {
   case Setting::weight: if(!pempty(p)) pten(p,w); break;
   case Setting::ignore: o.ignore_index(int64(p,c)); break;
   case Setting::reduce: s=lsym(p,c); break;
   default: AT_ERROR("Unrecognized option: ",p.k," for ",lmap(c)," loss"); break;
  }
 if(s) reduce(o.reduction(),c,s);
 if(w.defined()) o.weight(w);
 return o;
}

// ------------------------------------------------------------------------------------------------------
// classwt - get optional class weights & reduction mode, also index to ignore for some losses
// ------------------------------------------------------------------------------------------------------
static void classwt(bool a,K x,const BCEWithLogitsLossOptions& o) {
 if(a || o.pos_weight().defined()) OPTION(x,weight,kget(o.pos_weight()));
 reduce(a,x,o);
}

static void classwt(bool a,K x,const torch::nn::MultiLabelSoftMarginLossOptions& o) {
 if(a || o.weight().defined()) OPTION(x,weight,kget(o.weight()));
 reduce(a,x,o);
}

template<typename O> static void classwt(bool a,K x,const O& o) {
 const O d;
 if(a || o.weight().defined()) OPTION(x, weight, kget(o.weight()));
 if(a || d.ignore_index() != o.ignore_index()) OPTION(x, ignore, kj(o.ignore_index()));
 reduce(a,x,o,d);
}

// ------------------------------------------------------------------------------------------------------
// classwt - functional form for cross entropy, nll, multi-label soft margin loss
// ------------------------------------------------------------------------------------------------------
static K classwt(K a,Cast c) {
 KTRY
  namespace nn=torch::nn; namespace f=nn::functional;
  bool p=false; Tensor r,x,y;
  TORCH_CHECK(a->t>=0, lmap(c),": not implemented for ",kname(a));
  TORCH_CHECK(a->n>1, lmap(c), " loss expects (input;target;optional arg(s)..)");
  if(a->t) r=kput(a), x=r[0], y=r[1];
  else     p=xtenarg(a,x,y);
  switch(c) {
   case Cast::ce:        r=f::cross_entropy(x,y,classwt<nn::CrossEntropyLossOptions>(a,2,c)); break;
   case Cast::nll:       r=f::nll_loss(x,y,classwt<nn::NLLLossOptions>(a,2,c)); break;
   case Cast::multisoft: r=f::multilabel_soft_margin_loss(x,y,classwt(a,2,c,nn::MultiLabelSoftMarginLossOptions())); break;
   default: AT_ERROR("Unrecognized loss function");
  }
  return kresult(p,r);
 KCATCH("loss");
}

KAPI ce(K x)        {return classwt(x, Cast::ce);}
KAPI nll(K x)       {return classwt(x, Cast::nll);}
KAPI multisoft(K x) {return classwt(x, Cast::multisoft);}

// ---------------------------------------------------------------------------------------
// bceloss - handle binary cross-entropy with logits, separate call if batch weights
// bcelogit1 - input & target, with optional class weights and reduction mode
// bcelogit2 - input, target & batch weights, along with options for class weight,reduce
// ---------------------------------------------------------------------------------------
static K bceloss(K a,bool b,const char* s) {  // a:args, b:true if batch wts, s:label
 KTRY
  bool p; J n=2+b; Tensor x,y,w;
  TORCH_CHECK(0<=a->t && a->t<11, s,": not implemented for ",kname(a));
  TORCH_CHECK(a->n>=n, s," expects at least ", n, " args, input", (b ? ", target & batch weights" : " & target"));
  auto o=classwt(a,n,Cast::bcelogits,BCEWithLogitsLossOptions());
  if(a->t) {
   p=false; x=kput(a); if(b) w=x[2]; y=x[1]; x=x[0];
  } else {
   p=b ? xtenarg(a,x,y,w) : xtenarg(a,x,y);
  }
  return kresult(p, torch::nn::functional::detail::binary_cross_entropy_with_logits(x, y, w, o.reduction(), o.pos_weight()));
 KCATCH(s);
}

KAPI bcelogit1(K x) {return bceloss(x, false, "bcelogit1");}
KAPI bcelogit2(K x) {return bceloss(x, true,  "bcelogit2");}

// ------------------------------------------------------------------------------------------------------
// margin - get/set optional margin & reduction arguments
// marginloss - functional form of loss functions w'margin & reduction args
// ------------------------------------------------------------------------------------------------------
template<typename O> static O margin(K x,J i,Cast c) {
 O o; Pairs p; J n=xargc(x,i,p); S s=nullptr;
 if(n && xsym(x,i+n-1,s)) n--;
 if(n) n--, o.margin(ldouble(x,i+n,c,Setting::margin));
 TORCH_CHECK(!n, lmap(c),": unrecognized positional arg(s), expected margin,reduce or (margin;reduce), e.g. (1.0;`mean)");
 while(xpair(p))
  switch(lset(p.k)) {
   case Setting::margin: o.margin(ldouble(p,c)); break;
   case Setting::reduce: s=lsym(p,c); break;
   default: AT_ERROR("Unrecognized ",lmap(c)," option: ",p.k); break;
  }
 if(s) reduce(o.reduction(),c,s);
 return o;
}

template<typename O> static void margin(bool a,K x,const O& o) {
 const O d;
 if(a || d.margin() != o.margin()) OPTION(x, margin, kf(o.margin()));
 reduce(a,x,o,d);
}

static K marginloss(K a,Cast c) {
 KTRY
  namespace nn=torch::nn; namespace f=nn::functional;
  bool b,p=false,h=c==Cast::hinge; Tensor r,x1,x2,y;
  TORCH_CHECK(a->t>=0, lmap(c),": not implemented for ",kname(a));
  TORCH_CHECK(a->n>=3-h, lmap(c), " loss expects (input", (h ? "" : "1;input2"),";target;optional arg(s)..)");
  b=a->n==3-h;
  if(a->t) {
   r=kput(a);
   if(h) x1=r[0], y=r[1];
   else  x1=r[0], x2=r[1], y=r[2];
  } else {
   p=h ? xtenarg(a,x1,y) : xtenarg(a,x1,x2,y);
  }
  switch(c) {
   case Cast::hinge: 
    r=b ? f::hinge_embedding_loss(x1,y) : f::hinge_embedding_loss(x1,y,margin<nn::HingeEmbeddingLossOptions>(a,2,c));
    break;
   case Cast::cosineloss:
    r=b ? f::cosine_embedding_loss(x1,x2,y) : f::cosine_embedding_loss(x1,x2,y,margin<nn::CosineEmbeddingLossOptions>(a,3,c));
    break;
   case Cast::margin:
    r=b ? f::margin_ranking_loss(x1,x2,y) : f::margin_ranking_loss(x1,x2,y,margin<nn::MarginRankingLossOptions>(a,3,c));
    break;
   default: AT_ERROR("Unrecognized loss function"); break;
  }
  return kresult(p,r);
 KCATCH("loss")
}

KAPI hinge(K x)      {return marginloss(x, Cast::hinge);}
KAPI cosineloss(K x) {return marginloss(x, Cast::cosineloss);}
KAPI Margin(K x)     {return marginloss(x, Cast::margin);}

// ----------------------------------------------------------------------------------------
// multi - get/set optional power,margin,weight & reduction arguments for multi margin loss
// multimargin - funcional form of multi margin loss function
// ----------------------------------------------------------------------------------------
static torch::nn::MultiMarginLossOptions multi(K x,J i,Cast c) {
 Pairs p; J n=xargc(x,i,p); S s=nullptr; Tensor w; torch::nn::MultiMarginLossOptions o;
 if(n && xsym(x,i+n-1,s)) n--;
 for(J j=0;j<n;++j)
  switch(j) {
   case 0: o.p(int64(x,i+j,c,Setting::p)); break;
   case 1: o.margin(ldouble(x,i+j,c,Setting::margin)); break;
   case 2: if(!xempty(x,i+j) && !xten(x,i+j,w)) w=kput(x,i+j); break;
   default: AT_ERROR(lmap(c),": unrecognized positional arg(s), expecting up to 4 args, p,margin,weight,reduce");
  }
 while(xpair(p))
  switch(lset(p.k)) {
   case Setting::p:      o.p(int64(p,c)); break;
   case Setting::margin: o.margin(ldouble(p,c)); break;
   case Setting::weight: if(!pempty(p)) pten(p,w); break;
   case Setting::reduce: s=lsym(p,c); break;
   default: AT_ERROR("Unrecognized option: ",p.k," for multi-margin loss"); break;
  }
 if(w.defined()) o.weight(w);
 if(s) reduce(o.reduction(),c,s);
 return o;
}

static void multi(bool a,K x,const torch::nn::MultiMarginLossOptions& o) {
 const torch::nn::MultiMarginLossOptions d;
 if(a || d.p()      != o.p())      OPTION(x, p,      kj(o.p()));
 if(a || d.margin() != o.margin()) OPTION(x, margin, kf(o.margin()));
 if(a || o.weight().defined())     OPTION(x, weight, kget(o.weight()));
 reduce(a,x,o,d);
}

KAPI multimargin(K a) {
 KTRY
  bool p; Tensor x,y; Cast c=Cast::multimargin;
  TORCH_CHECK(a->t>=0, lmap(c)," loss not implemented for ",kname(a));
  TORCH_CHECK(a->n>=2, lmap(c)," loss expects (input;target;optional arg(s)..)");
  if(a->t) p=false, x=kput(a), y=x[1], x=x[0]; 
  else     p=xtenarg(a,x,y);
  return kresult(p, a->n==2 ? torch::nn::functional::multi_margin_loss(x,y)
                            : torch::nn::functional::multi_margin_loss(x,y,multi(a,2,c)));
 KCATCH("multi-margin loss");
}

// ------------------------------------------------------------------------------------------------------
// triplet - get/set optional margin,p,eps,swap flag & reduction args in k array for triplet loss
// ------------------------------------------------------------------------------------------------------
static torch::nn::TripletMarginLossOptions triplet(K x,J i,Cast c) {
 Pairs p; J n=xargc(x,i,p); S s=nullptr; torch::nn::TripletMarginLossOptions o;
 if(n && xsym(x,i+n-1,s)) n--;
 for(J j=0;j<n;++j)
  switch(j) {
   case 0: o.margin(ldouble(x,i+j,c,Setting::margin)); break;
   case 1: o.p(ldouble(x,i+j,c,Setting::p)); break;
   case 2: o.eps(ldouble(x,i+j,c,Setting::eps)); break;
   case 3: o.swap(lbool(x,i+j,c,Setting::swap)); break;
   default: AT_ERROR(lmap(c),": unrecognized positional arg(s), expecting up to 5 args, margin,p,eps,swap flag,reduce");
  }
 while(xpair(p))
  switch(lset(p.k)) {
   case Setting::margin: o.margin(ldouble(p,c)); break;
   case Setting::p:      o.p(ldouble(p,c)); break;
   case Setting::eps:    o.eps(ldouble(p,c)); break;
   case Setting::swap:   o.swap(lbool(p,c)); break;
   case Setting::reduce: s=lsym(p,c); break;
   default: AT_ERROR("Unrecognized option: ",p.k," for multi-margin loss"); break;
  }
 if(s) reduce(o.reduction(),c,s);
 return o;
}

static void triplet(bool a,K x,const torch::nn::TripletMarginLossOptions& o) {
 const torch::nn::TripletMarginLossOptions d;
 if(a || d.margin() != o.margin()) OPTION(x, margin, kf(o.margin()));
 if(a || d.p()      != o.p())      OPTION(x, p,      kf(o.p()));
 if(a || d.eps()    != o.eps())    OPTION(x, eps,    kf(o.eps()));
 if(a || d.swap()   != o.swap())   OPTION(x, swap,   kb(o.swap()));
 reduce(a,x,o,d);
}

KAPI Triplet(K a) {
 KTRY
  bool p; Tensor x,y,z; Cast c=Cast::triplet;
  TORCH_CHECK(a->t>=0, lmap(c)," loss not implemented for ",kname(a));
  TORCH_CHECK(a->n>=2, lmap(c)," loss expects (anchor;positive;negative;optional arg(s)..)");
  if(a->t) p=false, x=kput(a), z=x[2], y=x[1], x=x[0];
  else     p=xtenarg(a,x,y,z);
  return kresult(p, a->n==3 ? torch::nn::functional::triplet_margin_loss(x,y,z)
                            : torch::nn::functional::triplet_margin_loss(x,y,z,triplet(a,3,c)));
 KCATCH("triplet margin loss");
}

// ------------------------------------------------------------------------------------------------------
// poisson - get/set optional margin,p,eps,swap flag & reduction args for poisson nll loss
// poissonloss  - functional form of poisson nll loss function
// ------------------------------------------------------------------------------------------------------
static torch::nn::PoissonNLLLossOptions poisson(K x,J i,Cast c) {
 Pairs p; J n=xargc(x,i,p); S s=nullptr; torch::nn::PoissonNLLLossOptions o;
 if(n && xsym(x,i+n-1,s)) n--;
 for(J j=0;j<n;++j)
  switch(j) {
   case 0: o.log_input(lbool(x,i+j,c,Setting::log)); break;
   case 1: o.full(lbool(x,i+j,c,Setting::full)); break;
   case 2: o.eps(ldouble(x,i+j,c,Setting::eps)); break;
   default: AT_ERROR(lmap(c),": unrecognized positional arg(s), expecting up to 4 args, log,full,eps,reduce");
  }
 while(xpair(p))
  switch(lset(p.k)) {
   case Setting::log:    o.log_input(lbool(p,c)); break;
   case Setting::full:   o.full(lbool(p,c)); break;
   case Setting::eps:    o.eps(ldouble(p,c)); break;
   case Setting::reduce: s=lsym(p,c); break;
   default: AT_ERROR("Unrecognized option: ",p.k," for poisson-nll loss"); break;
  }
 if(s) reduce(o.reduction(),c,s);
 return o;
}

static void poisson(bool a,K x,const torch::nn::PoissonNLLLossOptions& o) {
 const torch::nn::PoissonNLLLossOptions d;
 if(a || d.log_input() != o.log_input()) OPTION(x, log,  kb(o.log_input()));
 if(a || d.full()      != o.full())      OPTION(x, full, kb(o.full()));
 if(a || d.eps()       != o.eps())       OPTION(x, eps,  kf(o.eps()));
 reduce(a,x,o,d);
}

KAPI poissonloss(K a) {
 KTRY
  bool p; Tensor x,y; Cast c=Cast::poissonloss;
  TORCH_CHECK(a->t>=0, lmap(c)," loss not implemented for ",kname(a));
  TORCH_CHECK(a->n>=2, lmap(c)," loss expects (input;target;optional arg(s)..)");
  if(a->t) p=false, x=kput(a), y=x[1], x=x[0];
  else     p=xtenarg(a,x,y);
  return kresult(p, a->n==2 ? torch::nn::functional::poisson_nll_loss(x,y)
                            : torch::nn::functional::poisson_nll_loss(x,y,poisson(a,2,c)));
 KCATCH("poisson nll loss");
}

// -------------------------------------------------------------------------------------------------------------------
// ctc - connectionist temporal classification loss between continuous time series & target sequence
//       get/set args for CTC loss, blank value, flag for setting infinities -> zero & reduction method
// -------------------------------------------------------------------------------------------------------------------
static torch::nn::CTCLossOptions ctc(K x,J i,Cast c) {
 Pairs p; J n=xargc(x,i,p); S s=nullptr; torch::nn::CTCLossOptions o;
 if(n && xsym(x,i+n-1,s)) n--;
 for(J j=0;j<n;++j)
  switch(j) {
   case 0: o.blank(int64(x,i+j,c,Setting::blank)); break;
   case 1: o.zero_infinity(lbool(x,i+j,c,Setting::zeroinf)); break;
   default: AT_ERROR(lmap(c),": unrecognized positional arg(s), expecting up to 3 args, blank label, zero infinity flag & reduce mode");
  }
 while(xpair(p))
  switch(lset(p.k)) {
   case Setting::blank:   o.blank(int64(p,c)); break;
   case Setting::zeroinf: o.zero_infinity(lbool(p,c)); break;
   case Setting::reduce:  s=lsym(p,c); break;
   default: AT_ERROR("Unrecognized option: ",p.k," for CTC loss"); break;
  }
 if(s) reduce(o.reduction(),c,s);
 return o;
}

static void ctc(bool a,K x,const torch::nn::CTCLossOptions& o) {
 const torch::nn::CTCLossOptions d;
 if(a || d.blank()         != o.blank())         OPTION(x, blank,   kj(o.blank()));
 if(a || d.zero_infinity() != o.zero_infinity()) OPTION(x, zeroinf, kb(o.zero_infinity()));
 reduce(a,x,o,d);
}

KAPI Ctc(K a) {
 KTRY
  bool p; Tensor x,y,nx,ny;
  if(a->t) {
   AT_ERROR("CTC loss not implemented for ",kname(a->t));
  } else if(a->n < 4) {
   AT_ERROR("CTC loss expects at least 4 args, (input;target;input lengths;target lengths)");
  }
  p=xtenarg(a,x,y); xtenarg(a,2,nx,ny);
  return kresult(p, torch::nn::functional::ctc_loss(x,y,nx,ny,ctc(a,4,Cast::ctc)));
 KCATCH("CTC loss");
}

// ---------------------------------------------------------------------------------------------------
// lossinit - initialize loss modules by parsing loss fn name & optional args, return AnyModule
// lossopt - retrieve loss module options, return k dictionary of module name & options
// lossdict - dictionary of loss module & options or full state (w'class, empty name, parms & buffers)
// losswt - handle loss w'optional batch weights (e.g. bce/bcelogits)
// lossfwd - given loss object, calls forward function on remaining inputs and returns loss
// lossto - given loss object and device/data type, converts tensors in options (e.g. class weights)
// loss - main api function that creates/calls loss objects and queries their properties
// ---------------------------------------------------------------------------------------------------
static AnyModule lossinit(Cast c,K x,J i) {
 namespace nn=torch::nn;
 switch(c) {
  case Cast::bce:         return AnyModule(    BCELoss(             reduce<    BCELossOptions>(x,i,c)));
  case Cast::kl:          return AnyModule(nn::KLDivLoss(           reduce<nn::KLDivLossOptions>(x,i,c)));
  case Cast::l1:          return AnyModule(nn::L1Loss(              reduce<nn::L1LossOptions>(x,i,c)));
  case Cast::mse:         return AnyModule(nn::MSELoss(             reduce<nn::MSELossOptions>(x,i,c)));
  case Cast::multilabel:  return AnyModule(nn::MultiLabelMarginLoss(reduce<nn::MultiLabelMarginLossOptions>(x,i,c)));
  case Cast::smoothl1:    return AnyModule(nn::SmoothL1Loss(        reduce<nn::SmoothL1LossOptions>(x,i,c)));
  case Cast::softmargin:  return AnyModule(nn::SoftMarginLoss(      reduce<nn::SoftMarginLossOptions>(x,i,c)));

  case Cast::bcelogits:   return AnyModule(BCEWithLogitsLoss(classwt(x,i,c,BCEWithLogitsLossOptions())));
  case Cast::multisoft:   return AnyModule(nn::MultiLabelSoftMarginLoss(classwt(x,i,c,nn::MultiLabelSoftMarginLossOptions())));
  case Cast::ce:          return AnyModule(nn::CrossEntropyLoss(classwt<nn::CrossEntropyLossOptions>(x,i,c)));
  case Cast::nll:         return AnyModule(nn::NLLLoss(classwt<nn::NLLLossOptions>(x,i,c)));

  case Cast::hinge:       return AnyModule(nn::HingeEmbeddingLoss( margin<nn::HingeEmbeddingLossOptions>(x,i,c)));
  case Cast::cosineloss:  return AnyModule(nn::CosineEmbeddingLoss(margin<nn::CosineEmbeddingLossOptions>(x,i,c)));
  case Cast::margin:      return AnyModule(nn::MarginRankingLoss(  margin<nn::MarginRankingLossOptions>(x,i,c)));

  case Cast::multimargin: return AnyModule(nn::MultiMarginLoss(multi(x,i,c))); break;
  case Cast::triplet:     return AnyModule(nn::TripletMarginLoss(triplet(x,i,c))); break;
  case Cast::poissonloss: return AnyModule(nn::PoissonNLLLoss(poisson(x,i,c))); break;
  case Cast::ctc:         return AnyModule(nn::CTCLoss(ctc(x,i,c))); break;
  case Cast::pairwise:    return AnyModule(nn::PairwiseDistance(pairwise(x,i,c))); break;
  case Cast::similar:     return AnyModule(nn::CosineSimilarity(similar(x,i,c))); break;

  default: AT_ERROR("Unrecognized loss function: ",lmap(c));
 }
}

static K lossopt(bool a,Cast c,AnyModule& m) {
 namespace nn=torch::nn;
 K x=xD(ktn(KS,0),ktn(0,0));
 switch(c) {
  case Cast::bce:        reduce(a, x, m.get<BCELoss>()->options); break;
  case Cast::kl:         reduce(a, x, m.get<nn::KLDivLoss>()->options); break;
  case Cast::l1:         reduce(a, x, m.get<nn::L1Loss>()->options); break;
  case Cast::mse:        reduce(a, x, m.get<nn::MSELoss>()->options); break;
  case Cast::multilabel: reduce(a, x, m.get<nn::MultiLabelMarginLoss>()->options); break;
  case Cast::smoothl1:   reduce(a, x, m.get<nn::SmoothL1Loss>()->options); break;
  case Cast::softmargin: reduce(a, x, m.get<nn::SoftMarginLoss>()->options); break;

  case Cast::bcelogits:  classwt(a, x, m.get<BCEWithLogitsLoss>()->options); break;
  case Cast::multisoft:  classwt(a, x, m.get<nn::MultiLabelSoftMarginLoss>()->options); break;
  case Cast::ce:         classwt(a, x, m.get<nn::CrossEntropyLoss>()->options); break;
  case Cast::nll:        classwt(a, x, m.get<nn::NLLLoss>()->options); break;

  case Cast::hinge:       margin(a, x, m.get<nn::HingeEmbeddingLoss>()->options); break;
  case Cast::cosineloss:  margin(a, x, m.get<nn::CosineEmbeddingLoss>()->options); break;
  case Cast::margin:      margin(a, x, m.get<nn::MarginRankingLoss>()->options); break;

  case Cast::multimargin: multi(a, x, m.get<nn::MultiMarginLoss>()->options); break;
  case Cast::triplet:     triplet(a, x, m.get<nn::TripletMarginLoss>()->options); break;
  case Cast::poissonloss: poisson(a, x, m.get<nn::PoissonNLLLoss>()->options); break;
  case Cast::ctc:         ctc(a, x, m.get<nn::CTCLoss>()->options); break;
  case Cast::pairwise:    pairwise(a, x, m.get<torch::nn::PairwiseDistance>()->options); break;
  case Cast::similar:     similar (a, x, m.get<torch::nn::CosineSimilarity>()->options); break;

  default: AT_ERROR("Unrecognized loss module"); break;
 }
 return x;
}

K lossdict(bool a,bool b,Cast c,AnyModule &m) {
 //a:true if all options, b:true if full state (currently unreferenced, no parms/buffers for loss functions)
 K k=ktn(KS,2),v=ktn(0,2);
 kS(k)[0]=statekey(State::module);   kK(v)[0]=ks(lmap(c));
 kS(k)[1]=statekey(State::options);  kK(v)[1]=lossopt(a,c,m);
 return xD(k,v);
}

// this version of lossdict() called from generic state() function in k-level api
K lossdict(Ktag *g,K x) {
 bool a=env().alloptions;
 if(x->n==1 || (x->n==2 && xbool(x,1,a)))
  return lossdict(a,true,g->c,((Kmodule*)g)->m);
 else
  AT_ERROR("Loss state requires 1-2 args: previously allocated ptr or (ptr;options flag)");
}

Tensor losswt(Cast c,AnyModule& m,const Tensor& x,const Tensor&y) {
  return (c==Cast::bce || c==Cast::bcelogits) ? m.forward(x,y,Tensor{}) : m.forward(x,y);
}

static K lossfwd(Cast c,AnyModule& m,K a) {
 bool p; Tensor r,x,y,z;
 if(a->n==3) {
  p=xtenarg(a,1,x,y);
  r=losswt(c,m,x,y);
 } else if(a->n==4) {
  p=xtenarg(a,1,x,y,z);
  r=m.forward(x,y,z);
 } else if(c==Cast::ctc && a->n==5) {
  Tensor nx,ny; p=xtenarg(a,1,x,y); xtenarg(a,3,nx,ny);
  r=m.forward(x,y,nx,ny);
 } else {
  AT_ERROR("Unrecognized arg(s) for ",lmap(c)," forward call");
 }
 return kresult(p,r);
}

K lossto(Kmodule* l,const TensorOptions& o,bool a) {
 auto s=torch::typeMetaToScalarType(o.dtype()); auto m=l->m.ptr();
 if(o.has_device() && o.has_dtype()) m->to(o.device(),s,a);
 else if(o.has_device())             m->to(o.device(),a);
 else                                m->to(s,a);
 return (K)0;
}

KAPI loss(K x) {
 KTRY
  S s; bool a=env().alloptions; Kmodule *l; Kmodel *m;
  if(xsyms(x,s) || xsym(x,0,s)) {
   Cast c=lmap(s);
   return kloss(c, lossinit(c,x,1));
  } else if(xdict(x)) {    //define loss from state dictionary
   Cast c=lmap(statemodule(x));
   return kloss(c, lossinit(c,stateoptions(x),-1));
  } else if(((l=xloss(x))) || (xbool(x,1,a) && x->n==2 && ((l=xloss(x,0))))) {
   return lossdict(a,false,l->c,l->m); //given allocated loss ptr or ptr w'boolean, return options
  } else if((l=xloss(x,0)) && x->n>1) {
   return lossfwd(l->c,l->m,x); //else, run forward calculation w'loss and input,target,..
  } else if((m=xmodel(x))) {
   return kloss(m->lc,m->l);
  } else {
   AT_ERROR("Unrecognized arg(s)");
  }
 KCATCH("Loss module");
}

K lossattr(const AnyModule& m,Ktype k,Attr a) {
 switch(a) {
  case Attr::ref:     return kj(m.ptr().use_count()-1);
  case Attr::ptr:     return kj((intptr_t)m.ptr().get());
  case Attr::device:  return ks(objdevice(m.ptr()->buffers(), optsym(torch::Device(torch::kCPU))));
  default: AT_ERROR(mapattr(a),": not implemented for loss modules");
 }
}

// ----------------------------------
// loss fns defined in k namespace
// ----------------------------------
void lossfn(K x) {
 fn(x, "loss",        KFN(loss),1);
 fn(x, "bce",         KFN(bce),1);
 fn(x, "bcelogit1",   KFN(bcelogit1),1);
 fn(x, "bcelogit2",   KFN(bcelogit2),1);
 fn(x, "ce",          KFN(ce),1);
 fn(x, "cosineloss",  KFN(cosineloss),1);
 fn(x, "ctc",         KFN(Ctc),1);
 fn(x, "hinge",       KFN(hinge),1);
 fn(x, "kl",          KFN(kl),1);
 fn(x, "l1",          KFN(l1),1);
 fn(x, "margin",      KFN(Margin),1);
 fn(x, "mse",         KFN(mse),1);
 fn(x, "multilabel",  KFN(multilabel),1);
 fn(x, "multimargin", KFN(multimargin),1);
 fn(x, "multisoft",   KFN(multisoft),1);
 fn(x, "nll",         KFN(nll),1);
 fn(x, "poissonloss", KFN(poissonloss),1);
 fn(x, "smoothl1",    KFN(smoothl1),1);
 fn(x, "softmargin",  KFN(softmargin),1);
 fn(x, "triplet",     KFN(Triplet),1);
}
