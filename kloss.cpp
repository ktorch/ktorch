#include "ktorch.h"
#include "kloss.h"
#include "knn/distance.h"

namespace nn=torch::nn;
namespace fnn=torch::nn::functional;

// append a loss option to a k dictionary given dict,name & value
#define OPTION(x,k,v) dictadd(x, lset(Setting::k), v)

// -----------------------------------------------------------------------------
// Attrs - tuple for sym,cast,id,description,fwdflag,return type,argcount,args
// attr - template to derive forward result type & arg types
// lossattrs - return array of loss module attributes
// -----------------------------------------------------------------------------
template<class M,class R,class... A> static Attrs attr(size_t n,S s,Cast c,const char *d,R (M::*)(A...)) {
 auto m=sizeof...(A);
 static_assert(torch::detail::check_not_lvalue_references<A...>(), "module arg(s) must not take non-const references (use pointers)");
 static_assert(!std::is_void<R>::value, "module forward calculation returns void (use dummy arg)");
 if(!n) n=m;
 return std::make_tuple(s, c, typeid(M).hash_code(), d, true, knn::argmap<R>(), n, m, knn::argvector<A...>());
}

LossAttrs lossattrs() {
 return {{
  attr(2, cs("bce"),         Cast::bce,         "torch.nn.BCELoss",                  &knn::BCELossImpl::forward),
  attr(2, cs("bcelogits"),   Cast::bcelogits,   "torch.nn.BCEWithLogitsLoss",        &knn::BCEWithLogitsLossImpl::forward),
  attr(2, cs("ce"),          Cast::ce,          "torch.nn.CrossEntropyLoss",         &nn::CrossEntropyLossImpl::forward),
  attr(3, cs("cosineloss"),  Cast::cosineloss,  "torch.nn.CosineEmbeddingLoss",      &nn::CosineEmbeddingLossImpl::forward),
  attr(4, cs("ctc"),         Cast::ctc,         "torch.nn.CTCLoss",                  &nn::CTCLossImpl::forward),
  attr(2, cs("hinge"),       Cast::hinge,       "torch.nn.HingeEmbeddingLoss",       &nn::HingeEmbeddingLossImpl::forward),
  attr(2, cs("huber"),       Cast::huber,       "torch.nn.HuberLoss",                &nn::HuberLossImpl::forward),
  attr(2, cs("kl"),          Cast::kl,          "torch.nn.KLDivLoss",                &nn::KLDivLossImpl::forward),
  attr(2, cs("l1"),          Cast::l1,          "torch.nn.L1Loss",                   &nn::L1LossImpl::forward),
  attr(3, cs("margin"),      Cast::margin,      "torch.nn.MarginRankingLoss",        &nn::MarginRankingLossImpl::forward),
  attr(2, cs("mse"),         Cast::mse,         "torch.nn.MSELoss",                  &nn::MSELossImpl::forward),
  attr(2, cs("multilabel"),  Cast::multilabel,  "torch.nn.MultiLabelMarginLoss",     &nn::MultiLabelMarginLossImpl::forward),
  attr(2, cs("multimargin"), Cast::multimargin, "torch.nn.MultiMarginLoss",          &nn::MultiMarginLossImpl::forward),
  attr(2, cs("multisoft"),   Cast::multisoft,   "torch.nn.MultiLabelSoftMarginLoss", &nn::MultiLabelSoftMarginLossImpl::forward),
  attr(2, cs("nll"),         Cast::nll,         "torch.nn.NLLLoss",                  &nn::NLLLossImpl::forward),
  attr(2, cs("pairwise"),    Cast::pairwise,    "torch.nn.PairwiseDistance",         &nn::PairwiseDistanceImpl::forward),
  attr(2, cs("poissonloss"), Cast::poissonloss, "torch.nn.PoissonNLLLoss",           &nn::PoissonNLLLossImpl::forward),
  attr(2, cs("similar"),     Cast::similar,     "torch.nn.CosineSimilarity",         &nn::CosineSimilarityImpl::forward),
  attr(2, cs("smoothl1"),    Cast::smoothl1,    "torch.nn.SmoothL1Loss",             &nn::SmoothL1LossImpl::forward),
  attr(2, cs("softmargin"),  Cast::softmargin,  "torch.nn.SoftMarginLoss",           &nn::SoftMarginLossImpl::forward),
  attr(3, cs("triplet"),     Cast::triplet,     "torch.nn.TripletMarginLoss",        &nn::TripletMarginLossImpl::forward),
 }};
}

// -----------------------------------------------------------------------------
// lmap - map to/from sym to loss function name, e.g. `mse <-> Cast::mse
// lset - map to/from sym to loss setting enum, e.g. `reduce <-> Setting::reduce
// lpos - throw error if too many positional arguments
// lpair - throw error if unrecognized name in name-value pairs
// -----------------------------------------------------------------------------
Cast lmap(S s) {
 for(const auto& m:env().loss)
  if(std::get<0>(m)==s) return std::get<1>(m);
 TORCH_ERROR("unrecognized loss function: ",s);
}

S lmap(Cast c) {
 for(const auto& m:env().loss)
  if(std::get<1>(m)==c) return std::get<0>(m);
 TORCH_ERROR("unrecognized loss function: ",(I)c);
}

S lmap(Kmodule *l) {return lmap(l->c);}

static S lset(Setting s) {
 for(const auto& m:env().lset)
  if(std::get<1>(m)==s) return std::get<0>(m);
 TORCH_ERROR("unrecognized loss setting: ",(I)s);
}

static Setting lset(S s) {
 for(const auto& m:env().lset)
  if(std::get<0>(m)==s) return std::get<1>(m);
 TORCH_ERROR("unrecognized loss setting: ",s);
}

static void lpos(K x,Cast c,J n) {
 TORCH_ERROR(lmap(c),": expecting up to ",n," additional positional args, ",xlen(x)," given");
}

static void lpair(Cast c,const Pairs& p) {
 TORCH_ERROR(lmap(c)," option: ",p.k," not recognized");
}

// ----------------------------------------------------------------------------
// input checking fns with error msg specific to loss module name and setting
// check positional or name-value pairs for:
// lbool->boolean, lsym->sym, int64-integer, ldouble->double, ltensor -> tensor
// ----------------------------------------------------------------------------
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

static Tensor ltensor(K x,J i,Cast c,Setting s) {
 Tensor t;
 if(xten(x,i,t) || xempty(x,i))
  return t;
 else 
  return kput(x,i);
}

static Tensor ltensor(const Pairs& p,Cast c) {
 Tensor t;
 if(!pempty(p)) pten(p,t);
 return t;
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
  default: TORCH_ERROR(lmap(c)," reduce:",s," is not one of none,mean,sum");
 }
}

static void reduce(Reduce2& r,Cast c,S s) {
 switch(emap(s)) {
  case Enum::none:      r=torch::kNone; break;
  case Enum::batchmean: r=torch::kBatchMean; break;
  case Enum::mean:      r=torch::kMean; break;
  case Enum::sum:       r=torch::kSum; break;
  default: TORCH_ERROR(lmap(c)," reduce:",s," is not one of none,batchmean,mean,sum");
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
  TORCH_CHECK(lset(p.k)==Setting::reduce, "unrecognized option: ",p.k,", ",lmap(c)," loss expects single option: reduce");
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

template<typename O> static K reduce(bool a,const O& o,const O d=O());
template<typename O> static K reduce(bool a,const O& o,const O d) {K x=KDICT; reduce(a,x,o,d); return resolvedict(x);}

// ----------------------------------------------------------------------------
// beta - get/set beta & reduction options for smooth L1 loss
// ----------------------------------------------------------------------------
static nn::SmoothL1LossOptions beta(K x,J i,Cast c) {
 Pairs p; J n=xargc(x,i,p); S s=nullptr; nn::SmoothL1LossOptions o;
 if(n && xsym(x,i+n-1,s)) n--;
 for(J j=0;j<n;++j)
  switch(j) {
   case 0: o.beta(ldouble(x,i+j,c,Setting::beta)); break;
   default: lpos(x,c,i+j); break;
  }
 while(xpair(p))
  switch(lset(p.k)) {
   case Setting::beta:  o.beta(ldouble(p,c)); break;
   case Setting::reduce: s=lsym(p,c); break;
   default: lpair(c,p); break;
  }
 if(s) reduce(o.reduction(),c,s);
 return o;
}

static K beta(bool a,const nn::SmoothL1LossOptions& o) {
 K x=KDICT; const nn::SmoothL1LossOptions d;
 if(a || d.beta() != o.beta()) OPTION(x, beta, kf(o.beta()));
 reduce(a,x,o,d);
 return resolvedict(x);
}

// ------------------------------------------------------------
// crossentropy - set/get weight, index, smoothing & reduction
// ------------------------------------------------------------
static nn::CrossEntropyLossOptions crossentropy(K x,J i,Cast c) {
 Pairs p; J n=xargc(x,i,p); S s=nullptr; nn::CrossEntropyLossOptions o;
 if(n && xsym(x,i+n-1,s)) n--;
 for(J j=0;j<n;++j)
  switch(j) {
   case 0: o.weight(ltensor(x,i+j,c,Setting::weight)); break;
   case 1: o.ignore_index(int64(x,i+j,c,Setting::ignore)); break;
   case 2: o.label_smoothing(ldouble(x,i+j,c,Setting::smoothing)); break;
   default: lpos(x,c,i+j); break;
  }
 while(xpair(p))
  switch(lset(p.k)) {
   case Setting::weight:    o.weight(ltensor(p,c)); break;
   case Setting::ignore:    o.ignore_index(int64(p,c)); break;
   case Setting::smoothing: o.label_smoothing(ldouble(p,c)); break;
   case Setting::reduce:    s=lsym(p,c); break;
   default: lpair(c,p); break;
  }
 if(s) reduce(o.reduction(),c,s);
 return o;
}

static K crossentropy(bool a,const nn::CrossEntropyLossOptions& o) {
 K x=KDICT; const nn::CrossEntropyLossOptions d;
 if(a || o.weight().defined()) OPTION(x, weight, kget(o.weight()));
 if(a || d.ignore_index()    != o.ignore_index())    OPTION(x, ignore,    kj(o.ignore_index()));
 if(a || d.label_smoothing() != o.label_smoothing()) OPTION(x, smoothing, kf(o.label_smoothing()));
 reduce(a,x,o,d);
 return resolvedict(x);
}

// ----------------------------------------------------------------------------
// huber - get/set delta & reduction options for huber loss
// ----------------------------------------------------------------------------
static nn::HuberLossOptions huber(K x,J i,Cast c) {
 Pairs p; J n=xargc(x,i,p); S s=nullptr; nn::HuberLossOptions o;
 if(n && xsym(x,i+n-1,s)) n--;
 for(J j=0;j<n;++j)
  switch(j) {
   case 0: o.delta(ldouble(x,i+j,c,Setting::delta)); break;
   default: lpos(x,c,i+j); break;
  }
 while(xpair(p))
  switch(lset(p.k)) {
   case Setting::delta:  o.delta(ldouble(p,c)); break;
   case Setting::reduce: s=lsym(p,c); break;
   default: lpair(c,p); break;
  }
 if(s) reduce(o.reduction(),c,s);
 return o;
}

static K huber(bool a,const nn::HuberLossOptions& o) {
 K x=KDICT; const nn::HuberLossOptions d;
 if(a || d.delta() != o.delta()) OPTION(x, delta, kf(o.delta()));
 reduce(a,x,o,d);
 return resolvedict(x);
}

// ----------------------------------------------------------------------------------------
// multi - get/set optional power,margin,weight & reduction arguments for multi margin loss
// ----------------------------------------------------------------------------------------
static nn::MultiMarginLossOptions multi(K x,J i,Cast c) {
 Pairs p; J n=xargc(x,i,p); S s=nullptr; Tensor w; nn::MultiMarginLossOptions o;
 if(n && xsym(x,i+n-1,s)) n--;
 for(J j=0;j<n;++j)
  switch(j) {
   case 0: o.p(int64(x,i+j,c,Setting::p)); break;
   case 1: o.margin(ldouble(x,i+j,c,Setting::margin)); break;
   case 2: if(!xempty(x,i+j) && !xten(x,i+j,w)) w=kput(x,i+j); break;
   default: lpos(x,c,i+j); break;
  }
 while(xpair(p))
  switch(lset(p.k)) {
   case Setting::p:      o.p(int64(p,c)); break;
   case Setting::margin: o.margin(ldouble(p,c)); break;
   case Setting::weight: if(!pempty(p)) pten(p,w); break;
   case Setting::reduce: s=lsym(p,c); break;
   default: lpair(c,p); break;
  }
 if(w.defined()) o.weight(w);
 if(s) reduce(o.reduction(),c,s);
 return o;
}

static K multi(bool a,const nn::MultiMarginLossOptions& o) {
 K x=KDICT; const nn::MultiMarginLossOptions d;
 if(a || d.p()      != o.p())      OPTION(x, p,      kj(o.p()));
 if(a || d.margin() != o.margin()) OPTION(x, margin, kf(o.margin()));
 if(a || o.weight().defined())     OPTION(x, weight, kget(o.weight()));
 reduce(a,x,o,d);
 return resolvedict(x);
}

// -------------------------------------------------------------------------------------------
// classwt - set optional class weights & reduction mode, also index to ignore for some losses
//           classes with optional index use the same elements, so a templated fn is used,
//           but others use "weight" vs "pos_weight", requiring class-specific overloads
// -------------------------------------------------------------------------------------------
static void classwt(K x,J i,Cast c,S& s,Tensor& w) {
 Pairs p; J n=xargc(x,i,p); s=nullptr;
 if(n && xsym(x,i+n-1,s)) n--;
 if(n) {n--; if(!xempty(x,i+n) && !xten(x,i+n,w)) w=kput(x,i+n);}
 TORCH_CHECK(!n, lmap(c),": unrecognized positional arg(s), expected weights, reduce mode or (weights;reduce mode)");
 while(xpair(p))
  switch(lset(p.k)) {
   case Setting::weight: if(!pempty(p)) pten(p,w); break;
   case Setting::reduce: s=lsym(p,c); break;
   default: lpair(c,p); break;
  }
}

static auto& classwt(K x,J i,Cast c,knn::BCEWithLogitsLossOptions&& o) {
 S s; Tensor w; classwt(x,i,c,s,w);
 if(s) reduce(o.reduction(),c,s);
 if(w.defined()) o.pos_weight(w);
 return o;
}

static auto& classwt(K x,J i,Cast c,nn::MultiLabelSoftMarginLossOptions&& o) {
 S s; Tensor w; classwt(x,i,c,s,w);
 if(s) reduce(o.reduction(),c,s);
 if(w.defined()) o.weight(w);
 return o;
}

template<typename O> static O classwt(K x,J i,Cast c) {
 O o; Pairs p; J n=xargc(x,i,p); S s=nullptr; Tensor w;
 if(n && xsym(x,i+n-1,s)) n--; // allow last arg of symbol regardless
 for(J j=0;j<n;++j)
  switch(j) {
   case 0: if(!xempty(x,i+j) && !xten(x,i+j,w)) w=kput(x,i+j); break;
   case 1: o.ignore_index(int64(x,i+j,c,Setting::ignore)); break;
   default: lpos(x,c,i+j); break;
  }
 while(xpair(p))
  switch(lset(p.k)) {
   case Setting::weight: if(!pempty(p)) pten(p,w); break;
   case Setting::ignore: o.ignore_index(int64(p,c)); break;
   case Setting::reduce: s=lsym(p,c); break;
   default: lpair(c,p); break;
  }
 if(s) reduce(o.reduction(),c,s);
 if(w.defined()) o.weight(w);
 return o;
}

// -------------------------------------------------------------------------------------------
// classwt - get optional class weights & reduction mode, also index to ignore for some losses
// -------------------------------------------------------------------------------------------
static K classwt(bool a,const knn::BCEWithLogitsLossOptions& o) {
 K x=KDICT;
 if(a || o.pos_weight().defined()) OPTION(x,weight,kget(o.pos_weight()));
 reduce(a,x,o);
 return resolvedict(x);
}

static K classwt(bool a,const nn::MultiLabelSoftMarginLossOptions& o) {
 K x=KDICT;
 if(a || o.weight().defined()) OPTION(x,weight,kget(o.weight()));
 reduce(a,x,o);
 return resolvedict(x);
}

template<typename O> static K classwt(bool a,const O& o) {
 K x=KDICT; const O d;
 if(a || o.weight().defined()) OPTION(x, weight, kget(o.weight()));
 if(a || d.ignore_index() != o.ignore_index()) OPTION(x, ignore, kj(o.ignore_index()));
 reduce(a,x,o,d);
 return resolvedict(x);
}

// -------------------------------------------------------------------------
// poisson - set/get margin,p,eps,swap flag & reduction for poisson nll loss
// -------------------------------------------------------------------------
static nn::PoissonNLLLossOptions poisson(K x,J i,Cast c) {
 Pairs p; J n=xargc(x,i,p); S s=nullptr; nn::PoissonNLLLossOptions o;
 if(n && xsym(x,i+n-1,s)) n--;
 for(J j=0;j<n;++j)
  switch(j) {
   case 0: o.log_input(lbool(x,i+j,c,Setting::log)); break;
   case 1: o.full(lbool(x,i+j,c,Setting::full)); break;
   case 2: o.eps(ldouble(x,i+j,c,Setting::eps)); break;
   default: lpos(x,c,i+j); break;
  }
 while(xpair(p))
  switch(lset(p.k)) {
   case Setting::log:    o.log_input(lbool(p,c)); break;
   case Setting::full:   o.full(lbool(p,c)); break;
   case Setting::eps:    o.eps(ldouble(p,c)); break;
   case Setting::reduce: s=lsym(p,c); break;
   default: lpair(c,p); break;
  }
 if(s) reduce(o.reduction(),c,s);
 return o;
}

static K poisson(bool a,const nn::PoissonNLLLossOptions& o) {
 K x=KDICT; const nn::PoissonNLLLossOptions d;
 if(a || d.log_input() != o.log_input()) OPTION(x, log,  kb(o.log_input()));
 if(a || d.full()      != o.full())      OPTION(x, full, kb(o.full()));
 if(a || d.eps()       != o.eps())       OPTION(x, eps,  kf(o.eps()));
 reduce(a,x,o,d);
 return resolvedict(x);
}

// -----------------------------------------------------------------------------------------
// lossfunc - call loss function w'inputs,targets & optional reduction mode, other options..
// -----------------------------------------------------------------------------------------
static K lossfunc(K a,Cast c) {
 KTRY
  bool b,p; Tensor r,x,y;
  TORCH_CHECK(a->t>=0 && a->t<11, lmap(c),": not implemented for ",kname(a));
  TORCH_CHECK(a->n>=2, lmap(c),": expects at least 2 args, ",a->n," given");
  b=a->n==2;
  if(a->t) {
   TORCH_CHECK(b, lmap(c),": loss expects 2-element arg of input & target, ",a->n," value(s) given");
   x=kput(a); y=x[1]; x=x[0]; p=false;
  } else {
   p=xtenarg(a,x,y);
  }
  switch(c) {
   case Cast::ce: r=b ? fnn::cross_entropy(x,y) : fnn::cross_entropy(x,y,crossentropy(a,2,c)); break;
   case Cast::huber: r=b ? fnn::huber_loss(x,y) : fnn::huber_loss(x,y,huber(a,2,c)); break;
   case Cast::kl: r=b ? fnn::kl_div(x,y) : fnn::kl_div(x,y,reduce<nn::KLDivLossOptions>(a,2,c)); break;
   case Cast::l1: r=b ? fnn::l1_loss(x,y) : fnn::l1_loss(x,y,reduce<nn::L1LossOptions>(a,2,c)); break;
   case Cast::mse: r=b ? fnn::mse_loss(x,y) : fnn::mse_loss(x,y,reduce<nn::MSELossOptions>(a,2,c)); break;
   case Cast::multilabel:
    r=b ? fnn::multilabel_margin_loss(x,y)
        : fnn::multilabel_margin_loss(x,y,reduce<nn::MultiLabelMarginLossOptions>(a,2,c));
    break;
   case Cast::multimargin: r=b ? fnn::multi_margin_loss(x,y) : fnn::multi_margin_loss(x,y,multi(a,2,c)); break;
   case Cast::multisoft: r=b ? fnn::multilabel_soft_margin_loss(x,y) 
                             : fnn::multilabel_soft_margin_loss(x,y,classwt(a,2,c,nn::MultiLabelSoftMarginLossOptions()));
     break;
   case Cast::nll: r=b ? fnn::nll_loss(x,y) : fnn::nll_loss(x,y,classwt<nn::NLLLossOptions>(a,2,c)); break;
   case Cast::poissonloss: r=b ? fnn::poisson_nll_loss(x,y) : fnn::poisson_nll_loss(x,y,poisson(a,2,c)); break;
   case Cast::smoothl1:
    // bug in functional form w'beta requires workaround
    // https://github.com/pytorch/pytorch/issues/70163
    if(b) {
     r=fnn::smooth_l1_loss(x,y);
    } else {
     const auto &o=beta(a,2,c);
     r=fnn::detail::smooth_l1_loss(x,y,o.reduction(),o.beta());
    }
    break;
   case Cast::softmargin:
    r=b ? fnn::soft_margin_loss(x,y) : fnn::soft_margin_loss(x,y,reduce<nn::SoftMarginLossOptions>(a,2,c));
    break;
   default: TORCH_ERROR("unrecognized loss function"); break;
  }
 return kresult(p,r);
 KCATCH("loss");
}

KAPI ce(K x)          {return lossfunc(x, Cast::ce);}
KAPI Huber(K x)       {return lossfunc(x, Cast::huber);}
KAPI kl(K x)          {return lossfunc(x, Cast::kl);}
KAPI l1(K x)          {return lossfunc(x, Cast::l1);}
KAPI mse(K x)         {return lossfunc(x, Cast::mse);}
KAPI multilabel(K x)  {return lossfunc(x, Cast::multilabel);}
KAPI multimargin(K x) {return lossfunc(x, Cast::multimargin);}
KAPI multisoft(K x)   {return lossfunc(x, Cast::multisoft);}
KAPI nll(K x)         {return lossfunc(x, Cast::nll);}
KAPI poissonloss(K x) {return lossfunc(x, Cast::poissonloss);}
KAPI smoothl1(K x)    {return lossfunc(x, Cast::smoothl1);}
KAPI softmargin(K x)  {return lossfunc(x, Cast::softmargin);}

// --------------------------------------------------------------------------------
// binary cross entropy: optional 3rd input of batch weights
// bcearg - check arg to see if weight input or reduction option
// bce - binary cross entropy has option of batch weights, parses (x;y) or (x;y;wt)
// --------------------------------------------------------------------------------
static bool bcearg(K x) {return x->t==-KS || x->t==KS || xempty(x) || xdict(x);}

KAPI bce(K a) {
 KTRY
  auto r=knn::BCELossOptions().reduction(); bool p; Tensor x,y,w; Cast c=Cast::bce;
  TORCH_CHECK(0<=a->t && a->t<11, lmap(c),": not implemented for ",kname(a));
  TORCH_CHECK(a->n>1, lmap(c),": expects input,target,optional batch weight,optional reduce mode");
  if(a->t) {
   TORCH_CHECK(a->n<4,lmap(c),": unrecognized args, expecting 2-3 elements for input,target,optional weight");
   p=false; x=kput(a); if(a->n==3) w=x[2]; y=x[1]; x=x[0];
  } else {
   bool b=a->n==2 ? true : bcearg(kK(a)[2]);
   p = b ? xtenarg(a,x,y) : xtenarg(a,x,y,w);
   r=reduce<knn::BCELossOptions>(a,3-b,c).reduction();
  }
  return kresult(p, fnn::detail::binary_cross_entropy(x,y,w,r));
 KCATCH("bce");
}

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
  auto o=classwt(a,n,Cast::bcelogits,knn::BCEWithLogitsLossOptions());
  if(a->t) {
   p=false; x=kput(a); if(b) w=x[2]; y=x[1]; x=x[0];
  } else {
   p=b ? xtenarg(a,x,y,w) : xtenarg(a,x,y);
  }
  return kresult(p, fnn::detail::binary_cross_entropy_with_logits(x, y, w, o.reduction(), o.pos_weight()));
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
 TORCH_CHECK(!n, lmap(c),": unrecognized positional arg(s), expecting margin,reduce or both");
 while(xpair(p))
  switch(lset(p.k)) {
   case Setting::margin: o.margin(ldouble(p,c)); break;
   case Setting::reduce: s=lsym(p,c); break;
   default: lpair(c,p); break;
  }
 if(s) reduce(o.reduction(),c,s);
 return o;
}

template<typename O> static K margin(bool a,const O& o) {
 K x=KDICT; const O d;
 if(a || d.margin() != o.margin()) OPTION(x, margin, kf(o.margin()));
 reduce(a,x,o,d);
 return resolvedict(x);
}

static K marginloss(K a,Cast c) {
 KTRY
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
    r=b ? fnn::hinge_embedding_loss(x1,y) : fnn::hinge_embedding_loss(x1,y,margin<nn::HingeEmbeddingLossOptions>(a,2,c));
    break;
   case Cast::cosineloss:
    r=b ? fnn::cosine_embedding_loss(x1,x2,y) : fnn::cosine_embedding_loss(x1,x2,y,margin<nn::CosineEmbeddingLossOptions>(a,3,c));
    break;
   case Cast::margin:
    r=b ? fnn::margin_ranking_loss(x1,x2,y) : fnn::margin_ranking_loss(x1,x2,y,margin<nn::MarginRankingLossOptions>(a,3,c));
    break;
   default: TORCH_ERROR("unrecognized loss function"); break;
  }
  return kresult(p,r);
 KCATCH("loss")
}

KAPI hinge(K x)      {return marginloss(x, Cast::hinge);}
KAPI cosineloss(K x) {return marginloss(x, Cast::cosineloss);}
KAPI Margin(K x)     {return marginloss(x, Cast::margin);}

// ------------------------------------------------------------------------------------------------------
// triplet - get/set optional margin,p,eps,swap flag & reduction args in k array for triplet loss
// ------------------------------------------------------------------------------------------------------
static nn::TripletMarginLossOptions triplet(K x,J i,Cast c) {
 Pairs p; J n=xargc(x,i,p); S s=nullptr; nn::TripletMarginLossOptions o;
 if(n && xsym(x,i+n-1,s)) n--;
 for(J j=0;j<n;++j)
  switch(j) {
   case 0: o.margin(ldouble(x,i+j,c,Setting::margin)); break;
   case 1: o.p(ldouble(x,i+j,c,Setting::p)); break;
   case 2: o.eps(ldouble(x,i+j,c,Setting::eps)); break;
   case 3: o.swap(lbool(x,i+j,c,Setting::swap)); break;
   default: lpos(x,c,i+j); break;
  }
 while(xpair(p))
  switch(lset(p.k)) {
   case Setting::margin: o.margin(ldouble(p,c)); break;
   case Setting::p:      o.p(ldouble(p,c)); break;
   case Setting::eps:    o.eps(ldouble(p,c)); break;
   case Setting::swap:   o.swap(lbool(p,c)); break;
   case Setting::reduce: s=lsym(p,c); break;
   default: lpair(c,p); break;
  }
 if(s) reduce(o.reduction(),c,s);
 return o;
}

static K triplet(bool a,const nn::TripletMarginLossOptions& o) {
 K x=KDICT; const nn::TripletMarginLossOptions d;
 if(a || d.margin() != o.margin()) OPTION(x, margin, kf(o.margin()));
 if(a || d.p()      != o.p())      OPTION(x, p,      kf(o.p()));
 if(a || d.eps()    != o.eps())    OPTION(x, eps,    kf(o.eps()));
 if(a || d.swap()   != o.swap())   OPTION(x, swap,   kb(o.swap()));
 reduce(a,x,o,d);
 return resolvedict(x);
}

KAPI Triplet(K a) {
 KTRY
  bool p; Tensor x,y,z; Cast c=Cast::triplet;
  TORCH_CHECK(a->t>=0, lmap(c)," loss not implemented for ",kname(a));
  TORCH_CHECK(a->n>=2, lmap(c)," loss expects (anchor;positive;negative;optional arg(s)..)");
  if(a->t) p=false, x=kput(a), z=x[2], y=x[1], x=x[0];
  else     p=xtenarg(a,x,y,z);
  return kresult(p, a->n==3 ? fnn::triplet_margin_loss(x,y,z)
                            : fnn::triplet_margin_loss(x,y,z,triplet(a,3,c)));
 KCATCH("triplet margin loss");
}

// -------------------------------------------------------------------------------------------------------------------
// ctc - connectionist temporal classification loss between continuous time series & target sequence
//       get/set args for CTC loss, blank value, flag for setting infinities -> zero & reduction method
// -------------------------------------------------------------------------------------------------------------------
static nn::CTCLossOptions ctc(K x,J i,Cast c) {
 Pairs p; J n=xargc(x,i,p); S s=nullptr; nn::CTCLossOptions o;
 if(n && xsym(x,i+n-1,s)) n--;
 for(J j=0;j<n;++j)
  switch(j) {
   case 0: o.blank(int64(x,i+j,c,Setting::blank)); break;
   case 1: o.zero_infinity(lbool(x,i+j,c,Setting::zeroinf)); break;
   default: lpos(x,c,i+j); break;
  }
 while(xpair(p))
  switch(lset(p.k)) {
   case Setting::blank:   o.blank(int64(p,c)); break;
   case Setting::zeroinf: o.zero_infinity(lbool(p,c)); break;
   case Setting::reduce:  s=lsym(p,c); break;
   default: lpair(c,p); break;
  }
 if(s) reduce(o.reduction(),c,s);
 return o;
}

static K ctc(bool a,const nn::CTCLossOptions& o) {
 K x=KDICT; const nn::CTCLossOptions d;
 if(a || d.blank()         != o.blank())         OPTION(x, blank,   kj(o.blank()));
 if(a || d.zero_infinity() != o.zero_infinity()) OPTION(x, zeroinf, kb(o.zero_infinity()));
 reduce(a,x,o,d);
 return resolvedict(x);
}

KAPI Ctc(K a) {
 KTRY
  bool p; Tensor x,y,nx,ny;
  if(a->t) {
   TORCH_ERROR("cts loss not implemented for ",kname(a->t));
  } else if(a->n < 4) {
   TORCH_ERROR("ctc loss expects at least 4 args, (input;target;input lengths;target lengths)");
  }
  p=xtenarg(a,x,y); xtenarg(a,2,nx,ny);
  return kresult(p, fnn::ctc_loss(x,y,nx,ny,ctc(a,4,Cast::ctc)));
 KCATCH("ctc loss");
}

// ---------------------------------------------------------------------------------------------------
// lossinit - initialize loss modules by parsing loss fn name & optional args, return AnyModule
// lossoptions - retrieve loss module options, return k dictionary of module name & options
// lossget - dictionary of loss module & options or full state (w'class, empty name, parms & buffers)
// ---------------------------------------------------------------------------------------------------
static Moduleptr lossinit(Cast c,K x,J i) {
 switch(c) {
  case Cast::bce:         return knn::BCELoss(            reduce<knn::BCELossOptions>(x,i,c)).ptr();
  case Cast::kl:          return nn::KLDivLoss(           reduce<nn::KLDivLossOptions>(x,i,c)).ptr();
  case Cast::l1:          return nn::L1Loss(              reduce<nn::L1LossOptions>(x,i,c)).ptr();
  case Cast::mse:         return nn::MSELoss(             reduce<nn::MSELossOptions>(x,i,c)).ptr();
  case Cast::multilabel:  return nn::MultiLabelMarginLoss(reduce<nn::MultiLabelMarginLossOptions>(x,i,c)).ptr();
  case Cast::softmargin:  return nn::SoftMarginLoss(      reduce<nn::SoftMarginLossOptions>(x,i,c)).ptr();

  case Cast::smoothl1:    return nn::SmoothL1Loss(beta(x,i,c)).ptr();
  case Cast::huber:       return nn::HuberLoss(huber(x,i,c)).ptr();
  case Cast::bcelogits:   return knn::BCEWithLogitsLoss(classwt(x,i,c,knn::BCEWithLogitsLossOptions())).ptr();
  case Cast::multisoft:   return nn::MultiLabelSoftMarginLoss(classwt(x,i,c,nn::MultiLabelSoftMarginLossOptions())).ptr();
  case Cast::ce:          return nn::CrossEntropyLoss(crossentropy(x,i,c)).ptr();
  case Cast::nll:         return nn::NLLLoss(classwt<nn::NLLLossOptions>(x,i,c)).ptr();

  case Cast::hinge:       return nn::HingeEmbeddingLoss( margin<nn::HingeEmbeddingLossOptions>(x,i,c)).ptr();
  case Cast::cosineloss:  return nn::CosineEmbeddingLoss(margin<nn::CosineEmbeddingLossOptions>(x,i,c)).ptr();
  case Cast::margin:      return nn::MarginRankingLoss(  margin<nn::MarginRankingLossOptions>(x,i,c)).ptr();

  case Cast::multimargin: return nn::MultiMarginLoss(multi(x,i,c)).ptr();
  case Cast::triplet:     return nn::TripletMarginLoss(triplet(x,i,c)).ptr();
  case Cast::poissonloss: return nn::PoissonNLLLoss(poisson(x,i,c)).ptr();
  case Cast::ctc:         return nn::CTCLoss(ctc(x,i,c)).ptr();
  case Cast::pairwise:    return nn::PairwiseDistance(knn::pairwise(x,i,c)).ptr();
  case Cast::similar:     return nn::CosineSimilarity(knn::similar(x,i,c)).ptr();

  default: TORCH_ERROR("unrecognized loss function: ",lmap(c));
 }
}

K lossoptions(bool a,Cast c,const Module& m) {
 switch(c) {
  case Cast::bce:         return reduce(a,m.as<knn::BCELoss>()->options);
  case Cast::kl:          return reduce(a,m.as<nn::KLDivLoss>()->options);
  case Cast::l1:          return reduce(a,m.as<nn::L1Loss>()->options);
  case Cast::mse:         return reduce(a,m.as<nn::MSELoss>()->options);
  case Cast::multilabel:  return reduce(a,m.as<nn::MultiLabelMarginLoss>()->options);
  case Cast::softmargin:  return reduce(a,m.as<nn::SoftMarginLoss>()->options);

  case Cast::smoothl1:    return beta(a,m.as<nn::SmoothL1Loss>()->options);
  case Cast::huber:       return huber(a,m.as<nn::HuberLoss>()->options);
  case Cast::bcelogits:   return classwt(a,m.as<knn::BCEWithLogitsLoss>()->options);
  case Cast::multisoft:   return classwt(a,m.as<nn::MultiLabelSoftMarginLoss>()->options);
  case Cast::ce:          return crossentropy(a,m.as<nn::CrossEntropyLoss>()->options);
  case Cast::nll:         return classwt(a,m.as<nn::NLLLoss>()->options);

  case Cast::hinge:       return margin(a,m.as<nn::HingeEmbeddingLoss>()->options);
  case Cast::cosineloss:  return margin(a,m.as<nn::CosineEmbeddingLoss>()->options);
  case Cast::margin:      return margin(a,m.as<nn::MarginRankingLoss>()->options);

  case Cast::multimargin: return multi(a,m.as<nn::MultiMarginLoss>()->options);
  case Cast::triplet:     return triplet(a,m.as<nn::TripletMarginLoss>()->options);
  case Cast::poissonloss: return poisson(a,m.as<nn::PoissonNLLLoss>()->options);
  case Cast::ctc:         return ctc(a,m.as<nn::CTCLoss>()->options);
  case Cast::pairwise:    return knn::pairwise(a,m.as<nn::PairwiseDistance>()->options);
  case Cast::similar:     return knn::similar(a,m.as<nn::CosineSimilarity>()->options);

  default: TORCH_ERROR("unrecognized loss module");
 }
}

K lossget(bool a,bool b,Cast c,const Module &m) {
 //a:true if all options, b:true if full state (currently unreferenced, no parms/buffers for loss functions)
 K k=ktn(KS,2),v=ktn(0,2);
 kS(k)[0]=statekey(State::module);   kK(v)[0]=ks(lmap(c));
 kS(k)[1]=statekey(State::options);  kK(v)[1]=lossoptions(a,c,m);
 return xD(k,v);
}

// ----------------------------------------------------------------------------
// losscalc - calculate loss given input(s) & target(s), return tensor
// ----------------------------------------------------------------------------
Tensor losscalc(Kmodule *l,const Tensor& x,const Tensor&y) {
 auto& m=l->module();
 switch(l->c) {
  case Cast::bce:         return m.as<knn::BCELoss>()->forward(x,y,{});             // no batch weights defined
  case Cast::bcelogits:   return m.as<knn::BCEWithLogitsLoss>()->forward(x,y,{});   // no batch weights defined
  case Cast::ce:          return m.as<nn::CrossEntropyLoss>()->forward(x,y);
  case Cast::hinge:       return m.as<nn::HingeEmbeddingLoss>()->forward(x,y);
  case Cast::huber:       return m.as<nn::HuberLoss>()->forward(x,y);
  case Cast::kl:          return m.as<nn::KLDivLoss>()->forward(x,y);
  case Cast::l1:          return m.as<nn::L1Loss>()->forward(x,y);
  case Cast::mse:         return m.as<nn::MSELoss>()->forward(x,y);
  case Cast::multilabel:  return m.as<nn::MultiLabelMarginLoss>()->forward(x,y);
  case Cast::multimargin: return m.as<nn::MultiMarginLoss>()->forward(x,y);
  case Cast::multisoft:   return m.as<nn::MultiLabelSoftMarginLoss>()->forward(x,y);
  case Cast::nll:         return m.as<nn::NLLLoss>()->forward(x,y);
  case Cast::pairwise:    return m.as<nn::PairwiseDistance>()->forward(x,y);
  case Cast::poissonloss: return m.as<nn::PoissonNLLLoss>()->forward(x,y);
  case Cast::smoothl1:    return m.as<nn::SmoothL1Loss>()->forward(x,y);
  case Cast::similar:     return m.as<nn::CosineSimilarity>()->forward(x,y);
  case Cast::softmargin:  return m.as<nn::SoftMarginLoss>()->forward(x,y);
  default: TORCH_ERROR(lmap(l),": loss calculation not implemented for single tensor input & target");
 }
}

Tensor losscalc(Kmodule *l,const Tensor& x,const Tensor& y,const Tensor& z) {
 auto& m=l->module();
 switch(l->c) {
  case Cast::bce:        return m.as<knn::BCELoss>()->forward(x,y,z);                 // z=batch weights
  case Cast::bcelogits:  return m.as<knn::BCEWithLogitsLoss>()->forward(x,y,z);       // z=batch weights
  case Cast::cosineloss: return m.as<nn::CosineEmbeddingLoss>()->forward(x,y,z); // x=input1,y=input2,z=target
  case Cast::margin:     return m.as<nn::MarginRankingLoss>()->forward(x,y,z);   // x=input1,y=input2,z=target
  case Cast::triplet:    return m.as<nn::TripletMarginLoss>()->forward(x,y,z);   // x=anchor,y=positive,z=negative
  default: TORCH_ERROR(lmap(l),":  loss calculation not implemented for 3 tensors, e.g. input,target,weight  or input1,input2,target");
 }
}

Tensor losscalc(Kmodule *l,const Tensor& x,const Tensor& y,const Tensor& nx,const Tensor& ny) {
 auto& m=l->module();
 switch(l->c) {
  case Cast::ctc:  return m.as<nn::CTCLoss>()->forward(x,y,nx,ny);
  default: TORCH_ERROR(lmap(l),": unable to calculate loss given 4 tensors");
 }
}

// ---------------------------------------------------------------------------
// losscalc using variant inputs derived from k api arguments
// ---------------------------------------------------------------------------
static Tensor losscalc(Kmodule *l,const Input& x,const Input &y) {
 return c10::visit(
  c10::overloaded (
   [&l](const Tensor& x,const Tensor& y) {return losscalc(l,x,y);},
   [&l](const Tensor& x,const TensorVector& y) {
    switch(y.size()) {
     case 1:   return losscalc(l,x,y[0]);
     case 2:   return losscalc(l,x,y[0],y[1]);
     default: TORCH_ERROR(lmap(l),": loss calculation not implemented for ",y.size(), "-element vector of targets");
    }
   },
   [&l](const TensorVector& x,const Tensor& y) {
    switch(x.size()) {
     case 1:   return losscalc(l,x[0],y);
     case 2:   return losscalc(l,x[0],x[1],y);
     default: TORCH_ERROR(lmap(l),": loss calculation not implemented for ",x.size(), "-element vector of inputs");
    }
   },
   [&l](auto x,auto y) {TORCH_ERROR(lmap(l),": loss calculation not implemented for ",inputname(x)," input & ",inputname(y)," target"); return Tensor();}
  ), x,y);
}

// ---------------------------------------------------------------------------
// lossinput - extract relevant tensor(s) from module output as input to loss
//           - currently returns tensor only, but may include vector,etc.
// losscalc - w'variant output from model's forward calc & supplied target(s)
// ---------------------------------------------------------------------------
static Input lossinput(Kmodule *m,const Output& x) {
 return c10::visit(
  c10::overloaded (
   [](const Tensor& x)       {return x;},
   [](const Tuple&  x)       {return std::get<0>(x);},
   [](const Nested& x)       {return std::get<0>(x);},
   [](const TensorVector& x) {return x.front();},
   [&m](auto x) {TORCH_ERROR(msym(m->f.out() == Cast::undefined ? m->c : m->f.out()),
                             ": ",output(x)," output not implemented for loss calculation"); return Tensor();}
  ),x);
}

Tensor losscalc(Kmodel *m,const Output& x,const Input &y) {
 return losscalc(m->kloss(), lossinput(m->kmodule(),x), y);
}

// ---------------------------------------------------------------------------
// mixeddevices - return true if cpu & cuda, set first CUDA device encountered
// changedevice - move tensor to cuda device if mix of both
// lossarg - parse tensor args given module and outputs & targets
// ---------------------------------------------------------------------------
static bool mixeddevices(const TensorVector& v,Device& d) {
 bool a=false,b=false;
 for(const auto&t:v)
  if(!a && t.is_cuda())  // if first CUDA device encountered
   a=true, d=t.device(); // set CUDA flag true and set device
  else if(t.is_cpu())    // else if CPU
   b=true;               // set flag for CPU
 return a && b;          // return true if a mix of a devices: cuda & cpu
}

TensorVector changedevice(const TensorVector& a,Device& d) {
 size_t i=0; TensorVector v(a.size());
 for(const auto& t:a)
  v[i]=t.is_cpu() ? t.to(d) : t;
 return v;
}

static K lossarg(Kmodule *l,K a) {
 Input x=modelarg(a,1,"loss");
 if(auto vp=c10::get_if<TensorVector>(&x)) {
  Device d(DeviceType::CPU);
  const auto& v=mixeddevices(*vp,d) ? changedevice(*vp,d) : *vp;
  switch(v.size()) {
   case 2: return kten(losscalc(l,v[0],v[1]));
   case 3: return kten(losscalc(l,v[0],v[1],v[2]));
   case 4: return kten(losscalc(l,v[0],v[1],v[2],v[4]));
   default: TORCH_ERROR("loss: no calculation implemented for ",v.size()," tensors");
  }
 } else {
   TORCH_ERROR("loss: no calculation implemented for ",inputname(x)," input");
 }
}

// -----------------------------------------------------------------------------------------------
// loss - main api function that creates/calls loss objects and queries their properties
// -----------------------------------------------------------------------------------------------
KAPI loss(K x) {
 KTRY
  S s; bool a=env().alloptions; Kmodule *l; Kmodel *m;
  if(xsyms(x,s) || xsym(x,0,s)) {
   Cast c=lmap(s);
   return kmodule(c, lossinit(c,x,1), Class::loss);
  } else if(xdict(x)) {    //define loss from state dictionary
   Cast c=lmap(statemodule(x));
   return kmodule(c, lossinit(c,stateoptions(x),-1), Class::loss);
  } else if(((l=xloss(x))) || (xbool(x,1,a) && x->n==2 && ((l=xloss(x,0))))) {
   return lossget(a,false,l->c,l->module()); //return options
  } else if(((m=xmodel(x,0)) || (l=xloss(x,0))) && x->n>1) {
   if(m) l=m->kloss();                  // given model or loss w'additional arg(s)
// return losscalc(l,x);                // run forward calcs w'inputs & targets
   return lossarg(l,x);                 // run forward calcs w'inputs & targets
  } else if((m=xmodel(x))) {
   return kmodule(m->kloss());          // extract loss module from model
  } else {
   TORCH_ERROR("loss: unrecognized arg(s)");
  }
 KCATCH("loss");
}

// -----------------------------------------------------------------------
// lossexample - return dictionary of example options for each loss module
// -----------------------------------------------------------------------
K lossexample(Cast c) {
 switch(c) {
  case Cast::bce:         return reduce(true,nn::BCELossOptions());
  case Cast::bcelogits:   return classwt(true,knn::BCEWithLogitsLossOptions());
  case Cast::ce:          return crossentropy(true,nn::CrossEntropyLossOptions());
  case Cast::cosineloss:  return margin(true,nn::CosineEmbeddingLossOptions()); 
  case Cast::ctc:         return ctc(true,nn::CTCLossOptions());
  case Cast::hinge:       return margin(true,nn::HingeEmbeddingLossOptions()); 
  case Cast::huber:       return huber(true,nn::HuberLossOptions()); 
  case Cast::kl:          return reduce(true,nn::KLDivLossOptions());
  case Cast::l1:          return reduce(true,nn::L1LossOptions());
  case Cast::margin:      return margin(true,nn::MarginRankingLossOptions()); 
  case Cast::mse:         return reduce(true,nn::MSELossOptions());
  case Cast::multilabel:  return reduce(true,nn::MultiLabelMarginLossOptions());
  case Cast::multimargin: return multi(true,nn::MultiMarginLossOptions());
  case Cast::multisoft:   return classwt(true,nn::MultiLabelSoftMarginLossOptions());
  case Cast::nll:         return classwt(true,nn::NLLLossOptions());
  case Cast::pairwise:    return modexample(c);
  case Cast::poissonloss: return poisson(true,nn::PoissonNLLLossOptions());
  case Cast::similar:     return modexample(c);
  case Cast::smoothl1:    return beta(true,nn::SmoothL1LossOptions());
  case Cast::softmargin:  return reduce(true,nn::SoftMarginLossOptions());
  case Cast::triplet:     return triplet(true,nn::TripletMarginLossOptions());
  default: TORCH_ERROR("no example options available for loss enumeration: ",(I)c);
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
 fn(x, "huber",       KFN(Huber),1);
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
