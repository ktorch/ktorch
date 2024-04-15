#include "ktorch.h"
#include "kopt.h"
namespace nn=torch::nn;

#define OPTBUFFER(x,o,k) dictadd(x, #k, kget(o->k))
#define OPTSET(x,k,v) dictadd(x, oset(Setting::k), v)

const double LR = 0.01; // default learning rate, used with SGD, which has no default

using Options     = torch::optim::OptimizerOptions;
using ParamState  = torch::optim::OptimizerParamState;
using ParamGroup  = torch::optim::OptimizerParamGroup;
using ParamGroups = std::vector<ParamGroup>;

using Adagrad           = torch::optim::Adagrad;
using AdagradOptions    = torch::optim::AdagradOptions;
using AdagradParamState = torch::optim::AdagradParamState;
using Adam              = torch::optim::Adam;
using AdamOptions       = torch::optim::AdamOptions;
using AdamParamState    = torch::optim::AdamParamState;
using AdamW             = torch::optim::AdamW;
using AdamWOptions      = torch::optim::AdamWOptions;
using AdamWParamState   = torch::optim::AdamWParamState;
using Lamb              = torch::optim::Lamb;
using LambOptions       = torch::optim::LambOptions;
using LambParamState    = torch::optim::LambParamState;
using LBFGS             = torch::optim::LBFGS;
using LBFGSOptions      = torch::optim::LBFGSOptions;
using LBFGSParamState   = torch::optim::LBFGSParamState;
using RMSprop           = torch::optim::RMSprop;
using RMSpropOptions    = torch::optim::RMSpropOptions;
using RMSpropParamState = torch::optim::RMSpropParamState;
using SGD               = torch::optim::SGD;
using SGDOptions        = torch::optim::SGDOptions;
using SGDParamState     = torch::optim::SGDParamState;

// --------------------------------------------------------------------------------------
// kopt - given optimizer type & shared pointer to newly created optimizer, return k ptr
// omap - map to/from optimizer symbol/enumeration
// oset - optimizer settings, map sym <-> enum
// oten - return 1 if tensor defined else 0 (used to count number of tensors in buffers)
//        also count tensors in vector or deque for lbfgs
// osize - optimizer size, i.e. number of parameters defined 
//        (defined in pytorch, but marked with "obsolete" warning)
// --------------------------------------------------------------------------------------
K kopt(Cast x,const Optptr& y,const Moduleptr& z) {return kptr(new Kopt(x,y,z));}
static K kopt(Kopt* o) {return kopt(o->c, o->o, o->m);}

static Cast omap(S s) {
 for(const auto& m:env().opt)
  if(s==std::get<0>(m)) return std::get<1>(m);
 TORCH_ERROR("unrecognized optimizer: ",s);
}

S omap(Cast c) {
 for(auto& m:env().opt)
  if(c==std::get<1>(m)) return std::get<0>(m);
 TORCH_ERROR("unrecognized optimizer: ",(I)c);
}

static Setting oset(S s) {
 for(const auto& m:env().oset)
  if(s==std::get<0>(m)) return std::get<1>(m);
 TORCH_ERROR("unrecognized optimizer setting: ",s);
}

static S oset(Setting e) {
 for(auto& m:env().oset) if(e==std::get<1>(m)) return std::get<0>(m);
 TORCH_ERROR("unrecognized optimizer setting: ",(I)e);
}

static J oten(const int64_t& t) {return 0;}
static J oten(const Tensor& t)  {return t.defined() ? 1 : 0;}
static J oten(const TensorDeque& v) {return v.size();}
static J oten(const c10::optional<TensorVector>& v) {return v ? v.value().size() : 0;}

size_t osize(const Optimizer& o) {
  size_t n=0; for(const auto& g:o.param_groups()) n+=g.params().size(); return n;
}

// --------------------------------------------------------------------------------------
// code  - check args for symbol, else error w'optimizer & setting name
// --------------------------------------------------------------------------------------
static c10::optional<std::string> code(K x,J i,Cast c,Setting s) {
 S a;
 TORCH_CHECK(xsym(x,i,a), omap(c)," ",oset(s),": expected symbol, given ",kname(x,i));
 if(null(a))
  return c10::nullopt;
 else
  return a;
}

static c10::optional<std::string> code(const Pairs& p,Cast c) {
 TORCH_CHECK(p.t==-KS, omap(c)," ",p.k,": expected symbol, given ",kname(p.t));
 if(null(p.s))
  return c10::nullopt;
 else
  return p.s;
}

// -----------------------------------------------------------------------------
// flag - return boolean if k boolean supplied, else error w'optimizer & setting
// -----------------------------------------------------------------------------
static bool flag(K x,J i,Cast c,Setting s) {
 bool b;
 TORCH_CHECK(xbool(x,i,b), omap(c)," ",oset(s),": expected boolean, given ",kname(x,i));
 return b;
}

static bool flag(const Pairs& p,Cast c) {
 TORCH_CHECK(p.t==-KB, omap(c)," ",p.k,": expected boolean scalar, given ",kname(p.t));
 return p.b;
}

// ---------------------------------------------------------------------
// int64 - check args for long int, else error w'optimizer & option name
// int64n - int64 but returns optional, i.e. nullopt if k value is null
// ---------------------------------------------------------------------
static int64_t int64(K x,J i,Cast c,Setting s) {
 int64_t n;
 TORCH_CHECK(xint64(x,i,n), omap(c)," ",oset(s),": expected long scalar, given ",kname(x,i));
 return n;
}

static int64_t int64(const Pairs& p,Cast c) {
 TORCH_CHECK(p.t==-KJ, omap(c)," ",p.k,": expected long scalar, given ",kname(p.t));
 return p.j;
}

static c10::optional<int64_t> int64n(K x,J i,Cast c,Setting s) {
 auto n=int64(x,i,c,s);
 if(null(n))
  return c10::nullopt;
 else
  return n;
}

static c10::optional<int64_t> int64n(const Pairs& p,Cast c) {
 auto n=int64(p,c);
 if(null(n))
  return c10::nullopt;
 else
  return n;
}

// ---------------------------------------------------------------------------
// numeric - return double given long/double, else error w'optimizer & setting
// ---------------------------------------------------------------------------
static double numeric(K x,J i,Cast c,Setting s) {
 double f;
 TORCH_CHECK(xnum(x,i,f), omap(c)," ",oset(s),": expected long/double scalar, given ",kname(x,i));
 return f;
}

static double numeric(const Pairs& p,Cast c) {
 TORCH_CHECK(p.t==-KJ || p.t==-KF, omap(c)," ",p.k,": expected long/double scalar, given ",kname(p.t));
 return p.t==-KJ ? p.j : p.f;
}

// -----------------------------------------------------------------------------------
// opos - throw error if too many positional arguments
// opair - throw error if unrecognized name in name-value pairs
// -----------------------------------------------------------------------------------
static void opos(K x,Cast c,J n) {
 TORCH_ERROR(omap(c),": expecting up to ",n," positional args, ",xlen(x)," given");
}

void opair(Cast c,const Pairs& p) {
 TORCH_ERROR(omap(c)," option: ",p.k," not recognized");
}

// --------------------------------------------------------------------------------------
// getoptions - set defaults if undefined, return reference to optimizer-specific options
//            - specific function for SGD optimizer as it has no default learning rate
// --------------------------------------------------------------------------------------
static SGDOptions& getoptions(ParamGroup& g) {
 if(!g.has_options()) g.set_options(std::make_unique<SGDOptions>(LR));
 return static_cast<SGDOptions&>(g.options());
}

template<typename O> static O& getoptions(ParamGroup& g) {
 if(!g.has_options()) g.set_options(std::make_unique<O>());
 return static_cast<O&>(g.options());
}

// ---------------------------------------------------------------------------------------
// findbuffer - find buffer in parameter-level dictionary, w'required type for scalar
// deque - read x dictionary into a deque of tensors (used with LBFGS optimizer)
// ---------------------------------------------------------------------------------------
static K findbuffer(K x,const std::string &s,short t=nh);
static K findbuffer(K x,const std::string &s,short t) {
 TORCH_CHECK(xdict(x), "dictionary expected, ",kname(x)," given, unable to find parameter ",s);
 K k=kK(x)[0], v=kK(x)[1]; J i=kfind(k,s);
 if(i<0)
  return nullptr;
 TORCH_CHECK(!v->t, "general list of values expected, ",kname(v)," given, unable to find parameter ",s);
 K r=kK(v)[i];
 TORCH_CHECK(t==nh || t==r->t, s,": ",kname(t)," expected, ",kname(r->t)," supplied");
 return xnull(r) ? nullptr : r;
}

static TensorDeque deque(K x,const std::string& s,const Device& d) {
 TORCH_CHECK(!x->t, "deque buffer: ",s,", expected list but given ",kname(x->t));
 TensorDeque q; q.resize(x->n);
 for(J i=0; i<x->n; ++i)
  q[i]=kput(kK(x)[i]).to(d);
 return q;
}
  
// -------------------------------------------------------------------------------
// adagrad - set/get options for adagrad optimizer
// adaget - retrieve parameter buffers from adagrad optimizer into k dictionary
// adaput - given k dictionary of buffers, put values into adagrad optimizer state
// adasize - tensor count, elements or bytes in parameter buffers
// -------------------------------------------------------------------------------
static void adagrad(K x,J i,Cast c,ParamGroup& g) {
 auto& o=getoptions<AdagradOptions>(g); Pairs p; J n=xargc(x,i,p);
 for(J j=0;j<n;++j)
  switch(j) {
   case 0: o.lr(numeric(x,i+j,c,Setting::lr)); break;
   case 1: o.lr_decay(numeric(x,i+j,c,Setting::lrdecay)); break;
   case 2: o.weight_decay(numeric(x,i+j,c,Setting::decay)); break;
   case 3: o.initial_accumulator_value(numeric(x,i+j,c,Setting::init)); break;
   case 4: o.eps(numeric(x,i+j,c,Setting::eps)); break;
   default: opos(x,c,i+j); break;
  }
 while(xpair(p))
  switch(oset(p.k)) {
   case Setting::lr:      o.lr(numeric(p,c)); break;
   case Setting::lrdecay: o.lr_decay(numeric(p,c)); break;
   case Setting::decay:   o.weight_decay(numeric(p,c)); break;
   case Setting::init:    o.initial_accumulator_value(numeric(p,c)); break;
   case Setting::eps:     o.eps(numeric(p,c)); break;
   default: opair(c,p); break;
  }
}

static K adagrad(bool a,const AdagradOptions& o) {
 //return all or non-default options as k dictionary
 K x=KDICT; AdagradOptions d; OPTSET(x, lr, kf(o.lr()));
 if(a || d.lr_decay()     != o.lr_decay())     OPTSET(x, lrdecay, kf(o.lr_decay()));
 if(a || d.weight_decay() != o.weight_decay()) OPTSET(x, decay,   kf(o.weight_decay()));
 if(a || d.initial_accumulator_value() !=
         o.initial_accumulator_value())        OPTSET(x, init,    kf(o.initial_accumulator_value()));
 if(a || d.eps()          != o.eps())          OPTSET(x, eps,     kf(o.eps()));
 return resolvedict(x);
}

static K adaget(const AdagradParamState& s) {
 K x=KDICT;
 dictadd(x, "step", kj(s.step()));
 dictadd(x, "sum",  kget(s.sum()));
 return x;
}

//static void adaput(K x,const Device& d,const std::string& k,Optimizer& o) {
static void adaput(K x,const Device& d,void *k,Optimizer& o) {
 K v; auto s=std::make_unique<AdagradParamState>();
 if((v=findbuffer(x,"step",-KJ))) s->step(v->j);
 if((v=findbuffer(x,"sum")))      s->sum(kput(v).to(d));
 o.state()[k]=std::move(s);
}

static J adasize(Attr a,const AdagradParamState& s) {
 //count of tensors/elements/bytes in parm buffers
 switch(a) {
  case Attr::tensorcount: return     oten(s.step()) +     oten(s.sum());
  case Attr::elements:    return   objnum(s.step()) +   objnum(s.sum());
  case Attr::bytes:       return objbytes(s.step()) + objbytes(s.sum());
  default: TORCH_ERROR("adagrad: unexpected attribute for counting buffer sizes");
 }
}

// --------------------------------------------------------------------------------
// adam - set/get options for adam/adamw optimizer
// adamget - retrieve parameter buffers from adam/adamw optimizer into k dictionary
// adamput - given k dictionary of buffers, put into adam/adamw optimizer state
// adamsize - tensor count, elements or bytes in parameter buffers
// --------------------------------------------------------------------------------
template<typename O> static void adam(K x,J i,Cast c,ParamGroup& g) {
 auto& o=getoptions<O>(g); Pairs p; J n=xargc(x,i,p); 
 for(J j=0;j<n;++j)
  switch(j) {
   case 0: o.lr(numeric(x,i+j,c,Setting::lr)); break;
   case 1: o.betas(std::make_tuple(numeric(x,i+j,c,Setting::beta1),std::get<1>(o.betas()))); break;
   case 2: o.betas(std::make_tuple(std::get<0>(o.betas()),numeric(x,i+j,c,Setting::beta2))); break;
   case 3: o.eps(numeric(x,i+j,c,Setting::eps)); break;
   case 4: o.weight_decay(numeric(x,i+j,c,Setting::decay)); break;
   case 5: o.amsgrad(flag(x,i+j,c,Setting::amsgrad)); break;
   default: opos(x,c,i+j); break;
  }
 while(xpair(p))
  switch(oset(p.k)) {
   case Setting::lr:      o.lr(numeric(p,c)); break;
   case Setting::beta1:   o.betas(std::make_tuple(numeric(p,c),std::get<1>(o.betas()))); break;
   case Setting::beta2:   o.betas(std::make_tuple(std::get<0>(o.betas()),numeric(p,c))); break;
   case Setting::eps:     o.eps(numeric(p,c)); break;
   case Setting::decay:   o.weight_decay(numeric(p,c)); break;
   case Setting::amsgrad: o.amsgrad(flag(p,c)); break;
   default: opair(c,p); break;
  }
}

template<typename O> static K adam(bool a,const O& o) {
 //return all or non-default options as k dictionary
 K x=KDICT; const O d; OPTSET(x, lr, kf(o.lr()));
 if(a || std::get<0>(d.betas()) != std::get<0>(o.betas())) OPTSET(x, beta1,   kf(std::get<0>(o.betas())));
 if(a || std::get<1>(d.betas()) != std::get<1>(o.betas())) OPTSET(x, beta2,   kf(std::get<1>(o.betas())));
 if(a || d.eps()          != o.eps())                      OPTSET(x, eps,     kf(o.eps()));
 if(a || d.weight_decay() != o.weight_decay())             OPTSET(x, decay,   kf(o.weight_decay()));
 if(a || d.amsgrad()      != o.amsgrad())                  OPTSET(x, amsgrad, kb(o.amsgrad()));
 return resolvedict(x);
}

template<typename S> static K adamget(const S& s) { //template for adam/adamw
 K x=KDICT;
 dictadd(x, "step",           kj(s.step()));
 dictadd(x, "exp_avg",        kget(s.exp_avg()));
 dictadd(x, "exp_avg_sq",     kget(s.exp_avg_sq()));
 dictadd(x, "max_exp_avg_sq", kget(s.max_exp_avg_sq()));
 return x;
}

//template<typename S>static void adamput(K x,const Device& d,const std::string& k,Optimizer& o) {
template<typename S>static void adamput(K x,const Device& d,void *k,Optimizer& o) {
 K v; auto s=std::make_unique<S>();
 if((v=findbuffer(x,"step",-KJ)))       s->step(v->j);
 if((v=findbuffer(x,"exp_avg")))        s->exp_avg(kput(v).to(d));
 if((v=findbuffer(x,"exp_avg_sq")))     s->exp_avg_sq(kput(v).to(d));
 if((v=findbuffer(x,"max_exp_avg_sq"))) s->max_exp_avg_sq(kput(v).to(d));
 //o.state()[k]=std::move(s);
}

template<typename S> static J adamsize(Attr a,const S& s) {
 //count of tensors/elements/bytes in parameter buffers
 switch(a) {
  case Attr::tensorcount: return     oten(s.step()) +     oten(s.exp_avg()) +     oten(s.exp_avg_sq()) +     oten(s.max_exp_avg_sq());
  case Attr::elements:    return   objnum(s.step()) +   objnum(s.exp_avg()) +   objnum(s.exp_avg_sq()) +   objnum(s.max_exp_avg_sq());
  case Attr::bytes:       return objbytes(s.step()) + objbytes(s.exp_avg()) + objbytes(s.exp_avg_sq()) + objbytes(s.max_exp_avg_sq());
  default: TORCH_ERROR("adam/adamw: unexpected attribute for counting buffer sizes");
 }
}

// ----------------------------------------------------------------------------------------
// lamb - set/get options for lamb optimizer
// lambget - retrieve parameter buffers from lamb optimizer into k dictionary
// lambput - given k dictionary of buffers, put values into lamb optimizer state
// lambsize - tensor count, elements or bytes in parameter buffers
// ----------------------------------------------------------------------------------------
static void lamb(K x,J i,Cast c,ParamGroup& g) {
 auto& o=getoptions<LambOptions>(g); Pairs p; J n=xargc(x,i,p);
 for(J j=0;j<n;++j)
  switch(j) {
   case 0: o.lr(numeric(x,i+j,c,Setting::lr)); break;
   case 1: o.betas(std::make_tuple(numeric(x,i+j,c,Setting::beta1),std::get<1>(o.betas()))); break;
   case 2: o.betas(std::make_tuple(std::get<0>(o.betas()),numeric(x,i+j,c,Setting::beta2))); break;
   case 3: o.eps(numeric(x,i+j,c,Setting::eps)); break;
   case 4: o.weight_decay(numeric(x,i+j,c,Setting::decay)); break;
   case 5: o.unbiased(flag(x,i+j,c,Setting::unbiased)); break;
   case 6: o.globalnorm(flag(x,i+j,c,Setting::globalnorm)); break;
   case 7: o.trustclip(flag(x,i+j,c,Setting::trustclip)); break;
   case 8: o.trustmin(numeric(x,i+j,c,Setting::trustmin)); break;
   case 9: o.trustmax(numeric(x,i+j,c,Setting::trustmax)); break;
   default: opos(x,c,i+j); break;
  }
 while(xpair(p))
  switch(oset(p.k)) {
   case Setting::lr:         o.lr(numeric(p,c)); break;
   case Setting::beta1:      o.betas(std::make_tuple(numeric(p,c),std::get<1>(o.betas()))); break;
   case Setting::beta2:      o.betas(std::make_tuple(std::get<0>(o.betas()),numeric(p,c))); break;
   case Setting::eps:        o.eps(numeric(p,c)); break;
   case Setting::decay:      o.weight_decay(numeric(p,c)); break;
   case Setting::unbiased:   o.unbiased(flag(p,c)); break;
   case Setting::globalnorm: o.globalnorm(flag(p,c)); break;
   case Setting::trustclip:  o.trustclip(flag(p,c)); break;
   case Setting::trustmin:   o.trustmin(numeric(p,c)); break;
   case Setting::trustmax:   o.trustmax(numeric(p,c)); break;
   default: opair(c,p); break;
  }
}

static K lamb(bool a,const LambOptions& o) {
 //return all or non-default options as k dictionary
 K x=KDICT; const LambOptions d; OPTSET(x, lr, kf(o.lr()));
 if(a || std::get<0>(d.betas()) != std::get<0>(o.betas())) OPTSET(x, beta1,      kf(std::get<0>(o.betas())));
 if(a || std::get<1>(d.betas()) != std::get<1>(o.betas())) OPTSET(x, beta2,      kf(std::get<1>(o.betas())));
 if(a || d.eps()          != o.eps())                      OPTSET(x, eps,        kf(o.eps()));
 if(a || d.weight_decay() != o.weight_decay())             OPTSET(x, decay,      kf(o.weight_decay()));
 if(a || d.unbiased()     != o.unbiased())                 OPTSET(x, unbiased,   kb(o.unbiased()));
 if(a || d.globalnorm()   != o.globalnorm())               OPTSET(x, globalnorm, kb(o.globalnorm()));
 if(a || d.trustclip()    != o.trustclip())                OPTSET(x, trustclip,  kb(o.trustclip()));
 if(a || d.trustmin()     != o.trustmin())                 OPTSET(x, trustmin,   kf(o.trustmin()));
 if(a || d.trustmax()     != o.trustmax())                 OPTSET(x, trustmax,   kf(o.trustmax()));
 return resolvedict(x);
}

static K lambget(const LambParamState& s) {
 K x=KDICT;
 dictadd(x, "step",           kj(s.step()));
 dictadd(x, "exp_avg",        kget(s.exp_avg()));
 dictadd(x, "exp_avg_sq",     kget(s.exp_avg_sq()));
 return x;
}

//static void lambput(K x,const Device& d,const std::string& k,Optimizer& o) {
static void lambput(K x,const Device& d,void *k,Optimizer& o) {
 K v; auto s=std::make_unique<LambParamState>();
 if((v=findbuffer(x,"step",-KJ)))   s->step(v->j);
 if((v=findbuffer(x,"exp_avg")))    s->exp_avg(kput(v).to(d));
 if((v=findbuffer(x,"exp_avg_sq"))) s->exp_avg_sq(kput(v).to(d));
 o.state()[k]=std::move(s);
}

static J lambsize(Attr a,const LambParamState& s) {
 //count of tensors/elements/bytes in parameter buffers
 switch(a) {
  case Attr::tensorcount: return     oten(s.step()) +     oten(s.exp_avg()) +     oten(s.exp_avg_sq());
  case Attr::elements:    return   objnum(s.step()) +   objnum(s.exp_avg()) +   objnum(s.exp_avg_sq());
  case Attr::bytes:       return objbytes(s.step()) + objbytes(s.exp_avg()) + objbytes(s.exp_avg_sq());
  default: TORCH_ERROR("lamb: unexpected attribute for counting buffer sizes");
 }
}

// --------------------------------------------------------------------------------
// lbfgs - set/get options for lbfgs optimizer
// lbget - retrieve parameter buffers from lbfgs optimizer into k dictionary
// lbput - given k dictionary of buffers, put values into lbfgs optimizer state
// lbsize - tensor count, elements or bytes in parameter buffers
// --------------------------------------------------------------------------------
static void lbfgs(K x,J i,Cast c,ParamGroup& g) {
 auto& o=getoptions<LBFGSOptions>(g); Pairs p; J n=xargc(x,i,p);
 for(J j=0;j<n;++j)
  switch(j) {
   case 0: o.lr(numeric(x,i+j,c,Setting::lr)); break;
   case 1: o.max_iter(int64(x,i+j,c,Setting::iter)); break;
   case 2: o.max_eval(int64n(x,i+j,c,Setting::eval)); break;
   case 3: o.tolerance_grad(numeric(x,i+j,c,Setting::gradtol)); break;
   case 4: o.tolerance_change(numeric(x,i+j,c,Setting::changetol)); break;
   case 5: o.history_size(int64(x,i+j,c,Setting::history)); break;
   case 6: o.line_search_fn(code(x,i+j,c,Setting::search)); break;
   default: opos(x,c,i+j); break;
  }
 while(xpair(p))
  switch(oset(p.k)) {
   case Setting::lr:        o.lr(numeric(p,c)); break;
   case Setting::iter:      o.max_iter(int64(p,c)); break;
   case Setting::eval:      o.max_eval(int64n(p,c)); break;
   case Setting::gradtol:   o.tolerance_grad(numeric(p,c)); break;
   case Setting::changetol: o.tolerance_change(numeric(p,c)); break;
   case Setting::history:   o.history_size(int64(p,c)); break;
   case Setting::search:    o.line_search_fn(code(p,c)); break;
   default: opair(c,p); break;
  }
 if(!o.max_eval()) o.max_eval((o.max_iter()*5)/4);
}

static K lbfgs(bool a,const LBFGSOptions& o) {
 //return all or non-default options as k dictionary
 K x=KDICT; LBFGSOptions d; OPTSET(x, lr, kf(o.lr()));
 if(a || d.max_iter()         != o.max_iter())         OPTSET(x, iter,      kj(o.max_iter()));
 if(a || o.max_eval())                                 OPTSET(x, eval,      kj(o.max_eval() ? *o.max_eval() : nj));
 if(a || d.tolerance_grad()   != o.tolerance_grad())   OPTSET(x, gradtol,   kf(o.tolerance_grad()));
 if(a || d.tolerance_change() != o.tolerance_change()) OPTSET(x, changetol, kf(o.tolerance_change()));
 if(a || d.history_size()     != o.history_size())     OPTSET(x, history,   kj(o.history_size()));
 if(o.line_search_fn().has_value())                    OPTSET(x, search,    ks(cs(o.line_search_fn().value().c_str())));
 return resolvedict(x);
}

static K lbget(const LBFGSParamState& s) {
 K x=KDICT;
 dictadd(x, "func_evals",     kj(s.func_evals()));       // scalar long
 dictadd(x, "n_iter",         kj(s.n_iter()));           // scalar long
 dictadd(x, "t",              kf(s.t()));                // scalar long
 dictadd(x, "prev_loss",      kf(s.prev_loss()));        // scalar double
 dictadd(x, "d",              kget(s.d()));              // tensor
 dictadd(x, "h_diag",         kget(s.H_diag()));         // tensor
 dictadd(x, "prev_flag_grad", kget(s.prev_flat_grad())); // tensor
 dictadd(x, "old_dirs",       kget(s.old_dirs()));       // deque
 dictadd(x, "old_stps",       kget(s.old_stps()));       // deque
 dictadd(x, "ro",             kget(s.ro()));             // deque
 if(s.al().has_value())                                  // optional vector of tensors
  dictadd(x, "al", kget(s.al().value()));
 return x;
}

//static void lbput(K x,const Device& d,const std::string& k,Optimizer& o) {
static void lbput(K x,const Device& d,void *k,Optimizer& o) {
 K v; auto s=std::make_unique<LBFGSParamState>();
 if((v=findbuffer(x,"func_evals",-KJ)))   s->func_evals(v->j);
 if((v=findbuffer(x,"n_iter",-KJ)))       s->n_iter(v->j);
 if((v=findbuffer(x,"t",-KF)))            s->t(v->f);
 if((v=findbuffer(x,"prev_loss",-KF)))    s->prev_loss(v->f);
 if((v=findbuffer(x,"d")))                s->d(kput(v).to(d));
 if((v=findbuffer(x,"H_diag")))           s->H_diag(kput(v).to(d));
 if((v=findbuffer(x,"prev_flat_grad")))   s->prev_flat_grad(kput(v).to(d));
 if((v=findbuffer(x,"old_dirs")))         s->old_dirs(deque(v,"old_dirs",d));
 if((v=findbuffer(x,"old_stps")))         s->old_stps(deque(v,"old_stps",d));
 if((v=findbuffer(x,"ro")))               s->ro(deque(v,"ro",d));
 if((v=findbuffer(x,"al")) && !xempty(v)) {auto w=vec(v); to(w,d,true); s->al(w);}
 o.state()[k]=std::move(s);
}

static J lbsize(Attr a,const LBFGSParamState& s) {
 //count of tensors/elements/bytes in parm buffers
 switch(a) {
  case Attr::tensorcount:
   return
    oten(s.func_evals()) +   oten(s.n_iter())   +   oten(s.t()) + oten(s.prev_loss()) +    // scalars
    oten(s.d())          +   oten(s.H_diag())   +   oten(s.prev_flat_grad()) +             // tensors
    oten(s.old_dirs())   +   oten(s.old_stps()) +   oten(s.ro()) +                         // deques
    oten(s.al());                                                       // optional vector of tensors
  case Attr::elements:
   return
    objnum(s.func_evals()) +   objnum(s.n_iter())   +   objnum(s.t()) + objnum(s.prev_loss()) +    // scalars
    objnum(s.d())          +   objnum(s.H_diag())   +   objnum(s.prev_flat_grad()) +               // tensors
    objnum(s.old_dirs())   +   objnum(s.old_stps()) +   objnum(s.ro()) +                           // deques
    objnum(s.al());                                                             // optional vector of tensors
  case Attr::bytes:
   return 
    objbytes(s.func_evals()) + objbytes(s.n_iter())   + objbytes(s.t()) + objbytes(s.prev_loss()) +  // scalars
    objbytes(s.d())          + objbytes(s.H_diag())   + objbytes(s.prev_flat_grad()) +               // tensors
    objbytes(s.old_dirs())   + objbytes(s.old_stps()) + objbytes(s.ro()) +                           // deques
    objbytes(s.al());                                                             // optional vector of tensors
  default: TORCH_ERROR("lbfgs: unexpected attribute for counting buffer sizes");
 }
}

// --------------------------------------------------------------------------------
// rmsprop - set/get options for rmsprop optimizer
// rmsget - retrieve parameter buffers from rmsprop optimizer into k dictionary
// rmsput - given k dictionary of buffers, put values into rmsprop optimizer state
// rmssize - tensor count, elements or bytes in parameter buffers
// --------------------------------------------------------------------------------
static void rmsprop(K x,J i,Cast c,ParamGroup& g) {
 auto& o=getoptions<RMSpropOptions>(g); Pairs p; J n=xargc(x,i,p);
 for(J j=0;j<n;++j)
  switch(j) {
   case 0: o.lr(numeric(x,i+j,c,Setting::lr)); break;
   case 1: o.alpha(numeric(x,i+j,c,Setting::alpha)); break;
   case 2: o.eps(numeric(x,i+j,c,Setting::eps)); break;
   case 3: o.weight_decay(numeric(x,i+j,c,Setting::decay)); break;
   case 4: o.momentum(numeric(x,i+j,c,Setting::momentum)); break;
   case 5: o.centered(flag(x,i+j,c,Setting::centered)); break;
   default: opos(x,c,i+j); break;
  }
 while(xpair(p))
  switch(oset(p.k)) {
   case Setting::lr:        o.lr(numeric(p,c)); break;
   case Setting::alpha:     o.alpha(numeric(p,c)); break;
   case Setting::eps:       o.eps(numeric(p,c)); break;
   case Setting::decay:     o.weight_decay(numeric(p,c)); break;
   case Setting::momentum:  o.momentum(numeric(p,c)); break;
   case Setting::centered:  o.centered(flag(p,c)); break;
   default: opair(c,p); break;
  }
}

static K rmsprop(bool a,const RMSpropOptions& o) {
 //return all or non-default options as k dictionary
 K x=KDICT; RMSpropOptions d; OPTSET(x, lr, kf(o.lr()));
 if(a || d.alpha()        != o.alpha())        OPTSET(x, alpha,    kf(o.alpha()));
 if(a || d.eps()          != o.eps())          OPTSET(x, eps,      kf(o.eps()));
 if(a || d.weight_decay() != o.weight_decay()) OPTSET(x, decay,    kf(o.weight_decay()));
 if(a || d.momentum()     != o.momentum())     OPTSET(x, momentum, kf(o.momentum()));
 if(a || d.centered()     != o.centered())     OPTSET(x, centered, kb(o.centered()));
 return resolvedict(x);
}

static K rmsget(const RMSpropParamState& s) {
 K x=KDICT;
 dictadd(x, "step",       kj(s.step()));
 dictadd(x, "square_avg", kget(s.square_avg()));
 dictadd(x, "momentum",   kget(s.momentum_buffer()));
 dictadd(x, "grad_avg",   kget(s.grad_avg()));
 return x;
}

//static void rmsput(K x,const Device& d,const std::string& k,Optimizer& o) {
static void rmsput(K x,const Device& d,void *k,Optimizer& o) {
 K v; auto s=std::make_unique<RMSpropParamState>();
 if((v=findbuffer(x,"step",-KJ)))        s->step(v->j);
 if((v=findbuffer(x,"square_avg")))      s->square_avg(kput(v).to(d));
 if((v=findbuffer(x,"momentum_buffer"))) s->momentum_buffer(kput(v).to(d));
 if((v=findbuffer(x,"grad_avg")))        s->grad_avg(kput(v).to(d));
 o.state()[k]=std::move(s);
}

static J rmssize(Attr a,const RMSpropParamState& s) {
 //count of tensors/elements/bytes in parm buffers
 switch(a) {
  case Attr::tensorcount: return   oten(s.step())   +    oten(s.square_avg())  +     oten(s.momentum_buffer()) +     oten(s.grad_avg());
  case Attr::elements:    return objnum(s.step())   +  objnum(s.square_avg())  +   objnum(s.momentum_buffer()) +   objnum(s.grad_avg());
  case Attr::bytes:       return objbytes(s.step()) + objbytes(s.square_avg()) + objbytes(s.momentum_buffer()) + objbytes(s.grad_avg());
  default: TORCH_ERROR("rmsprop: unexpected attribute for counting buffer sizes");
 }
}

// ----------------------------------------------------------------------------
// sgd - set/get options for sgd optimizer
// sgdget - retrieve parameter buffers from sgd optimizer into k dictionary
// sgdput - given k dictionary of buffers, put values into sgd optimizer state
// sgdsize - tensor count, elements or bytes in parameter buffers
// ----------------------------------------------------------------------------
static void sgd(K x,J i,Cast c,ParamGroup& g) {
 auto& o=getoptions(g); Pairs p; J n=xargc(x,i,p); 
 for(J j=0;j<n;++j)
  switch(j) {
   case 0: o.lr(numeric(x,i+j,c,Setting::lr)); break;
   case 1: o.momentum(numeric(x,i+j,c,Setting::momentum)); break;
   case 2: o.dampening(numeric(x,i+j,c,Setting::dampening)); break;
   case 3: o.weight_decay(numeric(x,i+j,c,Setting::decay)); break;
   case 4: o.nesterov(flag(x,i+j,c,Setting::nesterov)); break;
   default: opos(x,c,i+j); break;
  }
 while(xpair(p))
  switch(oset(p.k)) {
   case Setting::lr:        o.lr(numeric(p,c)); break;
   case Setting::momentum:  o.momentum(numeric(p,c)); break;
   case Setting::dampening: o.dampening(numeric(p,c)); break;
   case Setting::decay:     o.weight_decay(numeric(p,c)); break;
   case Setting::nesterov:  o.nesterov(flag(p,c)); break;
   default: opair(c,p); break;
  }
}

static K sgd(bool a,const SGDOptions& o) {
 //return all or non-default options as k dictionary
 K x=KDICT; SGDOptions d(LR); OPTSET(x, lr, kf(o.lr()));
 if(a || d.momentum()     != o.momentum())     OPTSET(x, momentum,  kf(o.momentum()));
 if(a || d.dampening()    != o.dampening())    OPTSET(x, dampening, kf(o.dampening()));
 if(a || d.weight_decay() != o.weight_decay()) OPTSET(x, decay,     kf(o.weight_decay()));
 if(a || d.nesterov()     != o.nesterov())     OPTSET(x, nesterov,  kb(o.nesterov()));
 return resolvedict(x);
}

static K sgdget(const SGDParamState& s) {
 K x=KDICT;
 dictadd(x, "momentum_buffer",  kget(s.momentum_buffer()));
 return x;
}

//static void sgdput(K x,const Device& d,const std::string& k,Optimizer& o) {
static void sgdput(K x,const Device& d,void *k,Optimizer& o) {
 K v; auto s=std::make_unique<SGDParamState>();
 if((v=findbuffer(x,"momentum_buffer"))) s->momentum_buffer(kput(v).to(d));
 o.state()[k]=std::move(s);
}

static J sgdsize(Attr a, const SGDParamState& s) {
 //count of tensors/elements/bytes in parm buffers
 switch(a) {
  case Attr::tensorcount: return     oten(s.momentum_buffer());
  case Attr::elements:    return   objnum(s.momentum_buffer());
  case Attr::bytes:       return objbytes(s.momentum_buffer());
  default: TORCH_ERROR("sgd: unexpected attribute for counting buffer sizes");
 }
}

// ---------------------------------------------------------------------------
// optimizer settings are handled as a dictionary per optimizer group
// lists of dictionaries are resolved to a table; the code gets elaborate
// to accomodate the option of maintaining only the non-default settings:
// ---------------------------------------------------------------------------
// findsym - given symbol, returns index in k list of symols if found, else -1
// checkgroup - return true if setting found in any group's dict of settings
// setting1 - assign group value to row & col in settings table (general list)
// setting2 - assign group value to row & col in settings table (simple list)
// tablecol - given setting symbol, default value and group dictionaries
//            find setting in each group and populate table column
// ---------------------------------------------------------------------------
static J findsym(S s,K x) {
 for(J i=0; i<x->n; ++i) if(s==kS(x)[i]) return i;
 return -1;
}

bool checkgroup(S s,K x) {
 for(J i=0; i<x->n; ++i) if(findsym(s, kK(kK(x)[i])[0])>-1) return true;
 return false;
}
 
static void setting1(Cast c,S s,K v,J i,K x) {
 switch(v->t) {
  case KS: TORCH_CHECK(x->t==-KS, omap(c),": group[",i,"] setting for '",s,"' expects symbol, given ",kname(x));  kS(v)[i]=x->s; break;
  case KB: TORCH_CHECK(x->t==-KB, omap(c),": group[",i,"] setting for '",s,"' expects boolean, given ",kname(x)); kG(v)[i]=x->g; break;
  case KJ: TORCH_CHECK(x->t==-KJ, omap(c),": group[",i,"] setting for '",s,"' expects long, given ",kname(x));    kJ(v)[i]=x->j; break;
  case KF: 
   TORCH_CHECK(x->t==-KF || x->t==-KJ, omap(c),": group[",i,"] setting for '",s,"' expects double, given ",kname(x));
   kF(v)[i]= x->t==-KF ? x->f : (F)x->j;
   break;
  default: TORCH_ERROR(omap(c),": unable to define setting for '",s,", ",kname(v)," unexpected"); break;
 }
}

static void setting2(Cast c,S s,K v,J i,K g,J j) {
 switch(v->t) {
  case KS: TORCH_CHECK(g->t==KS, omap(c),": group[",i,"] setting for '",s,"' expects symbol, given ",kname(g));  kS(v)[i]=kS(g)[j]; break;
  case KB: TORCH_CHECK(g->t==KB, omap(c),": group[",i,"] setting for '",s,"' expects boolean, given ",kname(g)); kG(v)[i]=kG(g)[j]; break;
  case KJ: TORCH_CHECK(g->t==KJ, omap(c),": group[",i,"] setting for '",s,"' expects long, given ",kname(g));    kJ(v)[i]=kJ(g)[j]; break;
  case KF: 
   TORCH_CHECK(g->t==KF || g->t==KJ, omap(c),": group[",i,"] setting for '",s,"' expects double, given ",kname(g));
   kF(v)[i]= g->t==KF ? kF(g)[j] : (F)kJ(g)[j];
   break;
  default: TORCH_ERROR(omap(c),": unable to define setting for '",s,", ",kname(v)," unexpected"); break;
 }
}

static K tablecol(Cast c,S s,K x,K y) {
 TORCH_CHECK(x->t<0, omap(c),": unable define default setting for '",s,"' using ",kname(x));
 K v=ktn(-x->t,y->n);
 for(J i=0; i<y->n;++i) {
  K z=kK(y)[i], k=kK(z)[0], g=kK(z)[1];
  J j=findsym(s,k);
  if(j<0)       setting1(c,s,v,i,x);         // no setting defined for group, use default
  else if(g->t) setting2(c,s,v,i,g,j);       // group settings are simple list
  else          setting1(c,s,v,i,kK(g)[j]);  // group settings are general list
 }
 return v;
}
 
// -------------------------------------------------------------------------------------
// optdefaults - return default options for single optimizer or table for all
// optsetting - return dictionary of options in parameter group of given optimizer type
// optsettings - return table of settings, one row per optimizer group
// buffersize - count tensors, elements or bytes of optimizer buffers for each parameter
// -------------------------------------------------------------------------------------
K optdefaults(Cast c) {
 switch(c) {
  case Cast::adagrad: return adagrad(true,AdagradOptions());
  case Cast::adam:    return adam(true,AdamOptions());
  case Cast::adamw:   return adam(true,AdamWOptions());
  case Cast::lamb:    return lamb(true,LambOptions());
  case Cast::lbfgs:   return lbfgs(true,LBFGSOptions().line_search_fn("strong_wolf"));
  case Cast::rmsprop: return rmsprop(true,RMSpropOptions());
  case Cast::sgd:     return sgd(true,SGDOptions(LR));

  case Cast::undefined: {
   const auto& e=env().opt; J i=0,n=e.size();
   K k=ktn(KS,3),s=ktn(KS,n),d=ktn(0,n),o=ktn(0,n);
   kS(k)[0]=cs("optimizer"); kS(k)[1]=cs("pytorch"); kS(k)[2]=cs("options");
   for(const auto& a:e) {
    kS(s)[i]=std::get<0>(a);
    kK(d)[i]=kp((S)std::get<2>(a).c_str());
    kK(o)[i]=optdefaults(std::get<1>(a)); ++i;
   }
   return xT(xD(k,knk(3,s,d,o)));
  }
  default: TORCH_ERROR("no help implemented for optimizer enumeration: ",(I)c);
 }
}

static K optsetting(bool a,Cast c,const Options& o) {
 switch(c) {
  case Cast::adagrad: return adagrad(a, static_cast<const AdagradOptions&>(o));
  case Cast::adam:    return adam(a,    static_cast<const AdamOptions&>   (o));
  case Cast::adamw:   return adam(a,    static_cast<const AdamWOptions&>  (o));
  case Cast::lamb:    return lamb(a,    static_cast<const LambOptions&>   (o));
  case Cast::lbfgs:   return lbfgs(a,   static_cast<const LBFGSOptions&>  (o));
  case Cast::rmsprop: return rmsprop(a, static_cast<const RMSpropOptions&>(o));
  case Cast::sgd:     return sgd(a,     static_cast<const SGDOptions&>    (o));
  default: TORCH_ERROR("Unrecognized optimizer: ",(I)c);
 }
}

static K maketable(Cast c,K x) {
 K o=optdefaults(c), s=kK(o)[0], d=kK(o)[1]; std::vector<J> j;
 for(J i=0; i<s->n; ++i)
  if(checkgroup(kS(s)[i],x)) j.push_back(i);
 K k=ktn(KS,j.size()), v=ktn(0,j.size());
 for(J i=0; i<k->n; i++)
  kS(k)[i]=kS(s)[j[i]],
  kK(v)[i]=tablecol(c, kS(k)[i], kK(d)[j[i]], x);
 r0(o);
 return xT(xD(k,v));
}
 
K optsettings(bool a,Cast c,const Optimizer& o) {
 size_t i=0,n=o.param_groups().size(); K d=ktn(0,n);
 for(const auto&g:o.param_groups())
  kK(d)[i++]=optsetting(a,c,g.options()); // build list of dictionaries
 K t=maketable(c,d); r0(d);               // convert list to table
 return t;
}

KAPI settingstest(K x,K y) {
 KTRY
  S s;
  TORCH_CHECK(xsym(x,s), "1st arg of symbol");
  TORCH_CHECK(!y->t && y->n, "2nd arg is non-empty list of dictionaries");
  for(J i=0;i<y->n;++i)
   TORCH_CHECK(xdict(kK(y)[i]), "element[",i,"] is not a dictionary");
  Cast c=omap(s);
  return maketable(c,y);
 KCATCH("settings test");
}

static J buffersize(Attr a,Cast c,const ParamState& p) {
 switch(c) {
  case Cast::adagrad: return   adasize(a, static_cast<const AdagradParamState&>(p));
  case Cast::adam:    return  adamsize(a, static_cast<const AdamParamState&>(p));
  case Cast::adamw:   return  adamsize(a, static_cast<const AdamWParamState&>(p));
  case Cast::lamb:    return  lambsize(a, static_cast<const  LambParamState&>(p));
  case Cast::lbfgs:   return    lbsize(a, static_cast<const LBFGSParamState&>(p));
  case Cast::rmsprop: return   rmssize(a, static_cast<const RMSpropParamState&>(p));
  case Cast::sgd:     return   sgdsize(a, static_cast<const SGDParamState&>(p));
  default: TORCH_ERROR("unrecognized optimizer: ",(I)c,", unable to retrieve parameter state");
 }
}

J buffersize(Attr a,Cast c,const Optimizer& o) {
 J n=0;
 for(const auto& p:o.state())
  n+=buffersize(a,c,*p.second);
 return n;
}

// --------------------------------------------------------------------------------------
// parmkeys - return columns for table describing optimizer parameter groups & buffers
// moduletype - given parameter, attempt to find parent module type
// parmname - given parameter, search containing module(s), return name if found
// parmsym - return string from parmname as symbol
// --------------------------------------------------------------------------------------
static K parmkeys(bool b) {
 K x=ktn(KS, b ? 6 : 5);
 kS(x)[0]=statekey(State::parmgroup);
 kS(x)[1]=statekey(State::pointer);
 kS(x)[2]=statekey(State::module);
 kS(x)[3]=statekey(State::name);
 kS(x)[4]=statekey(State::size);
 if(b) kS(x)[5]=statekey(State::buffers);
 return x;
}

static S moduletype(const Tensor& p,const Module& m) {
 for(const auto& a:m.modules(true))
  for(const auto& t:a->parameters(false))
   if(t.is_same(p)) return msym(*a);
 return nullsym();
}

static std::string parmname(const Tensor& p,const Module& m) {
 for(auto& a:m.named_parameters())
  if(a.value().is_same(p))
   return a.key();
 return {};
}

static S parmsym(const Tensor& p,const Module& m) {
 auto s=parmname(p,m);
 return s.size() ? cs(s.c_str()) : env().nullsym;
}

// --------------------------------------------------------------------------------------
// getparms - given optimizer type and parameter state, return buffers as k dictonary
//            also, get size, attempt to find name and type of containing module
// --------------------------------------------------------------------------------------
static K getparms(Cast c,const ParamState& p) {
 switch(c) {
  case Cast::adagrad: return   adaget(static_cast<const AdagradParamState&>(p));
  case Cast::adam:    return  adamget(static_cast<const AdamParamState&>(p));
  case Cast::adamw:   return  adamget(static_cast<const AdamWParamState&>(p));
  case Cast::lamb:    return  lambget(static_cast<const LambParamState&>(p));
  case Cast::lbfgs:   return    lbget(static_cast<const LBFGSParamState&>(p));
  case Cast::rmsprop: return   rmsget(static_cast<const RMSpropParamState&>(p));
  case Cast::sgd:     return   sgdget(static_cast<const SGDParamState&>(p));
  default: TORCH_ERROR("unrecognized optimizer: ",(I)c,", unable to retrieve parameter state");
 }
}

static K getparms(bool b,Cast c,const Optimizer& o,const Module& m) {
 J g=0,i=0,n=osize(o);
 K pt=ktn(KJ,n),gp=ktn(KJ,n),md=ktn(KS,n),nm=ktn(KS,n),sz=ktn(0,n),bf=nullptr; if(b) bf=ktn(0,n);
 const auto& s=o.state();
 for(const auto& pg:o.param_groups()) {
  for(const auto& p:pg.params()) {
    auto *t=p.unsafeGetTensorImpl();
    kJ(gp)[i]=g;
    kJ(pt)[i]=(intptr_t)t;
    kS(md)[i]=moduletype(p,m);
    kS(nm)[i]=parmsym(p,m);
    kK(sz)[i]=tensorsize(p,Attr::size);
    if(b) {
     //auto k=c10::guts::to_string(t);
     //kK(bf)[i]=s.count(k) ? getparms(c, *s.at(k)) : KDICT;
     kK(bf)[i]=s.count(t) ? getparms(c, *s.at(t)) : KDICT;
    }
    i++;
  }
  g++;
 }
 return xT(xD(parmkeys(b),b ? knk(6,gp,pt,md,nm,sz,bf) : knk(5,gp,pt,md,nm,sz)));
}

// ---------------------------------------------------------------------------------------
// dupname - check for duplicate names in container module before adding
// addmodule - add child module if not already registered in target module
// dictfind - return parameter dictionary module if exists or is top-level child
// ---------------------------------------------------------------------------------------
static void dupname(S s,const Module& m) {
 TORCH_CHECK(!m.named_children().contains(s),
  "opt: a ",msym(*m.named_children()[s])," module named `",s," already registered with the optimizer");
}

static void addmodule(const Moduleptr& a,Moduleptr& m) {
 if(m) {
  for(const auto& c:m->modules()) if(c.get() == a.get()) return;   // module already added
  S s=mname(*a); 
  if(auto* d=m->as<nn::ModuleDict>()) {
   if(s) dupname(s,*d);
   d->update({{s ? s : c10::to_string(d->children().size()), a}}); // update to include new module
  } else {                                                         // else create dictionary
   S r=mname(*m);                                                  // w'union of existing & new module
   nn::ModuleDict u(Modulemap{{r ? r : "0", m}});                  // create dict with existing module
   if(s && r) dupname(s,*u);                                       // check for name conflict
   u->update(Modulemap{{s ? s : "1", a}});                         // add new module to dictionary container
   m=std::move(u.ptr());
  }
 } else {
  m=std::move(a);
 }
}

static nn::ParameterDictImpl* dictfind(Moduleptr& m) {
 nn::ParameterDictImpl *p=nullptr;
 if((p=m->as<nn::ParameterDict>())) {           // module is a parameter dictionary
 } else if(auto *d=m->as<nn::ModuleDict>()) {   // search module dictionary children
   for(const auto& c:d->children())
    if((p=c->as<nn::ParameterDict>())) break;
 }
 return p;
}
 
// ---------------------------------------------------------------------------------------
// addname - name module "parms", if already found, try "parms1", "parms2", ..
// addtensor - add vector/dictionary of tensors to parameter dictionary in target module
// addvector - add list of tensors to parameter dictionary in target module
// adddict - add names & tensors to dictionary in target module
// ---------------------------------------------------------------------------------------
static void addname(Module& a,const Moduleptr& m) {
 std::string s1("parms");
 if(m) {
  size_t n=1; std::string s2;
  while(m->named_children().contains(s1+s2))
   s2=c10::to_string(n++);
  s1+=s2;
 }
 mname_(a)=s1;
}

static void addtensor(const TensorVector& v,nn::ParameterDictImpl *p) {
 for(const auto&t:v) p->insert(c10::to_string(p->size()),t);
}

static void addtensor(const TensorDict& d,nn::ParameterDictImpl *p) {
 for(const auto&a:d) p->insert(a.key(),a.value());
}

static void addvector(const TensorVector& v,Moduleptr& m) {
 nn::ParameterDictImpl *p;
 if(m && (p=dictfind(m))) {
  addtensor(v,p);
 } else {
  nn::ParameterDict d; addname(*d,m); addtensor(v,d.get()); addmodule(d.ptr(),m);
 }
}

static void adddict(const TensorDict& d,Moduleptr& m) {
 nn::ParameterDictImpl *p;
 if(m && (p=dictfind(m))) {
  addtensor(d,p);
 } else {
  nn::ParameterDict a; addname(*a,m); addtensor(d,a.get()); addmodule(a.ptr(),m);
 }
}

static void adddict(const TensorDict& d,J n,S *s,Moduleptr& m) {
 TensorDict a;
 for(J i=0; i<n; ++i) a.insert(s[i],d[s[i]]);
 adddict(a,m);
}

// -------------------------------------------------------------------------------------------
// duplicate - given vector or dictionary of tensors, check for duplicates, return vector
//           - also, boolean form, return true if tensor is duplicate 
// parmerror - signal specified parm already in optimizer group, attempt to get name, etc.
// parmcheck - check if each tensor in vector already defined in optimizer parameter group(s) 
// -------------------------------------------------------------------------------------------
static TensorVector duplicate(const TensorVector& v) {
 for(size_t i=0; i<v.size(); ++i) {
  const auto& t=v[i];
  for(size_t j=i+1; j<v.size(); ++j)
   if(t.is_same(v[j]))
    TORCH_ERROR("opt: parameter[",j,"] is duplicate of parameter[",i,"]");
 }
 return v;
}

static TensorVector duplicate(const TensorDict& d) {
 const auto& k=d.keys();
 for(size_t i=0; i<d.size(); ++i) {
  const auto& t=d[k[i]];
  for(size_t j=i+1; j<d.size(); ++j)
   if(t.is_same(d[k[j]]))
    TORCH_ERROR("opt: parameter[`",k[j],"] is duplicate of parameter[`",k[i],"]");
 }
 return d.values();
}

static bool duplicate(const TensorVector& v,const Tensor& p) {
 for(const auto& t:v)
  if(t.is_same(p)) return true;
 return false;
}

static void parmerror(const Module& m,const Tensor& p,size_t i,size_t g) {
 std::string s1=moduletype(p,m), s2=parmname(p,m);
 if(s1.size() || s2.size())
  TORCH_ERROR("opt: parameter[",i,"] already in group ",g, " (",s1," module parameter `",s2,")");
 else
  TORCH_ERROR("opt: parameter[",i,"] already in group ", g);
}

static void parmcheck(const Optimizer& o,const Module& m,const TensorVector& v) {
 size_t i=0;
 for(const auto& p:v) {
  TORCH_CHECK(p.is_leaf(), "opt: parameter[",i,"] is a not a leaf tensor");
  J j=0;
  for(const auto& g:o.param_groups()) {
   if(duplicate(g.params(),p))
    parmerror(m,p,i,j);
   j++;
  }
  i++;
 }
}

// ------------------------------------------------------------------------------------------
// moduleparms - return parm vector from module w'optional child indices or module/parm names
// ------------------------------------------------------------------------------------------
static TensorVector moduleparms(const Moduleptr& m,J n,J *j) {
 TensorVector v;
 const auto& a=m->modules(false);
 for(J i=0;i<n;++i) {
  J k=j[i];
  TORCH_CHECK(k>=0, "opt: module[",k,"] invalid");
  TORCH_CHECK(k<(J)a.size(), "opt: module[",k,"] out of bounds, ",a.size()," submodule(s)");
  const auto& p=a.at(k)->parameters();
  for(const auto& t:p)
   TORCH_CHECK(!duplicate(v,t), "opt: duplicate parameter(s) with module[",k,"]");
  v.insert(v.end(), p.begin(), p.end());
 }
 return v;
}

static TensorVector moduleparms(const Moduleptr& m,J n,S *s) {
 TensorVector v;
 const auto& a=m->named_modules("",false);
 const auto& z=m->named_parameters(true);
 for(J i=0;i<n;++i) {
  S k=s[i];
  if(a.contains(k)) {
   const auto& p=a[k]->parameters();
   for(const auto& t:p)
    TORCH_CHECK(!duplicate(v,t), "opt: duplicate parameter(s) with module[`",k,"]");
   v.insert(v.end(), p.begin(), p.end());
  } else if(z.contains(k)) {
   const auto& p=z[k];
   TORCH_CHECK(!duplicate(v,p), "opt: duplicate parameter `",k);
   v.push_back(p);
  } else {
   TORCH_ERROR("opt: no module or parameter named `",k);
  }
 }
 return v;
}

static TensorVector moduleparms(const Moduleptr& a,K x,const Optimizer& o,Moduleptr& m) {
 TensorVector v;
 if(!x)
  v=a->parameters();
 else if(x->t==KJ || x->t==-KJ)
   v=x->t==KJ ? moduleparms(a,x->n,kJ(x)) : moduleparms(a,1,&x->j);
 else if(x->t==KS || x->t==-KS) 
   v=x->t==KS ? moduleparms(a,x->n,kS(x)) : moduleparms(a,1,&x->s);
 else
  TORCH_ERROR("opt: ",msym(*a)," module supplied with unrecognized ",kname(x)," selector(s)");
 parmcheck(o,*m,v); addmodule(a,m);
 return v;
}
 
// -------------------------------------------------------------
// dictparms - add parms from a dictionary w'optional parm keys
// -------------------------------------------------------------
static TensorVector dictparms(const TensorDict& d,J n,S *s) {
 TensorVector v;
 for(J i=0;i<n;++i) {
  S k=s[i];
  if(d.contains(k)) {
   const auto& t=d[k];
   TORCH_CHECK(!duplicate(v,t), "opt: duplicate parameter dict[`",k,"]");
   v.push_back(t);
  } else {
   TORCH_ERROR("opt: no dictionary parameter named `",k);
  }
 }
 return v;
}

static TensorVector dictparms(const TensorDict& d,K x,const Optimizer& o,Moduleptr& m) {
 TensorVector v;
 if(!x)
  v=duplicate(d);
 else if(x->t==KS || x->t==-KS) 
   v=x->t==KS ? dictparms(d,x->n,kS(x)) : dictparms(d,1,&x->s);
 else
  TORCH_ERROR("opt: tensor dictionary supplied with unrecognized ",kname(x)," selector(s)");
 parmcheck(o,*m,v);
 if(!x)            adddict(d,m);
 else if(x->t==KS) adddict(d,x->n,kS(x),m);
 else              adddict(d,1,&x->s,m);
 return v;
}
 
// -------------------------------------------------------------
// vectorparms - add parms from a vector w'optional indices
// -------------------------------------------------------------
static TensorVector vectorparms(const TensorVector& a,J n,J *j) {
 TensorVector v;
 for(J i=0;i<n;++i) {
  J k=j[i];
  TORCH_CHECK(k>=0, "opt: vector[",k,"] invalid");
  TORCH_CHECK(k<(J)a.size(), "opt: vector[",k,"] out of bounds, index must be less than ",a.size());
  const auto& t=a[k];
  TORCH_CHECK(!duplicate(v,t), "opt: duplicate parameter from vector[",k,"]");
  v.push_back(t);
 }
 return v;
}

static TensorVector vectorparms(const TensorVector& a,K x,const Optimizer& o,Moduleptr& m) {
 TensorVector v;
 if(!x)
  v=duplicate(a);
 else if(x->t==KJ || x->t==-KJ) 
  v=x->t==KJ ? vectorparms(a,x->n,kJ(x)) : vectorparms(a,1,&x->j);
 else
  TORCH_ERROR("opt: tensor vector supplied with unrecognized ",kname(x)," selector(s)");
 parmcheck(o,*m,v); addvector(v,m);
 return v;
}
 
// ----------------------------------------------------------------------------
// optget - return optimizer name & options, with or without internal buffers
// ----------------------------------------------------------------------------
K optget(bool a,bool b,Cast c,const Optimizer &o,const Module& m) {
 K k=ktn(KS,3),v=ktn(0,3);
 kS(k)[0]=statekey(State::optimizer); kK(v)[0]=ks(omap(c));
 kS(k)[1]=statekey(State::options);   kK(v)[1]=optsettings(a,c,o);
 kS(k)[2]=statekey(State::parms);     kK(v)[2]=getparms(b,c,o,m);
 return xD(k,v);
}

// ---------------------------------------------------------------------------------------
// addoptions - parse k args into optimizer-specific options stored in a parameter group
// addparms - return vector of parameters, checking for duplicates in existing groups
//            also updates module that tracks inputs to optimizer to recreate from state
// ---------------------------------------------------------------------------------------
static void addoptions(Cast c,K x,J i,ParamGroup& g) { // c:type, x-arg(s), i-offset into args
 switch(c) {
  case Cast::adagrad: adagrad(x,i,c,g); break;
  case Cast::adam:    adam<AdamOptions>(x,i,c,g); break;
  case Cast::adamw:   adam<AdamWOptions>(x,i,c,g); break;
  case Cast::lamb:    lamb(x,i,c,g); break;
  case Cast::lbfgs:   lbfgs(x,i,c,g); break;
  case Cast::rmsprop: rmsprop(x,i,c,g); break;
  case Cast::sgd:     sgd(x,i,c,g); break;
  default: TORCH_ERROR("unrecognized optimizer enumeration: ",(I)c);
 }
}

static TensorVector addparms(K x,const Optimizer& o,Moduleptr& m) {
 K y=nullptr; Ktag *k; TensorVector v;
 if(!xempty(x)) {
  if(!(k=xtag(x))) {
   k=xtag(x,0);
   TORCH_CHECK(k, "opt: expecting module, model or tensors after optimizer & optional group number, given ",kname(x));
   TORCH_CHECK(x->n==2,"opt: expecting one arg to select parameters from ",mapclass(k->a)," but given ",x->n-1," args");
   y=kK(x)[1];
  }
  switch(k->a) {
   case Class::tensor: v=vectorparms({k->tensor()}, y,o,m); break;
   case Class::vector: v=vectorparms(k->vector(), y,o,m); break;
   case Class::dict:   v=dictparms(k->dict(), y,o,m); break;
   case Class::module:
   case Class::model:  v=moduleparms(k->moduleptr(), y,o,m); break;
   default: TORCH_ERROR("opt: cannot derive parameters from ",mapclass(k->a));
  }
 }
 return v;
}

// --------------------------------------------------------------------------------
// optinit - create new optimizer given parameter and option arg(s)
// optedit - edit existing optimizer group or add new group of parameters & options
// --------------------------------------------------------------------------------
static Optptr optinit(Cast c,ParamGroup& g) {
 switch(c) {
  case Cast::adagrad: return std::make_shared<Adagrad>(ParamGroups{g},getoptions<AdagradOptions>(g));
  case Cast::adam:    return std::make_shared<Adam>   (ParamGroups{g},getoptions<AdamOptions>(g));
  case Cast::adamw:   return std::make_shared<AdamW>  (ParamGroups{g},getoptions<AdamWOptions>(g));
  case Cast::lamb:    return std::make_shared<Lamb>   (ParamGroups{g},getoptions<LambOptions>(g));
  case Cast::lbfgs:   return std::make_shared<LBFGS>  (ParamGroups{g},getoptions<LBFGSOptions>(g));
  case Cast::rmsprop: return std::make_shared<RMSprop>(ParamGroups{g},getoptions<RMSpropOptions>(g));
  case Cast::sgd:     return std::make_shared<SGD>    (ParamGroups{g},getoptions(g));
  default: TORCH_ERROR("opt: unrecognized optimizer enumeration: ",(I)c);
 }
}

static K optinit(bool b,Cast c,K x,K y) { //b:true if group given, x:overall args, y:parms
 Moduleptr m; ParamGroup g({}); addoptions(c,x,2+b,g);
 Optptr o=optinit(c,g);
 if(y) o->param_groups()[0].params()=addparms(y,*o,m);
 return kopt(c,o,m);
}

static void optedit(bool b,Cast c,K x,K y,J i,Optimizer& o,Moduleptr& m) {
 auto& p=o.param_groups(); J n=p.size();
 TORCH_CHECK(i>=0, "opt: group ",i," invalid, cannot be negative");
 TORCH_CHECK(i<=n, "opt: group ",i," invalid, cannot be greater than number of groups(",n,")");
 if(i==n) {                           // add new parameter group
  auto d=o.defaults().clone();        // get defaults created w'first group
  ParamGroup g({},std::move(d));      // initialize empty group
  addoptions(c,x,2+b,g);              // define options for the group
  if(y) g.params()=addparms(y,o,m);   // get parms for new group
  o.add_param_group(g);               // add to optimizer
 } else {
  auto g=p[i];                        // create a copy of i'th group
  addoptions(c,x,2+b,g);              // [re]set optimizer-specific options for the group
  if(y) {                             // add parameters to the end of the group's tensors
   const auto& v=addparms(y,o,m);
   g.params().insert(g.params().end(),v.begin(),v.end());
  }
  p[i].params()=g.params();              // use parameters from edited group
  p[i].set_options(g.options().clone()); // and updated options
 }
}

// ---------------------------------------------------------------------------------------
// grouprow - extract group settings from table row as a dictionary
// putgroups - given type & table of options, add empty group(s), return optimizer
// putbuffers - put buffers in k dict -> optimizer's state for one parameter
// putparms - assign optimizer parameters into group(s), add parameter buffers
// optput - given optimizer state and module, recreate optimizer (w'buffer state if given)
// ---------------------------------------------------------------------------------------
static K grouprow(K x,J r) {
 K s=kK(x->k)[0], c=kK(x->k)[1], k=ktn(KS,s->n), v=ktn(0,s->n);
 for(J i=0;i<s->n;++i) kS(k)[i]=kS(s)[i];
  for(J i=0;i<s->n;++i) {
  K y=kK(c)[i];
  switch(y->t) {
   case KB: kK(v)[i]=kb(kG(y)[r]); break;
   case KI: kK(v)[i]=ki(kI(y)[r]); break;
   case KJ: kK(v)[i]=kj(kJ(y)[r]); break;
   case KE: kK(v)[i]=ke(kE(y)[r]); break;
   case KF: kK(v)[i]=kf(kF(y)[r]); break;
   case KS: kK(v)[i]=ks(kS(y)[r]); break;
   default: TORCH_ERROR("opt: unable to extract setting from table of group options, ",kname(y)," column not implemented"); break;
  }
 }
 return xD(k,v);
}

static Optptr putgroups(Cast c,K x) {
 TORCH_CHECK(x->t==98, "opt: unrecognized settings for ",omap(c)," optimizer, expecting table, given ",kname(x));
 J n=xlen(x);
 TORCH_CHECK(n, "opt: empty table of settings supplied for ",omap(c)," optimizer, need at least one row");
 Optptr o; 
 for(J i=0; i<n; ++i) {
  K d=grouprow(x,i); ParamGroup g({});
  try {
   addoptions(c,d,0,g); r0(d);
  } catch(...) {
   r0(d);
   throw;
  }
  if(!i) o=optinit(c,g);
  else   o->add_param_group(g);
 }
 return o;
}

//static void putbuffers(Cast c,K x,const Device& d,const std::string& k,Optimizer& o) {
static void putbuffers(Cast c,K x,const Device& d,void *k,Optimizer& o) {
 switch(c) {
  case Cast::adagrad: adaput(x,d,k,o); break;
  case Cast::adam:    adamput<AdamParamState>(x,d,k,o); break;
  case Cast::adamw:   adamput<AdamWParamState>(x,d,k,o); break;
  case Cast::lamb:    lambput(x,d,k,o); break;
  case Cast::lbfgs:   lbput(x,d,k,o); break;
  case Cast::rmsprop: rmsput(x,d,k,o); break;
  case Cast::sgd:     sgdput(x,d,k,o); break;
  default: TORCH_ERROR("opt: unable to set buffers, unrecognized optimizer enumeration: ",(I)c);
 }
}

static void putparms(Cast c,K x,Optimizer& o,const Module& m) {
 J n=xlen(x); const auto& p=m.named_parameters(); auto& g=o.param_groups(); auto& s=o.state();
 for(J i=0; i<n; ++i) {
  S s1=statemodule(x,i),s2=statename(x,i); J j=stategroup(x,i); IntArrayRef sz;
  std::string nm=nullsym(s1) ? "tensor" : s1; nm+=" parameter `"; nm+=s2;
  TORCH_CHECK(-1<j && j<(J)g.size(), "opt: group[",j,"] for ",nm, " is invalid, ",g.size()," group(s) defined");
  TORCH_CHECK(p.contains(s2), "opt: unable to find ",nm);
  //const auto& t=p[s2]; const auto& k=c10::guts::to_string(t.unsafeGetTensorImpl());
  const auto& t=p[s2]; const auto& k=t.unsafeGetTensorImpl();
  TORCH_CHECK(s.count(k)==0, "opt: ",nm," is repeated");
  TORCH_CHECK(xsize(statesize(x,i),sz), "opt: unable to get size of ",nm);
  TORCH_CHECK(t.sizes()==sz, "opt: size mismatch for ",nm,", expected ",sz,", given ",t.sizes());
  g[j].params().push_back(t);
  K b=statebuffers(x,i);
  if(b && xlen(b))
   putbuffers(c,b,t.device(),k,o);
 }
}

static K optput(S s,K x,K y,const Moduleptr& m) {
// s:optimizer name, x:table of group options, y:table of parameters
 Cast c=omap(s); Optptr o=putgroups(c,x); putparms(c,y,*o,*m);
 return kopt(c,o,m);
}

// ---------------------------------------------------------------------------
// opt - main optimizer interface function for q
// kstep - given model/optimizer, perform optimizer step unless closure needed
// ---------------------------------------------------------------------------
KAPI opt(K x) {
 KTRY
  J i=0; bool a=env().alloptions,b=xlong(x,1,i); S s; Kopt *o; Kmodule *m; Kmodel *l;
  if(xsym(x,s) || (xsym(x,0,s) && x->t==0)) {
   J n=x->t==-KS ? 1 : x->n;
   TORCH_CHECK(!i, "opt: cannot define group ",i," until optimizer is created with initial parameter group");
   return optinit(b, omap(s), x, n>1+b ? kK(x)[1+b] : nullptr);
  } else if(((o=xoptim(x))) || ((o=xoptim(x,0)))) {
   if(x->n==1 || (x->n==2 && xbool(x,1,a))) {
    return optget(a,false,o->c,o->opt(),o->module());
   } else {
    optedit(b, o->c, x, x->n>1+b ? kK(x)[1+b] : nullptr, i, *o->o, o->m);
    return (K)0;
   }
  } else if(xdict(x,0) && (m=xmodule(x,1)) && x->n==2) {
   K s=kK(x)[0]; // state dictionary defining optimizer, group options and parameters
   return optput(statesym(State::optimizer,true,s),  // optimizer name, e.g. `sgd
                 statetable(State::options,s),       // table of options, a row for each group
                 statetable(State::parms,s),         // table of parameters
                 m->m);
  } else if((l=xmodel(x))) {
   return kopt(l->kopt());
  } else {
   TORCH_ERROR("opt: unrecognized arg(s)");
  }
  return(K)0;
 KCATCH("opt");
}

KAPI kstep(K x) {
 KTRY
  auto *m=xmodel(x); auto *o=m ? m->kopt() : xoptim(x);
  TORCH_CHECK(o, "step: expects model or optimizer, given ", kname(x));
  TORCH_CHECK(o->c != Cast::lbfgs, "LBFGS optimizer requires model, loss & inputs");
  o->opt().step();
  return (K)0;
 KCATCH("step");
}

// ---------------------------------------------------------------------------------------
// lrget - return a double list of learning rates, one per parameter group
// lrset - set each parameter group's learning rate from scalar/list input
// lr - function to query/set learning rate from k
// ---------------------------------------------------------------------------------------
static K lrget(Cast c,const std::vector<ParamGroup>& v) {
 J i=0; F r; K x=ktn(KF,v.size());
 for(const auto& g:v) {
  TORCH_CHECK(g.has_options(), "parameter group options not defined");
  switch(c) {
   case Cast::adagrad: r=static_cast<const AdagradOptions&>(g.options()).lr(); break;
   case Cast::adam:    r=static_cast<const    AdamOptions&>(g.options()).lr(); break;
   case Cast::adamw:   r=static_cast<const   AdamWOptions&>(g.options()).lr(); break;
   case Cast::lamb:    r=static_cast<const    LambOptions&>(g.options()).lr(); break;
   case Cast::lbfgs:   r=static_cast<const   LBFGSOptions&>(g.options()).lr(); break;
   case Cast::rmsprop: r=static_cast<const RMSpropOptions&>(g.options()).lr(); break;
   case Cast::sgd:     r=static_cast<const     SGDOptions&>(g.options()).lr(); break;
   default: TORCH_ERROR("unrecognized optimizer: ",(I)c,", unable to retrieve learning rate");
  }
  kF(x)[i++]=r;
 }
 return x;
}

static void lrset(Cast c,std::vector<ParamGroup>& v,J n,double *lr) {
 TORCH_CHECK(n==1 || (unsigned)n==v.size(),"length error: ",n," learning rates given for ",v.size()," parameter group",(v.size() !=1 ? "s" : ""));
 int64_t i=0; double r;
 for(auto& g:v) {
  TORCH_CHECK(g.has_options(), "parameter group options not defined");
  r=(n==1) ? lr[0] : lr[i++];
  switch(c) {
   case Cast::adagrad: static_cast<AdagradOptions&>(g.options()).lr(r); break;
   case Cast::adam:    static_cast<   AdamOptions&>(g.options()).lr(r); break;
   case Cast::adamw:   static_cast<  AdamWOptions&>(g.options()).lr(r); break;
   case Cast::lamb:    static_cast<   LambOptions&>(g.options()).lr(r); break;
   case Cast::lbfgs:   static_cast<  LBFGSOptions&>(g.options()).lr(r); break;
   case Cast::rmsprop: static_cast<RMSpropOptions&>(g.options()).lr(r); break;
   case Cast::sgd:     static_cast<    SGDOptions&>(g.options()).lr(r); break;
   default: TORCH_ERROR("unrecognized optimizer: ",(I)c,", unable to set learning rate");
  }
 }
}

KAPI lr(K x) {
 KTRY
  TORCH_CHECK(!x->t, "lr: not implemented for ",kname(x));
  TORCH_CHECK(x->n>0 && x->n<3, "lr: expecting 1-2 args, given ",x->n);
  if(x->n==1) {
   auto m=xmodel(x); auto o=m ? m->kopt() : xoptim(x);
   TORCH_CHECK(o, "lr: unrecognized arg(s), expecting model/optimizer to query learning rate(s)");
   return lrget(o->c, o->opt().param_groups());
  } else {
   auto m=xmodel(x,0); auto o=m ? m->kopt() : xoptim(x,0); J n; double *r;
   TORCH_CHECK(o && xdouble(x,1,n,r), "lr: unrecognized arg(s), expecting model/optimizer and learning rate(s) as double(s)");
   lrset(o->c, o->opt().param_groups(),n,r);
   return (K)0;
  }
 KCATCH("lr");
}

// --------------------------------------------------
// add optimizer api functions to library dictionary
// --------------------------------------------------
void optfn(K x) {
 fn(x, "opt",  KFN(opt),1);
 fn(x, "step", KFN(kstep),1);
 fn(x, "lr",   KFN(lr),1);
}
