#include "ktorch.h"

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
// osize - optimizer size, i.e. number of parameters defined 
//        (defined in pytorch, but marked with "obsolete" warning)
// --------------------------------------------------------------------------------------
K kopt(Cast x,const Optptr& y,const BaseModule& z) {return kptr(new Kopt(x,y,z));}

static Cast omap(S s) {
 for(auto& m:env().opt)
  if(s==std::get<0>(m)) return std::get<1>(m);
 AT_ERROR("unrecognized optimizer: ",s);
}

static S omap(Cast c) {
 for(auto& m:env().opt)
  if(c==std::get<1>(m)) return std::get<0>(m);
 AT_ERROR("unrecognized optimizer: ",(I)c);
}

static Setting oset(S s) {
 for(auto& m:env().oset)
  if(s==std::get<0>(m)) return std::get<1>(m);
 AT_ERROR("unrecognized optimizer setting: ",s);
}

static S oset(Setting e) {
 for(auto& m:env().oset) if(e==std::get<1>(m)) return std::get<0>(m);
 AT_ERROR("unrecognized optimizer setting: ",(I)e);
}

static size_t osize(const Optimizer& o) {
  size_t n=0; for(const auto& g:o.param_groups()) n+=g.params().size(); return n;
}

static size_t osize(const Optptr& o) {return osize(*o);}

// --------------------------------------------------------------------------------------
// getoptions - set defaults if undefined, return reference to optimizer-specific options
// --------------------------------------------------------------------------------------
static SGDOptions& getoptions(ParamGroup& g) {
 if(!g.has_options()) g.set_options(std::make_unique<SGDOptions>(LR));
 return static_cast<SGDOptions&>(g.options());
}

template<typename O> static O& getoptions(ParamGroup& g) {
 if(!g.has_options()) g.set_options(std::make_unique<O>());
 return static_cast<O&>(g.options());
}

// --------------------------------------------------------------------------------------
// bget - find buffer (vector of longs/tensors) from dictionary given name
// bset - set optimization buffers from k dictionary
//      - vector of longs (e.g. step_buffers, one per parameter)
//      - vector of tensors
//      - deque of tensors (LBFGS)
//      - single tensor (LBFGS)
// --------------------------------------------------------------------------------------
/*
static K bget(K x,const char* s) {
 auto i=xdict(x) ? kfind(kK(x)[0],s) : -1;
 return (i<0 || kK(x)[1]->t) ? nullptr : kK(kK(x)[1])[i];
}

static void bset(size_t n,const char* s,const TensorVector& p,LongVector& v,const K x) {
 // restore vector of longs (parameter vector not referenced, included for uniform call)
 K y=bget(x,s);
 if(!y || !y->n) return;
 if(y->t != KJ) AT_ERROR("type error: ",s,", expecting long list, input is ",kname(y->t));
 if(n != (unsigned)y->n) AT_ERROR("length error: ",s,", expecting ",n," longs, input has ",y->n);
 v.resize(n);
 for(size_t i=0; i<n; ++i) v.emplace_back(kJ(y)[i]);
}

static void bset(size_t n,const char* s,const TensorVector& p,TensorVector& v,const K x) {
 K y=bget(x,s);
 if(!y || !y->n) return;
 if(y->t) AT_ERROR("type error: ",s,", expecting general list, input is ",kname(y->t));
 if(n != (unsigned)y->n) AT_ERROR("length error: ",s,", expecting ",n," arrays, input has ",y->n);
 v.resize(n);
 for(size_t i=0; i<n; ++i) {
  auto a=kput(kK(y)[i]);
  auto b=torch::zeros_like(p.at(i));
  if(a.dtype() != b.dtype())
   AT_ERROR("type mismatch: ",s,"[",i,"] is ",b.dtype(),", input is ",a.dtype());
  if(!b.is_same_size(a))
   AT_ERROR("size mismatch: ",s,"[",i,"] is ",b.sizes(),", input is ",a.sizes());
  if(a.device() != b.device())
   b.set_data(a);
  else
   b.set_(a);
  v[i]=std::move(b);
 }
}

static void bset(size_t n,const char* s,const TensorVector& p,TensorDeque& v,const K x) {
 // used w'LBFGS, not sure if parameter count/type/device relevant
 K y=bget(x,s);
 if(!y || !y->n) return;
 if(y->t) AT_ERROR("type error: ",s,", expecting general list, input is ",kname(y->t));
 v.resize(n);
 for(size_t i=0; i<n; ++i)
  v[i]=kput(kK(y)[i]);
}

static void bset(size_t n,const char* s,const TensorVector& p,Tensor& t,const K x) {
 // used w'LBFGS, not sure if parameter count/type/device relevant
 K y=bget(x,s);
 if(!y || !y->n) return;
 t=kput(y);
}
*/

// ----------------------------------------------------------------------------------------
// adagrad - parse args (lr;lrdecay;wtdecay) or (..;name/val pairs/dict)
//         - if given options,buffers, allocate new optimizer and return ptr
//         - if given previously allocated ptr, return dictionary of options & buffers
// ----------------------------------------------------------------------------------------
static void adagrad(K x,J i,AdagradOptions& o) {
 Pairs p; J n=xargc(x,i,p); double f;
 if(n && xnum(x,i,f)) {i++; n--; if(f==f) o.lr(f);}
 if(n && xnum(x,i,f)) {i++; n--; if(f==f) o.lr_decay(f);}
 if(n && xnum(x,i,f)) {i++; n--; if(f==f) o.weight_decay(f);}
 if(n) AT_ERROR("unrecognized arg(s) for Adagrad optimizer");
 while(xpair(p))
  switch(oset(p.k)) {
   case Setting::lr:      f=pdouble(p); if(f==f) o.lr(f); break;
   case Setting::lrdecay: f=pdouble(p); if(f==f) o.lr_decay(f); break;
   case Setting::decay:   f=pdouble(p); if(f==f) o.weight_decay(f); break;
   default: AT_ERROR("unrecognized option: ",p.k," for Adagrad optimization"); break;
  }
}

static void adagrad(K x,ParamGroup& g) {
 auto& o=getoptions<AdagradOptions>(g); Pairs p; J i=0,n=xargc(x,0,p); double f;
 if(n && xnum(x,i,f)) {i++; n--; if(f==f) o.lr(f);}
 if(n && xnum(x,i,f)) {i++; n--; if(f==f) o.lr_decay(f);}
 if(n && xnum(x,i,f)) {i++; n--; if(f==f) o.weight_decay(f);}
 if(n) AT_ERROR("unrecognized arg(s) for Adagrad optimizer");
 while(xpair(p))
  switch(oset(p.k)) {
   case Setting::lr:      f=pdouble(p); if(f==f) o.lr(f); break;
   case Setting::lrdecay: f=pdouble(p); if(f==f) o.lr_decay(f); break;
   case Setting::decay:   f=pdouble(p); if(f==f) o.weight_decay(f); break;
   default: AT_ERROR("unrecognized option: ",p.k," for Adagrad optimization"); break;
  }
}

static Optptr adagrad(const TensorVector& w,const AdagradOptions& a,K y) {
 auto o=std::make_shared<Adagrad>(w,a);
 auto n=o->state().size();
 if(y && n) {
/* PATCH
  bset(n, "step_buffers", o->parameters(), o->step_buffers, y);
  bset(n, "sum_buffers",  o->parameters(), o->sum_buffers,  y);
*/
 }
 return o;
}

static K adagrad(bool a,const AdagradOptions& o) { //return all or non-default options as k dictionary
 K x=xD(ktn(KS,0),ktn(0,0)); AdagradOptions d;
 if(a || d.lr()           != o.lr())           OPTSET(x, lr,      kf(o.lr()));
 if(a || d.lr_decay()     != o.lr_decay())     OPTSET(x, lrdecay, kf(o.lr_decay()));
 if(a || d.weight_decay() != o.weight_decay()) OPTSET(x, decay,   kf(o.weight_decay()));
 return x;
}

static J adasize(bool b,const AdagradParamState& s) { //return count of elements/bytes in parm buffers
 return b ? objnum(s.step()) + objnum(s.sum()) : objbytes(s.step()) + objbytes(s.sum());
}

static K adaget(const AdagradParamState& s) {
 K x=xD(ktn(KS,0),ktn(0,0));
 dictadd(x, "step", kj(s.step()));
 dictadd(x, "sum",  kget(s.sum()));
 return x;
}

// ----------------------------------------------------------------------------------------
// adam - parse args (lr;beta1;beta2;eps;wtdecay;amsgrad) or (..;name-value pairs/dict)
// ----------------------------------------------------------------------------------------
template<typename A> static void adam(K x,J i,A& o) {
 Pairs p; J n=xargc(x,i,p); bool b; double f;
 if(n && xnum(x,i,f)) {i++; n--; if(f==f) o.lr(f);}
 if(n && xnum(x,i,f)) {i++; n--; if(f==f) o.betas(std::make_tuple(f,std::get<1>(o.betas())));}
 if(n && xnum(x,i,f)) {i++; n--; if(f==f) o.betas(std::make_tuple(std::get<0>(o.betas()),f));}
 if(n && xnum(x,i,f)) {i++; n--; if(f==f) o.eps(f);}
 if(n && xnum(x,i,f)) {i++; n--; if(f==f) o.weight_decay(f);}
 if(n && xbool(x,i,b)){i++; n--; o.amsgrad(b);}
 if(n) AT_ERROR("unrecognized arg(s) for Adam optimizer");
 while(xpair(p))
  switch(oset(p.k)) {
   case Setting::lr:      f=pdouble(p); if(f==f) o.lr(f); break;
   case Setting::beta1:   f=pdouble(p); if(f==f) o.betas(std::make_tuple(f,std::get<1>(o.betas()))); break;
   case Setting::beta2:   f=pdouble(p); if(f==f) o.betas(std::make_tuple(std::get<0>(o.betas()),f)); break;
   case Setting::eps:     f=pdouble(p); if(f==f) o.eps(f); break;
   case Setting::decay:   f=pdouble(p); if(f==f) o.weight_decay(f); break;
   case Setting::amsgrad: o.amsgrad(pbool(p)); break;
   default: AT_ERROR("unrecognized option: ",p.k," for Adam optimization"); break;
  }
}

template<typename O> static void adam(K x,ParamGroup& g) {
 auto& o=getoptions<O>(g); Pairs p; J i=0,n=xargc(x,0,p); bool b; double f;
 if(n && xnum(x,i,f)) {i++; n--; if(f==f) o.lr(f);}
 if(n && xnum(x,i,f)) {i++; n--; if(f==f) o.betas(std::make_tuple(f,std::get<1>(o.betas())));}
 if(n && xnum(x,i,f)) {i++; n--; if(f==f) o.betas(std::make_tuple(std::get<0>(o.betas()),f));}
 if(n && xnum(x,i,f)) {i++; n--; if(f==f) o.eps(f);}
 if(n && xnum(x,i,f)) {i++; n--; if(f==f) o.weight_decay(f);}
 if(n && xbool(x,i,b)){i++; n--; o.amsgrad(b);}
 if(n) AT_ERROR("unrecognized arg(s) for Adam optimizer");
 while(xpair(p))
  switch(oset(p.k)) {
   case Setting::lr:      f=pdouble(p); if(f==f) o.lr(f); break;
   case Setting::beta1:   f=pdouble(p); if(f==f) o.betas(std::make_tuple(f,std::get<1>(o.betas()))); break;
   case Setting::beta2:   f=pdouble(p); if(f==f) o.betas(std::make_tuple(std::get<0>(o.betas()),f)); break;
   case Setting::eps:     f=pdouble(p); if(f==f) o.eps(f); break;
   case Setting::decay:   f=pdouble(p); if(f==f) o.weight_decay(f); break;
   case Setting::amsgrad: o.amsgrad(pbool(p)); break;
   default: AT_ERROR("unrecognized option: ",p.k," for Adam optimization"); break;
  }
}

template<typename O,typename A> static Optptr adam(const TensorVector& w,const A& a,K y) {
 auto o=std::make_shared<O>(w,a);
 auto n=o->state().size();
 if(y && n) {
/* PATCH
  bset(n, "step_buffers",                o->parameters(), o->step_buffers,               y);
  bset(n, "exp_average_buffers",         o->parameters(), o->exp_average_buffers,        y);
  bset(n, "exp_average_sq_buffers",      o->parameters(), o->exp_average_sq_buffers,     y);
  bset(n, "max_exp_average_sq_buffers",  o->parameters(), o->max_exp_average_sq_buffers, y);
*/
 }
 return o;
}

template<typename O> static K adam(bool a,const Optimizer& o) { //return all or non-default options as k dictionary
 K x=xD(ktn(KS,0),ktn(0,0)); O d; const auto& v=static_cast<const O&>(o.defaults());
 if(a || d.lr()           != v.lr())                       OPTSET(x, lr,      kf(v.lr()));
 if(a || std::get<0>(d.betas()) != std::get<0>(v.betas())) OPTSET(x, beta1,   kf(std::get<0>(v.betas())));
 if(a || std::get<1>(d.betas()) != std::get<1>(v.betas())) OPTSET(x, beta2,   kf(std::get<1>(v.betas())));
 if(a || d.eps()          != v.eps())                      OPTSET(x, eps,     kf(v.eps()));
 if(a || d.weight_decay() != v.weight_decay())             OPTSET(x, decay,   kf(v.weight_decay()));
 if(a || d.amsgrad()      != v.amsgrad())                  OPTSET(x, amsgrad, kb(v.amsgrad()));
 return x;
}

template<typename O> static K adam(bool a,const O& o) { //return all or non-default options as k dictionary
 K x=xD(ktn(KS,0),ktn(0,0)); const O d; 
 if(a || d.lr()           != o.lr())                       OPTSET(x, lr,      kf(o.lr()));
 if(a || std::get<0>(d.betas()) != std::get<0>(o.betas())) OPTSET(x, beta1,   kf(std::get<0>(o.betas())));
 if(a || std::get<1>(d.betas()) != std::get<1>(o.betas())) OPTSET(x, beta2,   kf(std::get<1>(o.betas())));
 if(a || d.eps()          != o.eps())                      OPTSET(x, eps,     kf(o.eps()));
 if(a || d.weight_decay() != o.weight_decay())             OPTSET(x, decay,   kf(o.weight_decay()));
 if(a || d.amsgrad()      != o.amsgrad())                  OPTSET(x, amsgrad, kb(o.amsgrad()));
 return x;
}

template<typename S> static J adamsize(bool b,const S& s) {
 return
  b ?   objnum(s.step()) +   objnum(s.exp_avg()) +   objnum(s.exp_avg_sq()) +   objnum(s.max_exp_avg_sq())
    : objbytes(s.step()) + objbytes(s.exp_avg()) + objbytes(s.exp_avg_sq()) + objbytes(s.max_exp_avg_sq());
}

template<typename S> static K adamget(const S& s) { //template for adam/adamw
 K x=xD(ktn(KS,0),ktn(0,0));
 dictadd(x, "step",           kj(s.step()));
 dictadd(x, "exp_avg",        kget(s.exp_avg()));
 dictadd(x, "exp_avg_sq",     kget(s.exp_avg_sq()));
 dictadd(x, "max_exp_avg_sq", kget(s.max_exp_avg_sq()));
 return x;
}

// ---------------------------------------------------------------------------------------
// lbfgs - (lr;max iter;max eval;tolerance grad;tolerance change;history size)
// ---------------------------------------------------------------------------------------
static void lbfgs(K x,J i,LBFGSOptions& o) {
 Pairs p; J j,n=xargc(x,i,p); double f;
 if(n && xnum(x,i,f))  {i++; n--; if(f==f)  o.lr(f);}
 if(n && xlong(x,i,j)) {i++; n--; if(j!=nj) o.max_iter(j);}
 if(n && xlong(x,i,j)) {i++; n--; if(j!=nj) o.max_eval(j);}
 if(n && xnum(x,i,f))  {i++; n--; if(f==f)  o.tolerance_grad(f);}
 if(n && xnum(x,i,f))  {i++; n--; if(f==f)  o.tolerance_change(f);}
 if(n && xlong(x,i,j)) {i++; n--; if(j!=nj) o.history_size(j);}
 if(n) AT_ERROR("unrecognized arg(s) for LBFGS optimizer");
 while(xpair(p))
  switch(oset(p.k)) {
   case Setting::lr:        f=pdouble(p); if(f==f)  o.lr(f); break;
   case Setting::iter:      j=plong(p);   if(j!=nj) o.max_iter(j); break;
   case Setting::eval:      j=plong(p);   if(j!=nj) o.max_eval(j); break;
   case Setting::gradtol:   f=pdouble(p); if(f==f)  o.tolerance_grad(f); break;
   case Setting::changetol: f=pdouble(p); if(f==f)  o.tolerance_change(f); break;
   case Setting::history:   j=plong(p);   if(j!=nj) o.history_size(j); break;
   default: AT_ERROR("unrecognized option: ",p.k," for LBFGS optimization"); break;
  }
 if(!o.max_eval()) o.max_eval((o.max_iter()*5)/4);
}

static void lbfgs(K x,ParamGroup& g) {
 auto& o=getoptions<LBFGSOptions>(g); Pairs p; J i=0,j,n=xargc(x,0,p); double f;
 if(n && xnum(x,i,f))  {i++; n--; if(f==f)  o.lr(f);}
 if(n && xlong(x,i,j)) {i++; n--; if(j!=nj) o.max_iter(j);}
 if(n && xlong(x,i,j)) {i++; n--; if(j!=nj) o.max_eval(j);}
 if(n && xnum(x,i,f))  {i++; n--; if(f==f)  o.tolerance_grad(f);}
 if(n && xnum(x,i,f))  {i++; n--; if(f==f)  o.tolerance_change(f);}
 if(n && xlong(x,i,j)) {i++; n--; if(j!=nj) o.history_size(j);}
 if(n) AT_ERROR("unrecognized arg(s) for LBFGS optimizer");
 while(xpair(p))
  switch(oset(p.k)) {
   case Setting::lr:        f=pdouble(p); if(f==f)  o.lr(f); break;
   case Setting::iter:      j=plong(p);   if(j!=nj) o.max_iter(j); break;
   case Setting::eval:      j=plong(p);   if(j!=nj) o.max_eval(j); break;
   case Setting::gradtol:   f=pdouble(p); if(f==f)  o.tolerance_grad(f); break;
   case Setting::changetol: f=pdouble(p); if(f==f)  o.tolerance_change(f); break;
   case Setting::history:   j=plong(p);   if(j!=nj) o.history_size(j); break;
   default: AT_ERROR("unrecognized option: ",p.k," for LBFGS optimization"); break;
  }
 if(!o.max_eval()) o.max_eval((o.max_iter()*5)/4);
}

static Optptr lbfgs(const TensorVector& w,const LBFGSOptions& a,K y) {
 auto o=std::make_shared<LBFGS>(w,a);
 //auto n=o->state().size();
 if(y) {
/* PATCH
  bset(n, "d",              o->parameters(), o->d, y);
  bset(n, "t",              o->parameters(), o->t, y);
  bset(n, "H_diag",         o->parameters(), o->H_diag, y);
  bset(n, "prev_flat_grad", o->parameters(), o->prev_flat_grad, y);
  bset(n, "prev_loss",      o->parameters(), o->prev_loss, y);
  bset(n, "old_dirs",       o->parameters(), o->old_dirs, y);
  bset(n, "old_stps",       o->parameters(), o->old_stps, y);
*/
 }
 return o;
}

static K lbfgs(bool a,const LBFGSOptions& o) { //return all or non-default options as k dictionary
 K x=xD(ktn(KS,0),ktn(0,0)); LBFGSOptions d;
 if(a || d.lr()               != o.lr())               OPTSET(x, lr,        kf(o.lr()));
 if(a || d.max_iter()         != o.max_iter())         OPTSET(x, iter,      kj(o.max_iter()));
 if(a || o.max_eval())                                 OPTSET(x, eval,      kj(o.max_eval() ? *o.max_eval() : nj));
 if(a || d.tolerance_grad()   != o.tolerance_grad())   OPTSET(x, gradtol,   kf(o.tolerance_grad()));
 if(a || d.tolerance_change() != o.tolerance_change()) OPTSET(x, changetol, kf(o.tolerance_change()));
 if(a || d.history_size()     != o.history_size())     OPTSET(x, history,   kj(o.history_size()));
 return x;
}

static J lbfgssize(bool b,const LBFGSParamState& s) {
 return
  b ?   objnum(s.func_evals()) +   objnum(s.n_iter())   +   objnum(s.t()) + objnum(s.prev_loss()) +    // scalars
        objnum(s.d())          +   objnum(s.H_diag())   +   objnum(s.prev_flat_grad()) +               // tensors
        objnum(s.old_dirs())   +   objnum(s.old_stps()) +   objnum(s.ro()) +                           // deques
        objnum(s.al())                                                              // optional vector of tensors
    : objbytes(s.func_evals()) + objbytes(s.n_iter())   + objbytes(s.t()) + objbytes(s.prev_loss()) +  // scalars
      objbytes(s.d())          + objbytes(s.H_diag())   + objbytes(s.prev_flat_grad()) +               // tensors
      objbytes(s.old_dirs())   + objbytes(s.old_stps()) + objbytes(s.ro()) +                           // deques
      objbytes(s.al());                                                             // optional vector of tensors
}

static K lbfgsget(const LBFGSParamState& s) {
 K x=xD(ktn(KS,0),ktn(0,0));
 dictadd(x, "func_evals",     kj(s.func_evals()));
 dictadd(x, "n_iter",         kj(s.n_iter()));
 dictadd(x, "t",              kf(s.t()));
 dictadd(x, "prev_loss",      kf(s.prev_loss()));
 dictadd(x, "d",              kget(s.d()));                              // tensor
 dictadd(x, "h_diag",         kget(s.H_diag()));                         // tensor
 dictadd(x, "prev_flag_grad", kget(s.prev_flat_grad()));                 // tensor
 dictadd(x, "old_dirs",       kget(s.old_dirs()));                       // deque
 dictadd(x, "old_stps",       kget(s.old_stps()));                       // deque
 dictadd(x, "ro",             kget(s.ro()));                             // deque
 dictadd(x, "al",             s.al() ? kget(s.al().value()) : ktn(0,0)); // optional vector of tensors
 return x;
}

// ----------------------------------------------------------------------------------------
// rmsprop - parse arg(s) (lr;alpha;eps;decay;momentum;centered) or (..;nm-val pairs/dict)
// ----------------------------------------------------------------------------------------
static void rmsprop(K x,J i,RMSpropOptions& o) {
 Pairs p; J n=xargc(x,i,p); bool b; double f;
 if(n && xnum(x,i,f)) {i++; n--; if(f==f) o.lr(f);}
 if(n && xnum(x,i,f)) {i++; n--; if(f==f) o.alpha(f);}
 if(n && xnum(x,i,f)) {i++; n--; if(f==f) o.eps(f);}
 if(n && xnum(x,i,f)) {i++; n--; if(f==f) o.weight_decay(f);}
 if(n && xnum(x,i,f)) {i++; n--; if(f==f) o.momentum(f);}
 if(n && xbool(x,i,b)){i++; n--; o.centered(b);}
 if(n) AT_ERROR("unrecognized arg(s) for RMSprop optimizer");
 while(xpair(p))
  switch(oset(p.k)) {
   case Setting::lr:        f=pdouble(p); if(f==f) o.lr(f); break;
   case Setting::alpha:     f=pdouble(p); if(f==f) o.alpha(f); break;
   case Setting::eps:       f=pdouble(p); if(f==f) o.eps(f); break;
   case Setting::decay:     f=pdouble(p); if(f==f) o.weight_decay(f); break;
   case Setting::momentum:  f=pdouble(p); if(f==f) o.momentum(f); break;
   case Setting::centered:  o.centered(pbool(p)); break;
   default: AT_ERROR("unrecognized option: ",p.k," for RMSprop optimization"); break;
  }
}

static void rmsprop(K x,ParamGroup& g) {
 auto& o=getoptions<RMSpropOptions>(g); Pairs p; J i=0,n=xargc(x,0,p); bool b; double f;
 if(n && xnum(x,i,f)) {i++; n--; if(f==f) o.lr(f);}
 if(n && xnum(x,i,f)) {i++; n--; if(f==f) o.alpha(f);}
 if(n && xnum(x,i,f)) {i++; n--; if(f==f) o.eps(f);}
 if(n && xnum(x,i,f)) {i++; n--; if(f==f) o.weight_decay(f);}
 if(n && xnum(x,i,f)) {i++; n--; if(f==f) o.momentum(f);}
 if(n && xbool(x,i,b)){i++; n--; o.centered(b);}
 if(n) AT_ERROR("unrecognized arg(s) for RMSprop optimizer");
 while(xpair(p))
  switch(oset(p.k)) {
   case Setting::lr:        f=pdouble(p); if(f==f) o.lr(f); break;
   case Setting::alpha:     f=pdouble(p); if(f==f) o.alpha(f); break;
   case Setting::eps:       f=pdouble(p); if(f==f) o.eps(f); break;
   case Setting::decay:     f=pdouble(p); if(f==f) o.weight_decay(f); break;
   case Setting::momentum:  f=pdouble(p); if(f==f) o.momentum(f); break;
   case Setting::centered:  o.centered(pbool(p)); break;
   default: AT_ERROR("unrecognized option: ",p.k," for RMSprop optimization"); break;
  }
}

static Optptr rmsprop(const TensorVector& w,const RMSpropOptions& a,K y) {
 auto o=std::make_shared<RMSprop>(w,a);
 auto n=o->state().size();
 if(y && n) {
/* PATCH
  bset(n, "square_average_buffers", o->parameters(), o->square_average_buffers, y);
  bset(n, "momentum_buffers",       o->parameters(), o->momentum_buffers,       y);
  bset(n, "grad_average_buffers",   o->parameters(), o->grad_average_buffers,   y);
*/
 }
 return o;
}

static K rmsprop(bool a,const RMSpropOptions& o) { //return all or non-default options as k dictionary
 K x=xD(ktn(KS,0),ktn(0,0)); RMSpropOptions d;
 if(a || d.lr()           != o.lr())           OPTSET(x, lr,       kf(o.lr()));
 if(a || d.alpha()        != o.alpha())        OPTSET(x, alpha,    kf(o.alpha()));
 if(a || d.eps()          != o.eps())          OPTSET(x, eps,      kf(o.eps()));
 if(a || d.weight_decay() != o.weight_decay()) OPTSET(x, decay,    kf(o.weight_decay()));
 if(a || d.momentum()     != o.momentum())     OPTSET(x, momentum, kf(o.momentum()));
 if(a || d.centered()     != o.centered())     OPTSET(x, centered, kb(o.centered()));
 return x;
}

static J rmssize(bool b,const RMSpropParamState& s) {
 return
  b ?   objnum(s.step()) +  objnum(s.square_avg()) +   objnum(s.momentum_buffer()) +   objnum(s.grad_avg())
    : objbytes(s.step()) +objbytes(s.square_avg()) + objbytes(s.momentum_buffer()) + objbytes(s.grad_avg());
}

static K rmsget(const RMSpropParamState& s) {
 K x=xD(ktn(KS,0),ktn(0,0));
 dictadd(x, "step",       kj(s.step()));
 dictadd(x, "square_avg", kget(s.square_avg()));
 dictadd(x, "momentum",   kget(s.momentum_buffer()));
 dictadd(x, "grad_avg",   kget(s.grad_avg()));
 return x;
}

// ----------------------------------------------------------------------------------------
// SGD parse args (lr;momentum;dampening;wtdecay;nesterov) or (..;name-value pairs/dict)
// ----------------------------------------------------------------------------------------
static void sgd(K x,J i,SGDOptions& o) {
 Pairs p; J n=xargc(x,i,p); bool b; double f;
 if(n && xnum(x,i,f)) {i++; n--; if(f==f) o.lr(f);}
 if(n && xnum(x,i,f)) {i++; n--; if(f==f) o.momentum(f);}
 if(n && xnum(x,i,f)) {i++; n--; if(f==f) o.dampening(f);}
 if(n && xnum(x,i,f)) {i++; n--; if(f==f) o.weight_decay(f);}
 if(n && xbool(x,i,b)){i++; n--; o.nesterov(b);}
 if(n) AT_ERROR("unrecognized arg(s) for SGD optimizer");
 while(xpair(p))
  switch(oset(p.k)) {
   case Setting::lr:        f=pdouble(p); if(f==f) o.lr(f); break;
   case Setting::momentum:  f=pdouble(p); if(f==f) o.momentum(f); break;
   case Setting::dampening: f=pdouble(p); if(f==f) o.dampening(f); break;
   case Setting::decay:     f=pdouble(p); if(f==f) o.weight_decay(f); break;
   case Setting::nesterov:  o.nesterov(pbool(p)); break;
   default: AT_ERROR("unrecognized option: ",p.k," for SGD optimization"); break;
  }
}

static void sgd(K x,ParamGroup& g) {
 auto o=getoptions(g); Pairs p; J i=0,n=xargc(x,0,p); bool b; double f;
 if(n && xnum(x,i,f)) {i++; n--; if(f==f) o.lr(f);}
 if(n && xnum(x,i,f)) {i++; n--; if(f==f) o.momentum(f);}
 if(n && xnum(x,i,f)) {i++; n--; if(f==f) o.dampening(f);}
 if(n && xnum(x,i,f)) {i++; n--; if(f==f) o.weight_decay(f);}
 if(n && xbool(x,i,b)){i++; n--; o.nesterov(b);}
 if(n) AT_ERROR("unrecognized arg(s) for SGD optimizer");
 while(xpair(p))
  switch(oset(p.k)) {
   case Setting::lr:        f=pdouble(p); if(f==f) o.lr(f); break;
   case Setting::momentum:  f=pdouble(p); if(f==f) o.momentum(f); break;
   case Setting::dampening: f=pdouble(p); if(f==f) o.dampening(f); break;
   case Setting::decay:     f=pdouble(p); if(f==f) o.weight_decay(f); break;
   case Setting::nesterov:  o.nesterov(pbool(p)); break;
   default: AT_ERROR("unrecognized option: ",p.k," for SGD optimization"); break;
  }
}

static Optptr sgd(const TensorVector& w,const SGDOptions& a,K y) {
 auto o=std::make_shared<SGD>(w,a);
 auto n=o->state().size();
 if(y && n) {
  // PATCH: bset(n, "momentum_buffers", o->parameters(), o->momentum_buffers, y);
 }
 return o;
}

static K sgd(bool a,const SGDOptions& o) { //return all or non-default options as k dictionary
 K x=xD(ktn(KS,0),ktn(0,0)); SGDOptions d(LR);
 if(a || d.lr()           != o.lr())           OPTSET(x, lr,        kf(o.lr()));
 if(a || d.momentum()     != o.momentum())     OPTSET(x, momentum,  kf(o.momentum()));
 if(a || d.dampening()    != o.dampening())    OPTSET(x, dampening, kf(o.dampening()));
 if(a || d.weight_decay() != o.weight_decay()) OPTSET(x, decay,     kf(o.weight_decay()));
 if(a || d.nesterov()     != o.nesterov())     OPTSET(x, nesterov,  kb(o.nesterov()));
 return x;
}

static J sgdsize(bool b, const SGDParamState& s) {
 return b ? objnum(s.momentum_buffer()) : objbytes(s.momentum_buffer());
}

static K sgdget(const SGDParamState& s) {
 K x=xD(ktn(KS,0),ktn(0,0));
 dictadd(x, "momentum",  kget(s.momentum_buffer()));
 return x;
}

// -------------------------------------------------------------------------------
// optdict - return a list of dictionaries, one per group of optimizer settings
// buffersize - number of elements or bytes of optimizer buffers for each parameter
// -------------------------------------------------------------------------------
static K optdict(bool a,Cast c,const Optimizer& o) {
 size_t i=0,n=o.param_groups().size(); K x,r=ktn(0,n);
 for(const auto&g:o.param_groups()) {
  switch(c) {
   case Cast::adagrad: x=adagrad(a, static_cast<const AdagradOptions&>(g.options())); break;
   case Cast::adam:       x=adam(a, static_cast<const AdamOptions&>   (g.options())); break;
   case Cast::adamw:      x=adam(a, static_cast<const AdamWOptions&>  (g.options())); break;
   case Cast::lbfgs:     x=lbfgs(a, static_cast<const LBFGSOptions&>  (g.options())); break;
   case Cast::rmsprop: x=rmsprop(a, static_cast<const RMSpropOptions&>(g.options())); break;
   case Cast::sgd:         x=sgd(a, static_cast<const SGDOptions&>    (g.options())); break;
   default: AT_ERROR("Unrecognized optimizer: ",(I)c);
  }
  kK(r)[i++]=x;
 }
 return r;
}

static J buffersize(bool b,Cast c,const ParamState& p) {
 switch(c) {
  case Cast::adagrad: return   adasize(b, static_cast<const AdagradParamState&>(p));
  case Cast::adam:    return  adamsize(b, static_cast<const AdamParamState&>(p));
  case Cast::adamw:   return  adamsize(b, static_cast<const AdamWParamState&>(p));
  case Cast::lbfgs:   return lbfgssize(b, static_cast<const LBFGSParamState&>(p));
  case Cast::rmsprop: return   rmssize(b, static_cast<const RMSpropParamState&>(p));
  case Cast::sgd:     return   sgdsize(b, static_cast<const SGDParamState&>(p));
  default: AT_ERROR("unrecognized optimizer: ",(I)c,", unable to retrieve parameter state");
 }
}

J buffersize(bool b,Cast c,const Optimizer& o) {
 J n=0;
 for(const auto& p:o.state())
  n+=buffersize(b,c,*p.second);
 return n;
}

// --------------------------------------------------------------------------------------
// parmkeys - return columns for table describing optimizer parameter groups & buffers
// moduletype - given parameter, attempt to find parent module type
// basemodule - return single module if container has only one child, no parms or buffers
// parmname - given parameter, search containing module(s), return name if found
// parmsym - return string from parmname as symbol
// --------------------------------------------------------------------------------------
static K parmkeys(bool b) {
 K x=ktn(KS, b ? 6 : 5);
 kS(x)[0]=statekey(State::id);
 kS(x)[1]=statekey(State::group);
 kS(x)[2]=statekey(State::module);
 kS(x)[3]=statekey(State::name);
 kS(x)[4]=statekey(State::size);
 if(b) kS(x)[5]=statekey(State::buffers);
 return x;
}

static S moduletype(const Tensor& p,const Module& m) {
 for(const auto& a:m.modules(false))
  for(const auto& t:a->parameters(false))
   if(t.is_same(p)) return msym(*a);
 return env().nullsym;
}

static const Module& basemodule(const Module& m) {
 if(m.children().size()==1 && m.parameters(false).size()==0 && m.buffers(false).size()==0) 
  return *m.children()[0];  // container has only one child module, remove a layer
 else
  return m;
}

static std::string parmname(const Tensor& p,const Module& m) {
 for(const auto& a:basemodule(m).named_parameters())
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
  case Cast::lbfgs:   return lbfgsget(static_cast<const LBFGSParamState&>(p));
  case Cast::rmsprop: return   rmsget(static_cast<const RMSpropParamState&>(p));
  case Cast::sgd:     return   sgdget(static_cast<const SGDParamState&>(p));
  default: AT_ERROR("unrecognized optimizer: ",(I)c,", unable to retrieve parameter state");
 }
}

static K getparms(bool b,Cast c,const Optimizer& o,const Module& m) {
 J g=0,i=0,n=osize(o);
 K id=ktn(KJ,n),gp=ktn(KJ,n),md=ktn(KS,n),nm=ktn(KS,n),sz=ktn(0,n),bf; if(b) bf=ktn(0,n);
 const auto& s=o.state();
 for(auto& pg:o.param_groups()) {
  for(auto& p:pg.params()) {
    auto *t=p.unsafeGetTensorImpl();
    kJ(id)[i]=(intptr_t)t;
    kJ(gp)[i]=g;
    kS(md)[i]=moduletype(p,m);
    kS(nm)[i]=parmsym(p,m);
    kK(sz)[i]=tensorsize(p,Attr::size);
    if(b) kK(bf)[i]=s.size() ? getparms(c, *s.at(c10::guts::to_string(t))) : KDICT;
    i++;
  }
  g++;
 }
 return xT(xD(parmkeys(b),b ? knk(6,id,gp,md,nm,sz,bf) : knk(5,id,gp,md,nm,sz)));
}

// ---------------------------------------------------------------------------------------
// addtensor - add tensor w'optional name if not already registered in target module
// addvector - add vector of tensors if not already registered in target module
// adddict - add named tensors if not already registered in target module
// addmodule - add child module if not already registered in target module
// ---------------------------------------------------------------------------------------
static void addtensor(const Tensor& t,Module& m,std::string s={});
static void addtensor(const Tensor& t,Module& m,std::string s) {
 for(const auto& p:m.parameters()) if(p.is_same(t)) return; //tensor already added
 m.register_parameter(s.size() ? s : c10::to_string(m.parameters().size()),t);
}

static void addvector(const TensorVector& v,Module& m) {
 for(const auto& t:v) addtensor(t,m);
}

static void adddict(const TensorDict& d,Module& m) {
 for(const auto& a:d) addtensor(a.value(),m,a.key());
}

static void adddict(const TensorDict& d,J n,S *s,Module& m) {
 for(J i=0; i<n; ++i) addtensor(d[s[i]],m,s[i]);
}

static void addmodule(Module& a,Module& m) {
 for(const auto& c:m.children()) if(c.get() == &a) return; //module already added
 S s=mname(a); m.register_module(s ? s : c10::to_string(m.children().size()),a.shared_from_this());
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
    AT_ERROR("opt: parameter[",j,"] is duplicate of parameter[",i,"]");
 }
 return v;
}

static TensorVector duplicate(const TensorDict& d) {
 const auto& k=d.keys();
 for(size_t i=0; i<d.size(); ++i) {
  const auto& t=d[k[i]];
  for(size_t j=i+1; j<d.size(); ++j)
   if(t.is_same(d[k[j]]))
    AT_ERROR("opt: parameter[`",k[j],"] is duplicate of parameter[`",k[i],"]");
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
  AT_ERROR("opt: parameter[",i,"] already in group ",g, " (",s1," module parameter `",s2,")");
 else
  AT_ERROR("opt: parameter[",i,"] already in group ", g);
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
static TensorVector moduleparms(const Module& m,J n,J *j) {
 TensorVector v;
 const auto& a=m.modules(false);
 for(J i=0;i<n;++i) {
  J k=j[i];
  TORCH_CHECK(k>=0, "opt: module[",k,"] invalid");
  TORCH_CHECK(k<a.size(), "opt: module[",k,"] out of bounds, index must be less than ",a.size());
  const auto& p=a.at(k)->parameters();
  for(const auto& t:p)
   TORCH_CHECK(!duplicate(v,t), "opt: duplicate parameter(s) with module[",k,"]");
  v.insert(v.end(), p.begin(), p.end());
 }
 return v;
}

static TensorVector moduleparms(const Module& m,J n,S *s) {
 TensorVector v;
 const auto& a=m.named_modules("",false);
 const auto& z=m.named_parameters(true);
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
   AT_ERROR("opt: no module or parameter named `",k);
  }
 }
 return v;
}

static TensorVector moduleparms(const Module& a,K x) {
 TensorVector v;
 if(!x)
  v=duplicate(a.named_parameters());
 else if(x->t==KJ || x->t==-KJ)
   v=x->t==KJ ? moduleparms(a,x->n,kJ(x)) : moduleparms(a,1,&x->j);
 else if(x->t==KS || x->t==-KS) 
   v=x->t==KS ? moduleparms(a,x->n,kS(x)) : moduleparms(a,1,&x->s);
 else
  AT_ERROR("opt: ",msym(a)," module supplied with unrecognized ",kname(x)," selector(s)");
 return v;
}

static TensorVector moduleparms(Module& a,K x,const Optimizer& o,Module& m) {
 TensorVector v;
 if(!x)
  v=a.parameters();
 else if(x->t==KJ || x->t==-KJ)
   v=x->t==KJ ? moduleparms(a,x->n,kJ(x)) : moduleparms(a,1,&x->j);
 else if(x->t==KS || x->t==-KS) 
   v=x->t==KS ? moduleparms(a,x->n,kS(x)) : moduleparms(a,1,&x->s);
 else
  AT_ERROR("opt: ",msym(a)," module supplied with unrecognized ",kname(x)," selector(s)");
 parmcheck(o,m,v); addmodule(a,m);
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
   AT_ERROR("opt: no dictionary parameter named `",k);
  }
 }
 return v;
}

static TensorVector dictparms(const TensorDict& d,K x,const Optimizer& o,Module& m) {
 TensorVector v;
 if(!x)
  v=duplicate(d.values());
 else if(x->t==KS || x->t==-KS) 
   v=x->t==KS ? dictparms(d,x->n,kS(x)) : dictparms(d,1,&x->s);
 else
  AT_ERROR("opt: tensor dictionary supplied with unrecognized ",kname(x)," selector(s)");
 parmcheck(o,m,v);
 if(!x)
  adddict(d,m);
 else if(x->t==KS)
  adddict(d,x->n,kS(x),m);
 else
  adddict(d,1,&x->s,m);
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
  TORCH_CHECK(k<a.size(), "opt: vector[",k,"] out of bounds, index must be less than ",a.size());
  const auto& t=a[k];
  TORCH_CHECK(!duplicate(v,t), "opt: duplicate parameter from vector[",k,"]");
  v.push_back(t);
 }
 return v;
}

static TensorVector vectorparms(const TensorVector& a,K x,const Optimizer& o,Module& m) {
 TensorVector v;
 if(!x)
  v=duplicate(a);
 else if(x->t==KJ || x->t==-KJ) 
   v=x->t==KJ ? vectorparms(a,x->n,kJ(x)) : vectorparms(a,1,&x->j);
 else
  AT_ERROR("opt: tensor vector supplied with unrecognized ",kname(x)," selector(s)");
 parmcheck(o,m,v); addvector(v,m);
 return v;
}
 
// ---------------------------------------------------------------------------------------
// optparms - return vector of parameters from given tensor/sequential ptr
// optinit - initialize one of the supported optimizers, return pointer to k
// optstate - return optimizer name & options and optionally, internal buffer values
// opt - main optimizer interface function for q
// ---------------------------------------------------------------------------------------
static TensorVector optparms(K x,J i) {
 if(auto *a=xmodule(x,i))
  return mref(a->m).parameters();
 else if(auto *a=xvec(x,i))
  return *a;
 else if(auto *a=xten(x,i))
  return {*a};
 else if(x->t==-KS || xempty(x,i) || xdict(x))
  return {};
 else if(xptr(x,i))
  AT_ERROR("unrecognized pointer, expecting tensor(s) or module(s)");
 else
  AT_ERROR("unrecognized argument, ",kname(x->t ? x->t : kK(x)[i]->t),", expecting tensor(s) or module(s)");
}

static K optinit(S s,K x,K y=nullptr);  //s:optimizer name, x:options, y:buffers
static K optinit(S s,K x,K y) {
 J i=xdict(x) ? -1 : 2; Cast c=omap(s);
 if(!(x->t==-KS || xdict(x) || xempty(x,1) || xptr(x,1)))
  AT_ERROR("optimizer ",s," expects args of form:\n",
           "name\n", "(name; parm(s); option(s)..)\n" "(saved state; parm(s))");
 auto w=optparms(x,1); Kmodule *k; Optptr o; BaseModule m;
 switch(c) {
  case Cast::adagrad: {auto a=AdagradOptions();  adagrad(x,i,a); o=adagrad(w,a,y); break;}
  case Cast::adam:    {auto a=AdamOptions();     adam(x,i,a);    o=adam<Adam>(w,a,y);  break;}
  case Cast::adamw:   {auto a=AdamWOptions();    adam(x,i,a);    o=adam<AdamW>(w,a,y); break;}
  case Cast::lbfgs:   {auto a=LBFGSOptions();    lbfgs(x,i,a);   o=lbfgs(w,a,y);   break;}
  case Cast::rmsprop: {auto a=RMSpropOptions();  rmsprop(x,i,a); o=rmsprop(w,a,y); break;}
  case Cast::sgd:     {auto a=SGDOptions(LR); sgd(x,i,a);     o=sgd(w,a,y);     break;}
  default: AT_ERROR("unrecognized optimizer: ",s); break;
 }
 if((k=xmodule(x,1)))
  addmodule(mref(k),*m);
 return kopt(c,o,m);
}

K optstate(bool a,bool b,Cast c,const Optimizer &o,const Module& m) {
 K k=ktn(KS,3),v=ktn(0,3);
 kS(k)[0]=statekey(State::module);  kK(v)[0]=ks(omap(c));
 kS(k)[1]=statekey(State::options); kK(v)[1]=optdict(a,c,o);
 kS(k)[2]=statekey(State::parms);   kK(v)[2]=getparms(b,c,o,m);
 return xD(k,v);
}

// this version of optstate called from generic state function in k-level api
K optstate(Ktag *g,K x) {
 bool a=env().alloptions;
 if(x->n==1 || (x->n==2 && xbool(x,1,a)))
  return optstate(a,true,g->c,*((Kopt*)g)->o,*((Kopt*)g)->m);
 else
  AT_ERROR("optimizer state requires 1-2 args: previously allocated ptr or (ptr;options flag)");
}

KAPI optstate2(K x,K y) {
 KTRY
  Kopt* o=xoptim(x); Kmodule *m=xmodule(y);
  TORCH_CHECK(o && m, "need optimizer & module");
  return optstate(true,true,o->c,*o->o,mref(m->m));
 KCATCH("optstate");
}

// ---------------------------------------------------------------------------------------
// addparms - 
// addoptions - parse k args into optimizer-specific options stored in a parameter group
// optinit - create a parameter group (vector of parameters & optimizer options)
//           from the parameter group initialize the optimizer 
//           and "base" module which stores requisite module(s) & parameters needed to
//           recreate the optimizer group(s)
// ---------------------------------------------------------------------------------------
static TensorVector addparms(K x,J i,const Optimizer& o,Module& m) {
 K y=nullptr; Ktag *k; TensorVector v;
 if(!(k=xtag(x))) {
  k=xtag(x,0);
  TORCH_CHECK(k, "opt: expecting module, model or tensor(s) as ",(i==1 ? "2nd" : "3rd")," arg, given ",kname(x));
  TORCH_CHECK(x->n==2,"opt: expecting one arg to index parameters from ",mapclass(k->a)," but given ",x->n-1," args");
  y=kK(x)[1];
 }
 switch(k->a) {
  case Class::tensor: v=vectorparms({((Kten*)k)->t}, y,o,m); break;
  case Class::vector: v=vectorparms(((Kvec*)k)->v, y,o,m); break;
  case Class::dict:   v=dictparms(((Kdict*)k)->d, y,o,m); break;
  case Class::module: v=moduleparms(mref((Kmodule*)k), y,o,m); break;
  case Class::model:  v=moduleparms(mref((Kmodel*)k), y,o,m); break;
  default: AT_ERROR("opt: cannot derive parameters from ",mapclass(k->a));
 }
 return v;
}

static void addoptions(Cast c,K x,ParamGroup& g) {
 switch(c) {
  case Cast::adagrad: adagrad(x,g); break;
  case Cast::adam:    adam<AdamOptions>(x,g); break;
  case Cast::adamw:   adam<AdamWOptions>(x,g); break;
  case Cast::lbfgs:   lbfgs(x,g); break;
  case Cast::rmsprop: rmsprop(x,g); break;
  case Cast::sgd:     sgd(x,g); break;
  default: AT_ERROR("unrecognized optimizer enumeration: ",(I)c);
 }
}

// ----------------------------------------------------------------------
// optinit - create new optimizer given parameter and option arg(s)
// optedit - edit existing optimizer group or add new group of parameters & options
// ----------------------------------------------------------------------
static K optinit(Cast c,J i,K x,K y) {
 BaseModule m; Optptr o; ParamGroup g({}); addoptions(c,y,g);
 switch(c) {
  case Cast::adagrad: o=std::make_shared<Adagrad>(ParamGroups{g}); break;
  case Cast::adam:    o=std::make_shared<Adam>   (ParamGroups{g}); break;
  case Cast::adamw:   o=std::make_shared<AdamW>  (ParamGroups{g}); break;
  case Cast::lbfgs:   o=std::make_shared<LBFGS>  (ParamGroups{g}); break;
  case Cast::rmsprop: o=std::make_shared<RMSprop>(ParamGroups{g}); break;
  case Cast::sgd:     o=std::make_shared<SGD>(ParamGroups{g},SGDOptions{LR}); break;
  default: AT_ERROR("opt: unrecognized optimizer enumeration: ",(I)c);
 }
 o->param_groups()[0].params()=addparms(x,i,*o,*m);
 return kopt(c,o,m);
}

static void optedit(K x,K y,J i,Cast c,Module& m,Optimizer* o) {
 auto& p=o->param_groups();
 if(i<p.size()) {
  auto v=moduleparms(m,x);
  // check parms ok..
  addoptions(c,y,p.at(i));
  p.at(i).params().insert(p.at(i).params().end(), v.begin(), v.end());
  /*
  auto g=p[i];
  addoptions(c,y,g);
  p[i].params()=g.params();
  p[i].set_options(g.options().clone());
  */
 } else {
  ParamGroup g(moduleparms(m,x)); addoptions(c,y,g);
  o->add_param_group(g);
 }
}

KAPI otest(K a,K x,K y) {
 KTRY
  TORCH_CHECK(a->t==-KS && x->t==0 && y->t==0, "bad args");
  Cast c=omap(a->s);
  return optinit(c,1,x,y);
 KCATCH("otest");
}
 
void gtest1(Optimizer *o) {
 std::cerr << (o ? "optimizer defined\n" : "NO optimizer defined\n");
}

KAPI gtest(K x) {
 KTRY
  J i=0;
  Optptr op; gtest1(op.get());
  Adam a({torch::rand({1,2})});
  gtest1(&a);
  ParamGroup g({torch::rand({1,3})});
  a.add_param_group(g);
  auto& p=a.param_groups();
  for(const auto& g:p) {
   std::cerr << "group: " << i << ", lr: " << static_cast<const AdamOptions&>(g.options()).lr() << "\n";
   for(const auto& t:g.params())
    std::cerr << "tensor: " << tensorlong(t,Attr::ptr) << ", " << t << "\n";
   i++;
  }
  ParamGroup h=p[1];
  h.params().emplace_back(torch::rand({1,4}));         //add parm(s)
  auto& o=getoptions<AdamOptions>(h); o.lr(1.234);     //modify option(s)
  p[1].params()=h.params();                            //rewrite previous group with new one
  p[1].set_options(h.options().clone());

  i=0; std::cerr <<"\n";
  for(const auto& g:p) {
   std::cerr << "group: " << i << ", lr: " << static_cast<const AdamOptions&>(g.options()).lr() << "\n";
   for(const auto& t:g.params())
    std::cerr << "tensor: " << tensorlong(t,Attr::ptr) << ", " << t << "\n";
   i++;
  }
  return (K)0;
 KCATCH("otest");
}
 
/*
std::vector<ParamGroup> optgroups(Cast c,K x) {
 std::vector<ParamGroup> v;
 TORCH_CHECK(x && x->n, "no options..");
 for(J i=0; i<x->n; ++i) {
  ParamGroup g({});
  addoptions(c,kK(x)[i],g);
  v.push_back(g);
 }
 return v;
}

optparms(Groups p,K x)
 TORCH_CHECK(x,"no parm table..");
 for(i..)
  g=
  TORCH_CHECK(-1<g && g<p.size(), "opt: parameter group [",g,"] is not defined");
  s=
  for(const auto &a:m.named_parameters())
   if(!a.key().compare(s)) {
    //check size..
    p[g].params().push_back(a.value());
   }
  }
 }
}
*/

// ---------------------------------------------------------------------------------------
// opt - main optimizer interface function for q
// optstep - recast underlying OptimizerBase to Optimizer unless LBFGS, run step() method
// kstep - given model or optimizer, perform an optimizer step unless closure required
// lr - query or set learning rate from k given pre-allocated optimizer ptr, or ptr & rate
// ---------------------------------------------------------------------------------------
KAPI opt(K x) {
 KTRY
  bool a=env().alloptions; S s; Kopt *o; Kmodel *m;
  if(xsym(x,s) || (xsym(x,0,s) && (xptr(x,1) || xempty(x,1)))) {
   return optinit(s,x);
  } else if(xdict(x)) {
   K y=stateoptlist(x);
   TORCH_CHECK(y && y->n, "opt: no options found");
   return optinit(statemodule(x), kK(y)[0]);
  } else if(xdict(x,0) && x->n==2 && (xptr(x,1) || xempty(x,1))) {
   K d=kK(x)[0];
   return optinit(statemodule(d),stateoptions(d),statebuffers(d));
  } else if((o=xoptim(x)) || (xbool(x,1,a) && (o=xoptim(x,0)) && x->n==2)) {
   return optstate(a,false,o->c,*o->o,*o->m);
  } else if((o=xoptim(x,0)) && xptr(x,1) && x->n==2) {
   return o->get()->add_parameters(optparms(x,1)), (K)0;
  } else if((m=xmodel(x))) {
   return kopt(m->oc,m->o,m->om);
  } else {
   AT_ERROR("unrecognized optimizer arg(s)");
  }
 KCATCH("optimizer error");
}

KAPI opt2(K x) {
 KTRY
  J g=0; bool a=env().alloptions,b=xlong(x,1,g); S s; Kopt *k;
  if(xsym(x,s) || xsym(x,0,s)) {
   TORCH_CHECK(x->n<4+b, "opt: ",s," optimizer definition expecting up to ",3+b,
                         " args(name;",(b? "group;" : ""),"parms;options) but ",x->n," given");
   //Cast c=omap(s); K y=(x->n>=2+b) ? kK(x)[1+b] : nullptr, z=(x->n==3+b) ? kK(x)[2+b] : nullptr;
   // c & z return an optimizer pointer
   //return kopt(..)   optinit(c,g,y,z);
  } else if(((k=xoptim(x))) || ((k=xoptim(x,0)))) {
   if(x->n==1 || (x->n==2 && xbool(x,1,a))) {
    return optstate(a,false,k->c,*k->o,*k->m);
   } else {
    //Cast c=omap(s); K y=(x->n>=2+b) ? kK(x)[1+b] : nullptr, z=(x->n==3+b) ? kK(x)[2+b] : nullptr;
    return (K)0;
   }
  } else if(xdict(x) || xdict(x,0)) {
   // initialize from dictonary and module(s)..?
   // count of groups: count of options list & highest group in k table
   // loop through options and create empty-vectored parameter group
   // 
  } else {
   AT_ERROR("opt: unrecognized arg(s)");
  }
  return(K)0;
 KCATCH("opt");
}

void optstep(Cast c,Optptr& o) {
 TORCH_CHECK(c != Cast::lbfgs, "LBFGS optimizer requires model, loss & inputs");
 ((Optimizer*)o.get())->step();
}

void optstep(Kopt   *o) {optstep(o->c, o->o);}
void optstep(Kmodel *m) {optstep(m->oc,m->o);}

KAPI kstep(K x) {
 KTRY
  if(auto* a=xoptim(x))
   optstep(a);
  else if(auto* a=xmodel(x))
   optstep(a);
  else
   AT_ERROR("step not implemented for ", kname(x));
  return (K)0;
 KCATCH("step");
}

// ---------------------------------------------------------------------------------------
// lrget - return a double list of learning rates, one per parameter group
// lrset - set each parameter group's learning rate from scalar/list input
// lr - function to query/set learning rate from k
// ---------------------------------------------------------------------------------------
static K lrget(const std::vector<ParamGroup>& v,Cast c) {
 J i=0; F r; K x=ktn(KF,v.size());
 for(auto& g:v) {
  TORCH_CHECK(g.has_options(), "parameter group options not defined");
  switch(c) {
   case Cast::adagrad: r=static_cast<const AdagradOptions&>(g.options()).lr(); break;
   case Cast::adam:    r=static_cast<const    AdamOptions&>(g.options()).lr(); break;
   case Cast::lbfgs:   r=static_cast<const   LBFGSOptions&>(g.options()).lr(); break;
   case Cast::rmsprop: r=static_cast<const RMSpropOptions&>(g.options()).lr(); break;
   case Cast::sgd:     r=static_cast<const     SGDOptions&>(g.options()).lr(); break;
   default: AT_ERROR("unrecognized optimizer: ",(I)c,", unable to retrieve learning rate");
  }
  kF(x)[i++]=r;
 }
 return x;
}

static void lrset(std::vector<ParamGroup>& v,Cast c,J n,double *lr) {
 TORCH_CHECK(n==1 || (unsigned)n==v.size(),"length error: ",n," learning rates given for ",v.size()," parameter group",(v.size() !=1 ? "s" : ""));
 int64_t i=0; double r;
 for(auto& g:v) {
  TORCH_CHECK(g.has_options(), "parameter group options not defined");
  r=(n==1) ? lr[0] : lr[i++];
  switch(c) {
   case Cast::adagrad: static_cast<AdagradOptions&>(g.options()).lr(r); break;
   case Cast::adam:    static_cast<   AdamOptions&>(g.options()).lr(r); break;
   case Cast::lbfgs:   static_cast<  LBFGSOptions&>(g.options()).lr(r); break;
   case Cast::rmsprop: static_cast<RMSpropOptions&>(g.options()).lr(r); break;
   case Cast::sgd:     static_cast<    SGDOptions&>(g.options()).lr(r); break;
   default: AT_ERROR("unrecognized optimizer: ",(I)c,", unable to set learning rate");
  }
 }
}

KAPI lr(K x) {
 KTRY
  bool b=false; J n; double *r; Ktag *g;
  TORCH_CHECK((g=xtag(x)) || ((g=xtag(x,0)) && (b=x->n==2) && xdouble(x,1,n,r)),
   "lr: unrecognized arg(s), expecting model/optimizer and optional learning rate(s) to set");
  Cast c; Optimizer *o; Kmodel *m;
  switch(g->a) {
   case Class::optimizer: c=g->c; o=((Kopt*)g)->o.get(); break;
   case Class::model: m=(Kmodel*)g; c=m->oc; o=m->o.get(); break;
   default: AT_ERROR("lr not implemented for ",mapclass(g->a));
  }
  if(b)
   return lrset(o->param_groups(),c,n,r), (K)0;
  else
   return lrget(o->param_groups(),c);
 KCATCH("lr");
}

// ---------------------------------------------------------------------------------------
// optattr - return attribute of given optimizer
// ---------------------------------------------------------------------------------------
K optattr(const Optptr& o,Ktype k,Attr a) {
 switch(a) {
  case Attr::ptr:  return kj((intptr_t)o.get());
  case Attr::ref:  return kj(o.use_count());
  case Attr::size: return kj(osize(o));
  default: AT_ERROR(mapattr(a),": not implemented for optimizers");
 }
}

K opthelp(Cast c) {
 switch(c) {
  case Cast::adagrad: return adagrad(true,AdagradOptions());
  case Cast::adam:    return adam(true,AdamOptions());
  case Cast::adamw:   return adam(true,AdamWOptions());
  case Cast::lbfgs:   return lbfgs(true,LBFGSOptions());
  case Cast::rmsprop: return rmsprop(true,RMSpropOptions());
  case Cast::sgd:     return sgd(true,SGDOptions(LR));

  case Cast::undefined: {
   const auto& e=env().opt; J i=0,n=e.size();
   K k=ktn(KS,3),s=ktn(KS,n),d=ktn(0,n),o=ktn(0,n);
   kS(k)[0]=cs("module"); kS(k)[1]=cs("pytorch"); kS(k)[2]=cs("options");
   for(auto& a:e) {
    kS(s)[i]=std::get<0>(a);
    kK(d)[i]=kp((S)std::get<2>(a).c_str());
    kK(o)[i]=opthelp(std::get<1>(a)); ++i;
   }
   return xT(xD(k,knk(3,s,d,o)));
  }
  default: AT_ERROR("no help implemented for optimizer enumeration: ",(I)c);
 }
}

// -------------------------------------------------------------------------------------------
// add optimizer api functions to library dictionary
// -------------------------------------------------------------------------------------------
void optfn(K x) {
 fn(x, "opt",  KFN(opt),1);
 fn(x, "opt2",  KFN(opt2),1);
 fn(x, "step", KFN(kstep),1);
 fn(x, "lr",   KFN(lr),1);
}
