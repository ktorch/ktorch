#include "ktorch.h"
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
K kopt(Cast x,const Optptr& y,const Moduleptr& z) {return kptr(new Kopt(x,y,z));}

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
// code  - check args for symbol, else error w'optimizer & setting name
// --------------------------------------------------------------------------------------
static S code(K x,J i,Cast c,Setting s) {
 S m;
 TORCH_CHECK(xsym(x,i,m), omap(c)," ",oset(s),": expected symbol, given ",kname(x,i));
 return m;
}

static S code(const Pairs& p,Cast c) {
 TORCH_CHECK(p.t==-KS, omap(c)," ",p.k,": expected symbol, given ",kname(p.t));
 return p.s;
}

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
  
// ----------------------------------------------------------------------------------------
// adagrad - parse args (lr;lrdecay;wtdecay) or (..;name/val pairs/dict)
//         - if given options,buffers, allocate new optimizer and return ptr
//         - if given previously allocated ptr, return dictionary of options & buffers
// ----------------------------------------------------------------------------------------
static void adagrad(K x,J i,ParamGroup& g) {
 auto& o=getoptions<AdagradOptions>(g); Pairs p; J n=xargc(x,i,p); double f;
 if(n && xnum(x,i,f)) {i++; n--; if(f==f) o.lr(f);}
 if(n && xnum(x,i,f)) {i++; n--; if(f==f) o.lr_decay(f);}
 if(n && xnum(x,i,f)) {i++; n--; if(f==f) o.weight_decay(f);}
 if(n && xnum(x,i,f)) {i++; n--; if(f==f) o.initial_accumulator_value(f);}
 if(n && xnum(x,i,f)) {i++; n--; if(f==f) o.eps(f);}
 if(n) AT_ERROR("opt: unrecognized option(s) for Adagrad optimizer");
 while(xpair(p))
  switch(oset(p.k)) {
   case Setting::lr:      f=pdouble(p); if(f==f) o.lr(f); break;
   case Setting::lrdecay: f=pdouble(p); if(f==f) o.lr_decay(f); break;
   case Setting::decay:   f=pdouble(p); if(f==f) o.weight_decay(f); break;
   case Setting::init:    f=pdouble(p); if(f==f) o.initial_accumulator_value(f); break;
   case Setting::eps:     f=pdouble(p); if(f==f) o.eps(f); break;
   default: AT_ERROR("unrecognized option: ",p.k," for Adagrad optimization"); break;
  }
}

static void adaput(K x,const Device& d,const std::string& k,Optimizer& o) {
 K v; auto s=std::make_unique<AdagradParamState>();
 if((v=findbuffer(x,"step",-KJ))) s->step(v->j);
 if((v=findbuffer(x,"sum")))      s->sum(kput(v).to(d));
 o.state()[k]=std::move(s);
}

static K adagrad(bool a,const AdagradOptions& o) { //return all or non-default options as k dictionary
 K x=xD(ktn(KS,0),ktn(0,0)); AdagradOptions d;
 if(a || d.lr()           != o.lr())           OPTSET(x, lr,      kf(o.lr()));
 if(a || d.lr_decay()     != o.lr_decay())     OPTSET(x, lrdecay, kf(o.lr_decay()));
 if(a || d.weight_decay() != o.weight_decay()) OPTSET(x, decay,   kf(o.weight_decay()));
 if(a || d.initial_accumulator_value() !=
         o.initial_accumulator_value())        OPTSET(x, init,    kf(o.initial_accumulator_value()));
 if(a || d.eps()          != o.eps())          OPTSET(x, eps,     kf(o.eps()));
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
template<typename O> static void adam(K x,J i,ParamGroup& g) {
 auto& o=getoptions<O>(g); Pairs p; J n=xargc(x,i,p); bool b; double f;
 if(n && xnum(x,i,f)) {i++; n--; if(f==f) o.lr(f);}
 if(n && xnum(x,i,f)) {i++; n--; if(f==f) o.betas(std::make_tuple(f,std::get<1>(o.betas())));}
 if(n && xnum(x,i,f)) {i++; n--; if(f==f) o.betas(std::make_tuple(std::get<0>(o.betas()),f));}
 if(n && xnum(x,i,f)) {i++; n--; if(f==f) o.eps(f);}
 if(n && xnum(x,i,f)) {i++; n--; if(f==f) o.weight_decay(f);}
 if(n && xbool(x,i,b)){i++; n--; o.amsgrad(b);}
 if(n) AT_ERROR("opt: unrecognized option(s) for Adam optimizer");
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

template<typename S>static void adamput(K x,const Device& d,const std::string& k,Optimizer& o) {
 K v; auto s=std::make_unique<S>();
 if((v=findbuffer(x,"step",-KJ)))       s->step(v->j);
 if((v=findbuffer(x,"exp_avg")))        s->exp_avg(kput(v).to(d));
 if((v=findbuffer(x,"exp_avg_sq")))     s->exp_avg_sq(kput(v).to(d));
 if((v=findbuffer(x,"max_exp_avg_sq"))) s->max_exp_avg_sq(kput(v).to(d));
 o.state()[k]=std::move(s);
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
static void lbfgs(K x,J i,ParamGroup& g) {
 auto& o=getoptions<LBFGSOptions>(g); Pairs p; J j,n=xargc(x,i,p); double f; S s;
 if(n && xnum(x,i,f))  {i++; n--; if(f==f)  o.lr(f);}
 if(n && xlong(x,i,j)) {i++; n--; if(j!=nj) o.max_iter(j);}
 if(n && xlong(x,i,j)) {i++; n--; if(j!=nj) o.max_eval(j);}
 if(n && xnum(x,i,f))  {i++; n--; if(f==f)  o.tolerance_grad(f);}
 if(n && xnum(x,i,f))  {i++; n--; if(f==f)  o.tolerance_change(f);}
 if(n && xlong(x,i,j)) {i++; n--; if(j!=nj) o.history_size(j);}
 if(n) {s=code(x,i,Cast::lbfgs,Setting::search); n--; i++; if(!nullsym(s)) o.line_search_fn(s);}
 if(n) AT_ERROR("opt: up to 7 positional args(s) for LBFGS optimizer, ",7+n," supplied");
 while(xpair(p))
  switch(oset(p.k)) {
   case Setting::lr:        f=pdouble(p); if(f==f)  o.lr(f); break;
   case Setting::iter:      j=plong(p);   if(j!=nj) o.max_iter(j); break;
   case Setting::eval:      j=plong(p);   if(j!=nj) o.max_eval(j); break;
   case Setting::gradtol:   f=pdouble(p); if(f==f)  o.tolerance_grad(f); break;
   case Setting::changetol: f=pdouble(p); if(f==f)  o.tolerance_change(f); break;
   case Setting::history:   j=plong(p);   if(j!=nj) o.history_size(j); break;
   case Setting::search:    s=code(p,Cast::lbfgs); if(!nullsym(s)) o.line_search_fn(s); break;
   default: AT_ERROR("unrecognized option: ",p.k," for LBFGS optimization"); break;
  }
 if(!o.max_eval()) o.max_eval((o.max_iter()*5)/4);
}

static void lbput(K x,const Device& d,const std::string& k,Optimizer& o) {
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

static K lbfgs(bool a,const LBFGSOptions& o) { //return all or non-default options as k dictionary
 K x=xD(ktn(KS,0),ktn(0,0)); LBFGSOptions d;
 if(a || d.lr()               != o.lr())               OPTSET(x, lr,        kf(o.lr()));
 if(a || d.max_iter()         != o.max_iter())         OPTSET(x, iter,      kj(o.max_iter()));
 if(a || o.max_eval())                                 OPTSET(x, eval,      kj(o.max_eval() ? *o.max_eval() : nj));
 if(a || d.tolerance_grad()   != o.tolerance_grad())   OPTSET(x, gradtol,   kf(o.tolerance_grad()));
 if(a || d.tolerance_change() != o.tolerance_change()) OPTSET(x, changetol, kf(o.tolerance_change()));
 if(a || d.history_size()     != o.history_size())     OPTSET(x, history,   kj(o.history_size()));
 if(o.line_search_fn().has_value())                    OPTSET(x, search,    ks(cs(o.line_search_fn().value().c_str())));
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

static K lbget(const LBFGSParamState& s) {
 K x=xD(ktn(KS,0),ktn(0,0));
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

// ----------------------------------------------------------------------------------------
// rmsprop - parse arg(s) (lr;alpha;eps;decay;momentum;centered) or (..;nm-val pairs/dict)
// ----------------------------------------------------------------------------------------
static void rmsprop(K x,J i,ParamGroup& g) {
 auto& o=getoptions<RMSpropOptions>(g); Pairs p; J n=xargc(x,i,p); bool b; double f;
 if(n && xnum(x,i,f)) {i++; n--; if(f==f) o.lr(f);}
 if(n && xnum(x,i,f)) {i++; n--; if(f==f) o.alpha(f);}
 if(n && xnum(x,i,f)) {i++; n--; if(f==f) o.eps(f);}
 if(n && xnum(x,i,f)) {i++; n--; if(f==f) o.weight_decay(f);}
 if(n && xnum(x,i,f)) {i++; n--; if(f==f) o.momentum(f);}
 if(n && xbool(x,i,b)){i++; n--; o.centered(b);}
 if(n) AT_ERROR("opt: unrecognized option(s) for RMSprop optimizer");
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

static void rmsput(K x,const Device& d,const std::string& k,Optimizer& o) {
 K v; auto s=std::make_unique<RMSpropParamState>();
 if((v=findbuffer(x,"step",-KJ)))        s->step(v->j);
 if((v=findbuffer(x,"square_avg")))      s->square_avg(kput(v).to(d));
 if((v=findbuffer(x,"momentum_buffer"))) s->momentum_buffer(kput(v).to(d));
 if((v=findbuffer(x,"grad_avg")))        s->grad_avg(kput(v).to(d));
 o.state()[k]=std::move(s);
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
static void sgd(K x,J i,ParamGroup& g) {
 auto& o=getoptions(g); Pairs p; J n=xargc(x,i,p); bool b; double f;
 if(n && xnum(x,i,f)) {i++; n--; if(f==f) o.lr(f);}
 if(n && xnum(x,i,f)) {i++; n--; if(f==f) o.momentum(f);}
 if(n && xnum(x,i,f)) {i++; n--; if(f==f) o.dampening(f);}
 if(n && xnum(x,i,f)) {i++; n--; if(f==f) o.weight_decay(f);}
 if(n && xbool(x,i,b)){i++; n--; o.nesterov(b);}
 if(n) AT_ERROR("opt: unrecognized option(s) for SGD optimizer");
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

static void sgdput(K x,const Device& d,const std::string& k,Optimizer& o) {
 K v; auto s=std::make_unique<SGDParamState>();
 if((v=findbuffer(x,"momentum_buffer"))) s->momentum_buffer(kput(v).to(d));
 o.state()[k]=std::move(s);
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
 dictadd(x, "momentum_buffer",  kget(s.momentum_buffer()));
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
  case Cast::lbfgs:   return   lbget(static_cast<const LBFGSParamState&>(p));
  case Cast::rmsprop: return   rmsget(static_cast<const RMSpropParamState&>(p));
  case Cast::sgd:     return   sgdget(static_cast<const SGDParamState&>(p));
  default: AT_ERROR("unrecognized optimizer: ",(I)c,", unable to retrieve parameter state");
 }
}

static K getparms(bool b,Cast c,const Optimizer& o,const Module& m) {
 J g=0,i=0,n=osize(o);
 K pt=ktn(KJ,n),gp=ktn(KJ,n),md=ktn(KS,n),nm=ktn(KS,n),sz=ktn(0,n),bf=nullptr; if(b) bf=ktn(0,n);
 const auto& s=o.state();
 for(auto& pg:o.param_groups()) {
  for(auto& p:pg.params()) {
    auto *t=p.unsafeGetTensorImpl();
    kJ(gp)[i]=g;
    kJ(pt)[i]=(intptr_t)t;
    kS(md)[i]=moduletype(p,m);
    kS(nm)[i]=parmsym(p,m);
    kK(sz)[i]=tensorsize(p,Attr::size);
    if(b) {
     auto k=c10::guts::to_string(t);
     kK(bf)[i]=s.count(k) ? getparms(c, *s.at(k)) : KDICT;
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
   for(auto& c:d->children())
    if((p=c->as<nn::ParameterDict>())) break;
 }
 return p;
}
 
// ---------------------------------------------------------------------------------------
// addname - name module "parms", if already found, try "parms1", "parms2", ..
// addtensor - add vector/dictionary of tensors to parameter dictionary in target module
// addvector - add list of tensors to parameter dictionary in target module
// adddict - add named tensors or subset of them to dictionary in target module
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
   AT_ERROR("opt: no module or parameter named `",k);
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
  AT_ERROR("opt: ",msym(*a)," module supplied with unrecognized ",kname(x)," selector(s)");
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
   AT_ERROR("opt: no dictionary parameter named `",k);
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
  AT_ERROR("opt: tensor dictionary supplied with unrecognized ",kname(x)," selector(s)");
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
  AT_ERROR("opt: tensor vector supplied with unrecognized ",kname(x)," selector(s)");
 parmcheck(o,*m,v); addvector(v,m);
 return v;
}
 
// ---------------------------------------------------------------------------------------
// optstate - return optimizer name & options and optionally, internal buffer values
// ---------------------------------------------------------------------------------------
static K optstate(bool a,bool b,Cast c,const Optimizer &o,const Module& m) {
 K k=ktn(KS,3),v=ktn(0,3);
 kS(k)[0]=statekey(State::optimizer); kK(v)[0]=ks(omap(c));
 kS(k)[1]=statekey(State::options);   kK(v)[1]=optdict(a,c,o);
 kS(k)[2]=statekey(State::parms);     kK(v)[2]=getparms(b,c,o,m);
 return xD(k,v);
}

K optstate(bool a,bool b,Kopt*   o) {return optstate(a,b,o->c, *o->o,*o->m);}
K optstate(bool a,bool b,Kmodel* m) {return optstate(a,b,m->oc,*m->o,*m->om);}

// ---------------------------------------------------------------------------------------
// addoptions - parse k args into optimizer-specific options stored in a parameter group
// addparms - return vector of parameters, checking for duplicates in existing groups
//            also updates module that tracks inputs to optimizer to recreate from state
// ---------------------------------------------------------------------------------------
static void addoptions(Cast c,K x,J i,ParamGroup& g) { // c:type, x-arg(s), i-offset into args
 switch(c) {
  case Cast::adagrad: adagrad(x,i,g); break;
  case Cast::adam:    adam<AdamOptions>(x,i,g); break;
  case Cast::adamw:   adam<AdamWOptions>(x,i,g); break;
  case Cast::lbfgs:   lbfgs(x,i,g); break;
  case Cast::rmsprop: rmsprop(x,i,g); break;
  case Cast::sgd:     sgd(x,i,g); break;
  default: AT_ERROR("unrecognized optimizer enumeration: ",(I)c);
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
   case Class::tensor: v=vectorparms({((Kten*)k)->t}, y,o,m); break;
   case Class::vector: v=vectorparms(((Kvec*)k)->v, y,o,m); break;
   case Class::dict:   v=dictparms(((Kdict*)k)->d, y,o,m); break;
   case Class::module: v=moduleparms(((Kmodule*)k)->m, y,o,m); break;
   case Class::model:  v=moduleparms(((Kmodel*) k)->m, y,o,m); break;
   default: AT_ERROR("opt: cannot derive parameters from ",mapclass(k->a));
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
  case Cast::adagrad: return std::make_shared<Adagrad>(ParamGroups{g});
  case Cast::adam:    return std::make_shared<Adam>   (ParamGroups{g});
  case Cast::adamw:   return std::make_shared<AdamW>  (ParamGroups{g});
  case Cast::lbfgs:   return std::make_shared<LBFGS>  (ParamGroups{g});
  case Cast::rmsprop: return std::make_shared<RMSprop>(ParamGroups{g});
  case Cast::sgd:     return std::make_shared<SGD>(ParamGroups{g},SGDOptions{LR});
  default: AT_ERROR("opt: unrecognized optimizer enumeration: ",(I)c);
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
  ParamGroup g({});                   // initialize empty group
  addoptions(c,x,2+b,g);              // define optimizer-specific options for the group
  if(y) g.params()=addparms(y,o,m);   // get parameters for new group
  o.add_param_group(g);               // add to optimizer
 } else {
  auto g=p[i];                        // create a copy of i'th group
  addoptions(c,x,2+b,g);              // [re]set optimizer-specific options for the group
  if(y) {                             // add parameters to the end of the group's tensors
   const auto& v=addparms(y,o,m);
   g.params().insert(g.params().end(),v.begin(),v.end());
  }
  p[i].params()=g.params();              // use parameters from edited group
  p[i].set_options(g.options().clone()); // and updated otions
 }
}

// ---------------------------------------------------------------------------------------
// putgroups - given type & list of options, add empty group(s), return optimizer
// putbuffers - put buffers in k dict -> optimizer's state for one parameter
// putparms - assign optimizer parameters into group(s), add parameter buffers
// optput - given optimizer state and module, recreate optimizer (w'buffer state if given)
// ---------------------------------------------------------------------------------------
static Optptr putgroups(Cast c,K x) {
 TORCH_CHECK(!x->t && x->n>0, "opt: unrecognized options dictionary list for ",omap(c)," optimizer");
 Optptr o;
 for(J i=0; i<x->n; ++i) {
  ParamGroup g({}); addoptions(c,kK(x)[i],0,g);
  if(!i) o=optinit(c,g);
  else   o->add_param_group(g);
 }
 return o;
}

static void putbuffers(Cast c,K x,const Device& d,const std::string& k,Optimizer& o) {
 switch(c) {
  case Cast::adagrad: adaput(x,d,k,o); break;
  case Cast::adam:    adamput<AdamParamState>(x,d,k,o); break;
  case Cast::adamw:   adamput<AdamWParamState>(x,d,k,o); break;
  case Cast::lbfgs:   lbput(x,d,k,o); break;
  case Cast::rmsprop: rmsput(x,d,k,o); break;
  case Cast::sgd:     sgdput(x,d,k,o); break;
  default: AT_ERROR("opt: unable to set buffers, unrecognized optimizer enumeration: ",(I)c);
 }
}

static void putparms(Cast c,K x,Optimizer& o,const Module& m) {
 J n=xlen(x); const auto& p=m.named_parameters(); auto& g=o.param_groups(); auto& s=o.state();
 for(J i=0; i<n; ++i) {
  S s1=statemodule(x,i),s2=statename(x,i); J j=stategroup(x,i); IntArrayRef sz;
  std::string nm=nullsym(s1) ? "tensor" : s1; nm+=" parameter `"; nm+=s2;
  TORCH_CHECK(-1<j && j<(J)g.size(), "opt: group[",j,"] for ",nm, " is invalid, ",g.size()," group(s) defined");
  TORCH_CHECK(p.contains(s2), "opt: unable to find ",nm);
  const auto& t=p[s2]; const auto& k=c10::guts::to_string(t.unsafeGetTensorImpl());
  TORCH_CHECK(s.count(k)==0, "opt: ",nm," is repeated");
  TORCH_CHECK(xsize(statesize(x,i),sz), "opt: unable to get size of ",nm);
  TORCH_CHECK(t.sizes()==sz, "opt: size mismatch for ",nm,", expected ",sz,", given ",t.sizes());
  g[j].params().push_back(t);
  K b=statebuffers(x,i);
  if(b && xlen(b))
   putbuffers(c,b,t.device(),k,o);
 }
}

static K optput(S s,K x,K y,const Moduleptr& m) { //s:optimizer name, x:options, y:parm table
 Cast c=omap(s); Optptr o=putgroups(c,x); putparms(c,y,*o,*m);
 return kopt(c,o,m);
}

// ---------------------------------------------------------------------------------------
// opt - main optimizer interface function for q
// optstep - recast underlying OptimizerBase to Optimizer unless LBFGS, run step() method
// kstep - given model or optimizer, perform an optimizer step unless closure required
// lr - query or set learning rate from k given pre-allocated optimizer ptr, or ptr & rate
// ---------------------------------------------------------------------------------------
KAPI opt(K x) {
 KTRY
  J i=0; bool a=env().alloptions,b=xlong(x,1,i); S s; Kopt *o; Kmodule *m; Kmodel *l;
  if(xsym(x,s) || (xsym(x,0,s) && x->t==0)) {
   J n=x->t==-KS ? 1 : x->n;
   TORCH_CHECK(!i, "opt: cannot define group ",i," until optimizer is created with initial parameter group");
   return optinit(b, omap(s), x, n>1+b ? kK(x)[1+b] : nullptr);
  } else if(((o=xoptim(x))) || ((o=xoptim(x,0)))) {
   if(x->n==1 || (x->n==2 && xbool(x,1,a))) {
    return optstate(a,false,o);
   } else {
    optedit(b, o->c, x, x->n>1+b ? kK(x)[1+b] : nullptr, i, *o->o, o->m);
    return (K)0;
   }
  } else if(xdict(x,0) && (m=xmodule(x,1)) && x->n==2) {
   return optput(stateoptimizer(kK(x)[0]), stateoptlist(kK(x)[0]), stategroups(kK(x)[0]), m->m);
  } else if((l=xmodel(x))) {
   return kopt(l->oc,l->o,l->m);
  } else {
   AT_ERROR("opt: unrecognized arg(s)");
  }
  return(K)0;
 KCATCH("opt");
}

void optstep(Cast c,Optptr& o) {
 TORCH_CHECK(c != Cast::lbfgs, "LBFGS optimizer requires model, loss & inputs");
 o->step();
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
   case Cast::adamw:   static_cast<  AdamWOptions&>(g.options()).lr(r); break;
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
  case Cast::lbfgs:   return lbfgs(true,LBFGSOptions().line_search_fn("strong_wolf"));
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
 fn(x, "step", KFN(kstep),1);
 fn(x, "lr",   KFN(lr),1);
}
